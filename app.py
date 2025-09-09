"""
Gradio – Dictaphone/Uploader → Transcription (FFmpeg v8 Whisper) → Meeting Minutes Summary via LM Studio (Gradio 4.x)

Quality Fixes
-------------
- **Missing start / silences**: on some files, the first words can be cut if the model starts too quickly or if VAD is too aggressive.
  - Added a **pre‑roll** (head silence padding) via `adelay=…` (default 250 ms) to capture sentence beginnings.
  - **VAD (optional)**: possibility to enable a Silero VAD model and adjust threshold/durations. Default is **disabled** to avoid cutting silences.
  - Choice of **output format** `text | srt | json` (default `srt` to visually check segments + timestamps).

References
----------
- Official options for the `whisper` filter (model, destination, format, VAD: `vad_model`, `vad_threshold`, `vad_min_*`, `queue`).
- Presentation of the `destination`, `format`, and VAD options in recent examples.
- `adelay=…:all=1` (adds silence at the start of all channels).

Requirements
------------
- Python 3.10+
- `pip install gradio requests python-dotenv`
- FFmpeg 8.0+ **compiled with `--enable-whisper`** + `whisper.cpp` available
- LM Studio running in local server mode (OpenAI-compatible) – http://localhost:1234

Run
---
python gradio_transcripteur_compte_rendu.py

"""

import os
import json
import subprocess
from datetime import datetime
from typing import Tuple, Dict

import gradio as gr
import requests
from dotenv import load_dotenv

# ----------------------
# Config (.env optional)
# ----------------------
load_dotenv()
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")
LMSTUDIO_API_PATH = os.getenv("LMSTUDIO_API_PATH", "/v1/chat/completions")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "Qwen2.5-7B-Instruct")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "./models/ggml-large-v3-turbo.bin")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "fr")

os.makedirs("transcripts", exist_ok=True)

# ----------------------
# Prompt templates
# ----------------------

TEMPLATES_PATH = "prompt_templates.json"
DEFAULT_TEMPLATE_NAME = "default"
DEFAULT_PROMPT = (
    "You are an assistant specialized in meeting minutes. "
    "From the transcript (may contain timestamps), generate a clear report in French (Markdown) with: Context, Agenda, Decisions, Actions (Owner→Action→Deadline), Next steps, Quotes, Risks."
)


def _ensure_templates_file() -> Dict[str, str]:
    if not os.path.exists(TEMPLATES_PATH):
        with open(TEMPLATES_PATH, "w", encoding="utf-8") as f:
            json.dump({DEFAULT_TEMPLATE_NAME: DEFAULT_PROMPT}, f, ensure_ascii=False, indent=2)
    with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_templates() -> Dict[str, str]:
    return _ensure_templates_file()


def save_templates(tpls: Dict[str, str]):
    with open(TEMPLATES_PATH, "w", encoding="utf-8") as f:
        json.dump(tpls, f, ensure_ascii=False, indent=2)


TEMPLATES = load_templates()


def get_template(name: str) -> str:
    return TEMPLATES.get(name, DEFAULT_PROMPT)


def list_templates():
    return sorted(TEMPLATES.keys())


def save_template(name: str, content: str):
    TEMPLATES[name] = content
    save_templates(TEMPLATES)


def delete_template(name: str):
    if name == DEFAULT_TEMPLATE_NAME:
        return
    if name in TEMPLATES:
        del TEMPLATES[name]
        save_templates(TEMPLATES)

# ----------------------
# UI settings & strings
# ----------------------

UI_SETTINGS_PATH = "ui_settings.json"
DEFAULT_UI_SETTINGS = {"language": "en", "theme": "light"}


def load_ui_settings() -> Dict[str, str]:
    if not os.path.exists(UI_SETTINGS_PATH):
        with open(UI_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_UI_SETTINGS, f, indent=2)
    with open(UI_SETTINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ui_settings(language: str, theme: str):
    with open(UI_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump({"language": language, "theme": theme}, f, indent=2)


UI_SETTINGS = load_ui_settings()
CURRENT_LANGUAGE = UI_SETTINGS.get("language", "en")
CURRENT_THEME = UI_SETTINGS.get("theme", "light")


STRINGS = {
    "en": {
        "intro_md": (
            """
# 🎙️ Transcriber → Meeting Minutes
1) **Record / Upload** an audio (WAV/MP3/M4A).
2) **Transcribe** (FFmpeg v8 + Whisper) – default format **SRT** to see timestamps.
3) **Summarize** (LM Studio) → **Markdown report**.

*Tip*: if the **start is truncated**, increase **pre‑roll** (e.g., 300–500 ms) or keep **VAD disabled**.
"""
        ),
        "tab_transcribe": "Transcribe / Summary",
        "tab_templates": "Templates",
        "tab_options": "Options",
        "audio_label": "🎙️ Record or Upload",
        "whisper_language": "Whisper Language",
        "output_format": "Output format",
        "advanced_settings": "Advanced settings",
        "preroll_label": "Pre‑roll (ms) – start padding",
        "queue_label": "Queue size for VAD (ms)",
        "vad_enable": "Enable VAD (Silero) – may cut silences if misconfigured",
        "vad_model": "VAD model path (e.g., ./models/silero-v5.1.2-ggml.bin)",
        "vad_thr": "VAD threshold",
        "vad_min_speech": "Minimum speech duration (ms)",
        "vad_min_silence": "Minimum silence duration (ms)",
        "btn_transcribe": "📝 Transcribe",
        "transcription_status": "Transcription status",
        "transcript_label": "Transcription / SRT / JSON",
        "generated_file": "Generated file",
        "template_selector": "Template",
        "btn_summarize": "🧾 Summarize → Markdown report",
        "summary_status": "Summary status",
        "summary_md": "(the report will appear here)",
        "template_name": "Template name",
        "template_content": "Template content",
        "btn_template_save": "💾 Save template",
        "btn_template_delete": "🗑️ Delete template",
        "template_status": "Template status",
        "options_language": "Interface Language",
        "options_theme": "Theme",
        "btn_save_options": "Save options",
        "options_status": "Options status",
        "options_saved": "Options saved. Reload page to apply theme.",
    },
    "fr": {
        "intro_md": (
            """
# 🎙️ Transcripteur → Compte-rendu
1) **Enregistrer / Téléverser** un audio (WAV/MP3/M4A).
2) **Transcrire** (FFmpeg v8 + Whisper) – format par défaut **SRT** pour voir les horodatages.
3) **Résumer** (LM Studio) → **rapport Markdown**.

*Astuce* : si le **début est tronqué**, augmentez le **pré‑roll** (ex. 300–500 ms) ou laissez le **VAD désactivé**.
"""
        ),
        "tab_transcribe": "Transcrire / Résumé",
        "tab_templates": "Modèles",
        "tab_options": "Options",
        "audio_label": "🎙️ Enregistrer ou Téléverser",
        "whisper_language": "Langue Whisper",
        "output_format": "Format de sortie",
        "advanced_settings": "Paramètres avancés",
        "preroll_label": "Pré‑roll (ms) – silence de début",
        "queue_label": "Taille de file pour VAD (ms)",
        "vad_enable": "Activer VAD (Silero) – peut couper les silences si mal configuré",
        "vad_model": "Chemin du modèle VAD (ex: ./models/silero-v5.1.2-ggml.bin)",
        "vad_thr": "Seuil VAD",
        "vad_min_speech": "Durée minimale de parole (ms)",
        "vad_min_silence": "Durée minimale de silence (ms)",
        "btn_transcribe": "📝 Transcrire",
        "transcription_status": "Statut de transcription",
        "transcript_label": "Transcription / SRT / JSON",
        "generated_file": "Fichier généré",
        "template_selector": "Modèle",
        "btn_summarize": "🧾 Résumer → Rapport Markdown",
        "summary_status": "Statut du résumé",
        "summary_md": "(le rapport apparaîtra ici)",
        "template_name": "Nom du modèle",
        "template_content": "Contenu du modèle",
        "btn_template_save": "💾 Enregistrer le modèle",
        "btn_template_delete": "🗑️ Supprimer le modèle",
        "template_status": "Statut du modèle",
        "options_language": "Langue de l'interface",
        "options_theme": "Thème",
        "btn_save_options": "Enregistrer options",
        "options_status": "Statut des options",
        "options_saved": "Options enregistrées. Recharger la page pour appliquer le thème.",
    },
}


THEMES = {
    "light": gr.themes.Soft(),
    "dark": gr.themes.Monochrome(),
}

# ----------------------
# Utilities
# ----------------------

def ffmpeg_has_whisper() -> bool:
    try:
        proc = subprocess.run([FFMPEG_BIN, "-hide_banner", "-filters"], capture_output=True, text=True, check=True)
        return "whisper" in proc.stdout
    except Exception:
        return False


def make_out_rel(suffix: str = "txt") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    rel = f"transcripts/transcript_{stamp}.{suffix}"
    return rel.replace("\\", "/")


def sanitize_path(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def run_ffmpeg_whisper_transcribe(
    audio_path: str,
    language: str = WHISPER_LANGUAGE,
    fmt: str = "srt",
    preroll_ms: int = 250,
    vad_enabled: bool = False,
    vad_model_path: str = "",
    vad_threshold: float = 0.5,
    vad_min_speech_ms: int = 100,
    vad_min_silence_ms: int = 500,
    queue_ms: int = 3000,
) -> Tuple[str, str]:
    """Transcribe `audio_path` via FFmpeg+Whisper.
    Returns (text_or_subtitles, relative_output_file_path).
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    if not ffmpeg_has_whisper():
        raise RuntimeError("FFmpeg does not include the 'whisper' filter. Build FFmpeg 8 with --enable-whisper required.")

    # Output file depending on requested format
    ext = "txt" if fmt == "text" else ("srt" if fmt == "srt" else "json")
    out_rel = make_out_rel(ext)

    # Whisper options
    whisper_opts = [
        f"model={sanitize_path(WHISPER_MODEL_PATH)}",
        f"language={language}",
        f"destination={out_rel}",
        f"format={fmt}",
        f"queue={queue_ms}ms",
    ]

    # Optional VAD (Silero). Activate only if a model is provided.
    if vad_enabled and vad_model_path:
        whisper_opts.append(f"vad_model={sanitize_path(vad_model_path)}")
        whisper_opts.append(f"vad_threshold={vad_threshold}")
        whisper_opts.append(f"vad_min_speech_duration={max(20, int(vad_min_speech_ms))}ms")
        whisper_opts.append(f"vad_min_silence_duration={max(0, int(vad_min_silence_ms))}ms")

    whisper_filter = "whisper=" + ":".join(whisper_opts)

    # Audio chain: pre‑roll (adelay) to capture the very beginning + resample for stability
    af_chain = []
    if preroll_ms and preroll_ms > 0:
        af_chain.append(f"adelay={int(preroll_ms)}:all=1")
    af_chain.append("aresample=16000")
    af_chain.append(whisper_filter)

    af_str = ",".join(af_chain)

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-hide_banner",
        "-i",
        audio_path,
        "-vn",
        "-af",
        af_str,
        "-f",
        "null",
        "-",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"""FFmpeg Whisper failed ({proc.returncode})
    STDERR:
    {proc.stderr}
    CMD:
    {' '.join(cmd)}"""
        )

    if not os.path.exists(out_rel):
        raise FileNotFoundError(
            f"""Output not found: {out_rel}
    Check model/paths and filter options."""
        )

    with open(out_rel, "r", encoding="utf-8") as f:
        text = f.read().strip()

    return text, out_rel


def call_lmstudio_summary(transcript: str, system_prompt: str) -> str:
    url = LMSTUDIO_BASE_URL.rstrip("/") + LMSTUDIO_API_PATH
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]

    payload = {"model": LMSTUDIO_MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 1800}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LMSTUDIO_API_KEY}"}

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

# ----------------------
# Gradio Callbacks
# ----------------------

import traceback

def do_transcribe(audio_path, language, fmt, preroll_ms, vad_enabled, vad_model_path, vad_threshold, vad_min_speech_ms, vad_min_silence_ms, queue_ms):
    try:
        if not audio_path or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            return "No or empty file. Please record/upload again.", "", ""
        text, out_rel = run_ffmpeg_whisper_transcribe(
            audio_path=audio_path, language=language, fmt=fmt, preroll_ms=preroll_ms,
            vad_enabled=vad_enabled, vad_model_path=vad_model_path, vad_threshold=vad_threshold,
            vad_min_speech_ms=vad_min_speech_ms, vad_min_silence_ms=vad_min_silence_ms, queue_ms=queue_ms
        )
        return f"Transcription OK ({len(text)} chars)", text, out_rel
    except Exception as e:
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return f"⚠️ Transcription error: {tb}", "", ""

def do_summarize(transcript: str, template_name: str):
    try:
        if not transcript or not transcript.strip():
            return "No transcription to summarize.", ""
        system_prompt = get_template(template_name)
        md = call_lmstudio_summary(transcript, system_prompt)
        return f"Summary OK ({len(md)} chars)", md
    except Exception as e:
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return f"⚠️ Summary error: {tb}", ""


# ----------------------
# UI (Gradio 4.x)
# ----------------------
theme_obj = THEMES.get(CURRENT_THEME, gr.themes.Soft())
strings = STRINGS[CURRENT_LANGUAGE]
with gr.Blocks(title="Transcriber → Meeting Minutes (FFmpeg Whisper)", theme=theme_obj) as demo:
    intro_md = gr.Markdown(strings["intro_md"])
    with gr.Tabs():
        with gr.TabItem(strings["tab_transcribe"]) as tab_trans:
            with gr.Row():
                audio = gr.Audio(type="filepath", label=strings["audio_label"])
                lang = gr.Textbox(label=strings["whisper_language"], value=WHISPER_LANGUAGE)
                fmt = gr.Dropdown(choices=["text", "srt", "json"], value="srt", label=strings["output_format"])
            with gr.Accordion(strings["advanced_settings"], open=False) as acc_adv:
                preroll = gr.Slider(0, 2000, value=250, step=50, label=strings["preroll_label"])
                queue = gr.Slider(200, 20000, value=3000, step=100, label=strings["queue_label"])
                vad_enable = gr.Checkbox(False, label=strings["vad_enable"])
                with gr.Row():
                    vad_model = gr.Textbox(label=strings["vad_model"], value="")
                    vad_thr = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label=strings["vad_thr"])
                with gr.Row():
                    vad_min_speech = gr.Slider(20, 2000, value=100, step=20, label=strings["vad_min_speech"])
                    vad_min_silence = gr.Slider(0, 2000, value=500, step=20, label=strings["vad_min_silence"])
            btn_transcribe = gr.Button(strings["btn_transcribe"])
            status_trans = gr.Textbox(label=strings["transcription_status"], interactive=False)
            transcript = gr.Textbox(label=strings["transcript_label"], lines=16)
            transcript_file = gr.Textbox(label=strings["generated_file"], interactive=False)
            template_use = gr.Dropdown(list_templates(), value=DEFAULT_TEMPLATE_NAME, label=strings["template_selector"])
            btn_summarize = gr.Button(strings["btn_summarize"])
            status_sum = gr.Textbox(label=strings["summary_status"], interactive=False)
            summary_md = gr.Markdown(strings["summary_md"])
        with gr.TabItem(strings["tab_templates"]) as tab_templates:
            tpl_dropdown = gr.Dropdown(list_templates(), value=DEFAULT_TEMPLATE_NAME, label=strings["template_selector"])
            tpl_name = gr.Textbox(value=DEFAULT_TEMPLATE_NAME, label=strings["template_name"])
            tpl_content = gr.Textbox(value=get_template(DEFAULT_TEMPLATE_NAME), lines=6, label=strings["template_content"])
            with gr.Row():
                btn_tpl_save = gr.Button(strings["btn_template_save"])
                btn_tpl_delete = gr.Button(strings["btn_template_delete"])
            tpl_status = gr.Textbox(label=strings["template_status"], interactive=False)
        with gr.TabItem(strings["tab_options"]) as tab_options:
            lang_selector = gr.Dropdown(choices=["en", "fr"], value=CURRENT_LANGUAGE, label=strings["options_language"])
            theme_selector = gr.Dropdown(choices=["light", "dark"], value=CURRENT_THEME, label=strings["options_theme"])
            btn_options_save = gr.Button(strings["btn_save_options"])
            options_status = gr.Textbox(label=strings["options_status"], interactive=False)

    # Wiring
    btn_transcribe.click(
        fn=do_transcribe,
        inputs=[audio, lang, fmt, preroll, vad_enable, vad_model, vad_thr, vad_min_speech, vad_min_silence, queue],
        outputs=[status_trans, transcript, transcript_file],
    )

    btn_summarize.click(
        fn=do_summarize,
        inputs=[transcript, template_use],
        outputs=[status_sum, summary_md],
    )

    def _load_template(name):
        return name, get_template(name)

    tpl_dropdown.change(
        fn=_load_template,
        inputs=[tpl_dropdown],
        outputs=[tpl_name, tpl_content],
    )

    def _save_template(name, content):
        save_template(name, content)
        choices = list_templates()
        return (
            f"Template '{name}' saved.",
            gr.Dropdown.update(choices=choices, value=name),
            gr.Dropdown.update(choices=choices, value=name),
        )

    btn_tpl_save.click(
        fn=_save_template,
        inputs=[tpl_name, tpl_content],
        outputs=[tpl_status, tpl_dropdown, template_use],
    )

    def _delete_template(name):
        if name == DEFAULT_TEMPLATE_NAME:
            choices = list_templates()
            return (
                "Cannot delete default template.",
                gr.Dropdown.update(choices=choices, value=DEFAULT_TEMPLATE_NAME),
                DEFAULT_TEMPLATE_NAME,
                get_template(DEFAULT_TEMPLATE_NAME),
                gr.Dropdown.update(choices=choices, value=DEFAULT_TEMPLATE_NAME),
            )
        delete_template(name)
        choices = list_templates()
        new_default = DEFAULT_TEMPLATE_NAME
        return (
            f"Template '{name}' deleted.",
            gr.Dropdown.update(choices=choices, value=new_default),
            new_default,
            get_template(new_default),
            gr.Dropdown.update(choices=choices, value=new_default),
        )

    btn_tpl_delete.click(
        fn=_delete_template,
        inputs=[tpl_dropdown],
        outputs=[tpl_status, tpl_dropdown, tpl_name, tpl_content, template_use],
    )

    def change_language(lang_choice):
        s = STRINGS[lang_choice]
        return (
            gr.update(label=s["tab_transcribe"]),
            gr.update(label=s["tab_templates"]),
            gr.update(label=s["tab_options"]),
            intro_md.update(value=s["intro_md"]),
            audio.update(label=s["audio_label"]),
            lang.update(label=s["whisper_language"]),
            fmt.update(label=s["output_format"]),
            acc_adv.update(label=s["advanced_settings"]),
            preroll.update(label=s["preroll_label"]),
            queue.update(label=s["queue_label"]),
            vad_enable.update(label=s["vad_enable"]),
            vad_model.update(label=s["vad_model"]),
            vad_thr.update(label=s["vad_thr"]),
            vad_min_speech.update(label=s["vad_min_speech"]),
            vad_min_silence.update(label=s["vad_min_silence"]),
            btn_transcribe.update(value=s["btn_transcribe"]),
            status_trans.update(label=s["transcription_status"]),
            transcript.update(label=s["transcript_label"]),
            transcript_file.update(label=s["generated_file"]),
            template_use.update(label=s["template_selector"]),
            btn_summarize.update(value=s["btn_summarize"]),
            status_sum.update(label=s["summary_status"]),
            summary_md.update(value=s["summary_md"]),
            tpl_dropdown.update(label=s["template_selector"]),
            tpl_name.update(label=s["template_name"]),
            tpl_content.update(label=s["template_content"]),
            btn_tpl_save.update(value=s["btn_template_save"]),
            btn_tpl_delete.update(value=s["btn_template_delete"]),
            tpl_status.update(label=s["template_status"]),
            lang_selector.update(label=s["options_language"]),
            theme_selector.update(label=s["options_theme"]),
            btn_options_save.update(value=s["btn_save_options"]),
            options_status.update(label=s["options_status"]),
        )

    lang_selector.change(
        fn=change_language,
        inputs=[lang_selector],
        outputs=[
            tab_trans,
            tab_templates,
            tab_options,
            intro_md,
            audio,
            lang,
            fmt,
            acc_adv,
            preroll,
            queue,
            vad_enable,
            vad_model,
            vad_thr,
            vad_min_speech,
            vad_min_silence,
            btn_transcribe,
            status_trans,
            transcript,
            transcript_file,
            template_use,
            btn_summarize,
            status_sum,
            summary_md,
            tpl_dropdown,
            tpl_name,
            tpl_content,
            btn_tpl_save,
            btn_tpl_delete,
            tpl_status,
            lang_selector,
            theme_selector,
            btn_options_save,
            options_status,
        ],
    )

    def save_options(lang_choice, theme_choice):
        save_ui_settings(lang_choice, theme_choice)
        return STRINGS[lang_choice]["options_saved"]

    btn_options_save.click(
        fn=save_options,
        inputs=[lang_selector, theme_selector],
        outputs=[options_status],
    )

if __name__ == "__main__":
    demo.launch()
