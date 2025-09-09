
"""
Gradio – Dictaphone/Uploader → Transcription (FFmpeg v8 Whisper) → Résumé CR via LM Studio (Gradio 4.x)

Correctifs qualité
------------------
- **Début manquant / silences** : sur certains fichiers, les premiers mots peuvent être mangés si le modèle démarre trop vite ou si le VAD est trop agressif.
  - Ajout d’un **pré‑roll** (padding de silence en tête) via `adelay=…` (par défaut 250 ms) pour capturer les débuts de phrase.
  - **VAD (facultatif)** : possibilité d’activer un modèle VAD Silero et d’ajuster seuil/durations. Par défaut **désactivé** pour ne pas couper les blancs.
  - Choix du **format de sortie** `text | srt | json` (par défaut `srt` pour vérifier visuellement les segments + horodatages).

Références
----------
- Options officielles du filtre `whisper` (modèle, destination, format, VAD : `vad_model`, `vad_threshold`, `vad_min_*`, `queue`). citeturn2view0
- Présentation des options `destination`, `format` et VAD dans des exemples récents. citeturn1search11turn0search0
- `adelay=…:all=1` (ajout de silence en tête sur tous les canaux). citeturn3search16

Prérequis
---------
- Python 3.10+
- `pip install gradio requests python-dotenv`
- FFmpeg 8.0+ **compilé avec `--enable-whisper`** + `whisper.cpp` présent
- LM Studio en mode serveur local (OpenAI-compatible) – http://localhost:1234

Lancement
---------
python gradio_transcripteur_compte_rendu.py

"""

import os
import json
import subprocess
from datetime import datetime
from typing import Tuple

import gradio as gr
import requests
from dotenv import load_dotenv

# ----------------------
# Config (.env facultatif)
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
# Utilitaires
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
    """Transcrit `audio_path` via FFmpeg+Whisper.
    Retourne (texte_ou_sous-titres, chemin_fichier_sortie_relatif).
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Fichier introuvable: {audio_path}")

    if not ffmpeg_has_whisper():
        raise RuntimeError("FFmpeg ne comporte pas le filtre 'whisper'. Build FFmpeg 8 avec --enable-whisper requis.")

    # Fichier de sortie selon format demandé
    ext = "txt" if fmt == "text" else ("srt" if fmt == "srt" else "json")
    out_rel = make_out_rel(ext)

    # Options whisper
    whisper_opts = [
        f"model={sanitize_path(WHISPER_MODEL_PATH)}",
        f"language={language}",
        f"destination={out_rel}",
        f"format={fmt}",
        f"queue={queue_ms}ms",
    ]

    # VAD facultatif (Silero). À activer uniquement si un modèle est fourni.
    if vad_enabled and vad_model_path:
        whisper_opts.append(f"vad_model={sanitize_path(vad_model_path)}")
        whisper_opts.append(f"vad_threshold={vad_threshold}")
        whisper_opts.append(f"vad_min_speech_duration={max(20, int(vad_min_speech_ms))}ms")
        whisper_opts.append(f"vad_min_silence_duration={max(0, int(vad_min_silence_ms))}ms")

    whisper_filter = "whisper=" + ":".join(whisper_opts)

    # Chaîne audio : pré‑roll (adelay) pour capturer le tout début + resample pour stabilité
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
            f"""Échec FFmpeg Whisper ({proc.returncode})
    STDERR:
    {proc.stderr}
    CMD:
    {' '.join(cmd)}"""
        )

    if not os.path.exists(out_rel):
        raise FileNotFoundError(
            f"""Sortie non trouvée: {out_rel}
    Vérifiez modèle/chemins et options du filtre."""
        )


    with open(out_rel, "r", encoding="utf-8") as f:
        text = f.read().strip()

    return text, out_rel


def call_lmstudio_summary(transcript: str) -> str:
    url = LMSTUDIO_BASE_URL.rstrip("/") + LMSTUDIO_API_PATH
    system_prompt = (
        "Tu es un assistant spécialisé en comptes rendus de réunion. "
        "À partir de la transcription (peut contenir des horodatages), génère un CR clair en français (Markdown) : Contexte, Ordre du jour, Décisions, Actions (Responsable→Action→Échéance), Prochaines étapes, Citations, Risques."
    )

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
# Callbacks Gradio
# ----------------------

def do_transcribe(audio_path: str, language: str, fmt: str, preroll_ms: int, vad_enabled: bool, vad_model_path: str, vad_threshold: float, vad_min_speech_ms: int, vad_min_silence_ms: int, queue_ms: int):
    if not audio_path:
        return "Aucun fichier fourni.", "", ""
    text, out_rel = run_ffmpeg_whisper_transcribe(
        audio_path=audio_path,
        language=language or WHISPER_LANGUAGE,
        fmt=fmt,
        preroll_ms=preroll_ms,
        vad_enabled=vad_enabled,
        vad_model_path=vad_model_path,
        vad_threshold=vad_threshold,
        vad_min_speech_ms=vad_min_speech_ms,
        vad_min_silence_ms=vad_min_silence_ms,
        queue_ms=queue_ms,
    )
    return f"Transcription OK ({len(text)} caractères)", text, out_rel


def do_summarize(transcript: str):
    if not transcript or not transcript.strip():
        return "Pas de transcription à résumer.", ""
    md = call_lmstudio_summary(transcript)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"Résumé généré ({len(md)} caractères) – {stamp}", md

# ----------------------
# UI (Gradio 4.x)
# ----------------------
with gr.Blocks(title="Transcripteur → CR Réunion (FFmpeg Whisper)") as demo:
    gr.Markdown(
        """
        # 🎙️ Transcripteur → CR de réunion
        1) **Enregistrer / Uploader** un audio (WAV/MP3/M4A).  
        2) **Transcrire** (FFmpeg v8 + Whisper) – format par défaut **SRT** pour voir les horodatages.  
        3) **Résumer** (LM Studio) → **CR Markdown**.
        
        *Astuce* : si le **début est tronqué**, augmentez le **pré‑roll** (ex. 300–500 ms) ou laissez le **VAD désactivé**.
        """
    )

    with gr.Row():
        audio = gr.Audio(type="filepath", label="🎙️ Enregistrer ou Uploader")
        lang = gr.Textbox(label="Langue Whisper", value=WHISPER_LANGUAGE)
        fmt = gr.Dropdown(choices=["text", "srt", "json"], value="srt", label="Format sortie")

    with gr.Accordion("Paramètres avancés", open=False):
        preroll = gr.Slider(0, 2000, value=250, step=50, label="Pré‑roll (ms) – padding début")
        queue = gr.Slider(200, 20000, value=3000, step=100, label="Taille de file (queue) pour VAD (ms)")
        vad_enable = gr.Checkbox(False, label="Activer VAD (Silero) – peut couper les blancs si mal réglé")
        with gr.Row():
            vad_model = gr.Textbox(label="Chemin modèle VAD (ex: ./models/silero-v5.1.2-ggml.bin)", value="")
            vad_thr = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Seuil VAD")
        with gr.Row():
            vad_min_speech = gr.Slider(20, 2000, value=100, step=20, label="Durée minimale parole (ms)")
            vad_min_silence = gr.Slider(0, 2000, value=500, step=20, label="Durée minimale silence (ms)")

    btn_transcribe = gr.Button("📝 Transcrire")
    status_trans = gr.Textbox(label="Statut transcription", interactive=False)
    transcript = gr.Textbox(label="Transcription / SRT / JSON", lines=16)
    transcript_file = gr.Textbox(label="Fichier généré", interactive=False)

    gr.Markdown("---")

    btn_summarize = gr.Button("🧾 Résumer → CR Markdown")
    status_sum = gr.Textbox(label="Statut résumé", interactive=False)
    summary_md = gr.Markdown("(le CR apparaîtra ici)")

    # Wiring
    btn_transcribe.click(
        fn=do_transcribe,
        inputs=[audio, lang, fmt, preroll, vad_enable, vad_model, vad_thr, vad_min_speech, vad_min_silence, queue],
        outputs=[status_trans, transcript, transcript_file],
    )
    btn_summarize.click(fn=do_summarize, inputs=[transcript], outputs=[status_sum, summary_md])

if __name__ == "__main__":
    demo.launch()
