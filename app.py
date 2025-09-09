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
from typing import Tuple

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


def call_lmstudio_summary(transcript: str) -> str:
    url = LMSTUDIO_BASE_URL.rstrip("/") + LMSTUDIO_API_PATH
    system_prompt = (
        "You are an assistant specialized in meeting minutes. "
        "From the transcript (may contain timestamps), generate a clear report in French (Markdown) with: Context, Agenda, Decisions, Actions (Owner→Action→Deadline), Next steps, Quotes, Risks."
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

def do_summarize(transcript: str):
    try:
        if not transcript or not transcript.strip():
            return "No transcription to summarize.", ""
        md = call_lmstudio_summary(transcript)
        return f"Summary OK ({len(md)} chars)", md
    except Exception as e:
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return f"⚠️ Summary error: {tb}", ""


# ----------------------
# UI (Gradio 4.x)
# ----------------------
with gr.Blocks(title="Transcriber → Meeting Minutes (FFmpeg Whisper)") as demo:
    gr.Markdown(
        """
        # 🎙️ Transcriber → Meeting Minutes
        1) **Record / Upload** an audio (WAV/MP3/M4A).  
        2) **Transcribe** (FFmpeg v8 + Whisper) – default format **SRT** to see timestamps.  
        3) **Summarize** (LM Studio) → **Markdown report**.
        
        *Tip*: if the **start is truncated**, increase **pre‑roll** (e.g., 300–500 ms) or keep **VAD disabled**.
        """
    )

    with gr.Row():
        audio = gr.Audio(type="filepath", label="🎙️ Record or Upload")
        lang = gr.Textbox(label="Whisper Language", value=WHISPER_LANGUAGE)
        fmt = gr.Dropdown(choices=["text", "srt", "json"], value="srt", label="Output format")

    with gr.Accordion("Advanced settings", open=False):
        preroll = gr.Slider(0, 2000, value=250, step=50, label="Pre‑roll (ms) – start padding")
        queue = gr.Slider(200, 20000, value=3000, step=100, label="Queue size for VAD (ms)")
        vad_enable = gr.Checkbox(False, label="Enable VAD (Silero) – may cut silences if misconfigured")
        with gr.Row():
            vad_model = gr.Textbox(label="VAD model path (e.g., ./models/silero-v5.1.2-ggml.bin)", value="")
            vad_thr = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="VAD threshold")
        with gr.Row():
            vad_min_speech = gr.Slider(20, 2000, value=100, step=20, label="Minimum speech duration (ms)")
            vad_min_silence = gr.Slider(0, 2000, value=500, step=20, label="Minimum silence duration (ms)")

    btn_transcribe = gr.Button("📝 Transcribe")
    status_trans = gr.Textbox(label="Transcription status", interactive=False)
    transcript = gr.Textbox(label="Transcription / SRT / JSON", lines=16)
    transcript_file = gr.Textbox(label="Generated file", interactive=False)

    gr.Markdown("---")

    btn_summarize = gr.Button("🧾 Summarize → Markdown report")
    status_sum = gr.Textbox(label="Summary status", interactive=False)
    summary_md = gr.Markdown("(the report will appear here)")

    # Wiring
    btn_transcribe.click(
        fn=do_transcribe,
        inputs=[audio, lang, fmt, preroll, vad_enable, vad_model, vad_thr, vad_min_speech, vad_min_silence, queue],
        outputs=[status_trans, transcript, transcript_file],
    )
    btn_summarize.click(fn=do_summarize, inputs=[transcript], outputs=[status_sum, summary_md])

if __name__ == "__main__":
    demo.launch()
