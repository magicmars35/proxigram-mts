# Gradio Meeting Transcriber & Summarizer (FFmpeg v8 Whisper + LM Studio)

A tiny Gradio app that records **from your browser microphone** or **uploads an audio file**, transcribes it **locally** using **FFmpeg v8 + the `whisper` filter** (with Whisper.cpp models), and sends the transcript to a **local LLM (LM Studio)** to produce a clean, structured **meeting minutes** (Markdown).

> **Highlights**
>
> * One-click **record or upload** (Gradio 4.x `Audio`).
> * Robust capture: **pre‑roll** (to avoid missing the first words), optional **VAD**, and **input normalization** to WAV 16 kHz mono.
> * Flexible output: **text / srt / json**.
> * Built‑in **prompt template CRUD** persisted to `prompt_templates.json` and used as **system prompt** for LM Studio.
> * Persistent **UI options** (language 🇬🇧/🇫🇷 and light/dark theme) saved to `ui_settings.json`.

---

## Architecture (at a glance)

1. **Gradio UI** (microphone/upload → `Audio` component).
2. **FFmpeg 8 + `whisper` filter** (with Whisper.cpp **ggml** model) → transcript file.

   * Optional **pre‑roll** (adds a short silence at the start) and **VAD**.
   * Audio is normalized to **WAV 16 kHz mono** for reliability.
3. **LM Studio** (OpenAI‑compatible API) → structured Markdown meeting minutes. You can use Ollama too.

---

## Requirements

* **Python** 3.10+
* **FFmpeg 8.0+** compiled with the **`whisper` filter** (requires Whisper.cpp). Verify with:

  * Linux/macOS: `ffmpeg -hide_banner -filters | grep whisper`
  * Windows: `ffmpeg -hide_banner -filters | findstr whisper`
* **LM Studio** running in local server/developer mode (OpenAI‑compatible), or Ollama,  usually at `http://localhost:1234`.
* **Whisper.cpp ggml model** file(s), e.g. `ggml-large-v3-turbo.bin`.

Python packages:

```bash
gradio>=4.44.1
uvicorn>=0.30
starlette>=0.37
anyio>=4.4
h11>=0.14
httpx>=0.27
httpcore>=1.0
python-dotenv
```

> **Privacy**: All audio processing happens locally via FFmpeg; the transcript is summarized by your local LM Studio instance.

---

## Installation

1. **Clone** the repo and enter it.
   ```bash
   git clone https://github.com/magicmars35/AutoTranscriptReport
   cd AutoTranscriptReport
   ```

   
3. (Optional but recommended) **Create a venv** and activate it.

   For Windows :
   
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
5. **Install deps**:

   ```bash
   pip install -r requirements.txt
   ```
6. **Place a Whisper.cpp model** under `./models/`, e.g. `./models/ggml-large-v3-turbo.bin`. Models can be found here : https://huggingface.co/ggerganov/whisper.cpp/tree/main
7. **Ensure FFmpeg 8** supports the `whisper` filter (see Requirements above).
8. **Start LM Studio** (or Ollama) in server mode (Developer tab → Start server). Default base URL is `http://localhost:1234` and the API path is usually `/v1/chat/completions`.

---

## Configuration (.env)

Create a `.env` at the project root if you want to override defaults:

```ini
# LM Studio
LMSTUDIO_BASE_URL=http://localhost:1234
LMSTUDIO_API_PATH=/v1/chat/completions
LMSTUDIO_MODEL=Qwen2.5-7B-Instruct
LMSTUDIO_API_KEY=lm-studio

# FFmpeg + Whisper
FFMPEG_BIN=ffmpeg
WHISPER_MODEL_PATH=./models/ggml-large-v3-turbo.bin
WHISPER_LANGUAGE=fr

# Templates storage
TEMPLATES_PATH=prompt_templates.json
```

Defaults are the same as the example above. You can also edit the constants at the top of the Python file.

---

## Run

```bash
python app.py
```

Gradio will print a local URL (e.g. `http://127.0.0.1:7860`).

---

## Using the App

1. **Record or Upload**: Use the single `Audio` widget to record from the browser mic or upload an existing file (`.wav`, `.mp3`, `.m4a`, etc.).
2. **Choose output format**: `srt` (default) is great to visually verify timestamps; `text` for plain text; `json` for programmatic use.
3. **(Optional) Advanced settings**:

   * **Pre‑roll (ms)**: add a short silence at the start so the very first words aren’t cut (try 250–500 ms).
   * **Queue (ms)**: buffering window for VAD; larger values may help stabilize segmentation.
   * **VAD (Silero)**: enable only if you provide a **Silero VAD** ggml model path; otherwise keep it off to preserve natural pauses.
4. Click **Transcribe** → FFmpeg runs the `whisper` filter and writes the transcript file into `./transcripts/`.
5. Inspect the transcript/SRT/JSON shown in the UI.
6. Pick or edit a **Prompt Template** and click **Summarize** → LM Studio returns a structured Markdown **meeting minutes**.
7. (Optional) Use the **Options** tab to switch UI language (English/French) or theme (light/dark). Choices persist to `ui_settings.json`.

---

## Prompt Templates (CRUD)

* Templates are persisted to **`prompt_templates.json`** in the project root.
* Each entry is a mapping of **name ➝ prompt content**.
* The UI provides buttons to **Reload**, **Save**, and **Delete** templates. The selected template’s content is sent as the system prompt.
* You can also hand‑edit `prompt_templates.json` while the app is stopped.

**Example `prompt_templates.json`:**

```json
{
  "default": "You are an assistant specializing in meeting minutes...",
  "brief": "Write a concise summary focusing on decisions and actions."
}
```

> The app trims the template content to a safe length before sending it to LM Studio.

---

## UI Settings

* The **Options** tab lets you choose interface language (English or French) and theme (light or dark).
* Labels for each language are stored in **`strings/<lang>.json`** for easy translation.
* Selections are persisted to **`ui_settings.json`** so your preferences are restored on next launch.
* Delete or edit this file to reset the UI settings.

---

## Model Choices

Place your chosen **Whisper.cpp ggml** model file under `./models` and point `WHISPER_MODEL_PATH` to it. Recommendations:

* **`ggml-large-v3-turbo.bin`** (\~1.5 GB): great quality/speed balance, multilingual.
* `ggml-large-v3.bin` (\~2.9 GB): highest quality, heavier.
* `ggml-medium.bin` (\~1.5 GB): good quality, lighter than large.
* `ggml-small.bin` (\~466 MB): fast and decent for French.

All models are multilingual unless the filename ends with `.en`.
Models can be found here : https://huggingface.co/ggerganov/whisper.cpp/tree/main

---

## Tips & Troubleshooting

* **Check `whisper` filter availability**: if `ffmpeg -filters` does not list `whisper`, your build isn’t compatible. Install/compile FFmpeg 8 with Whisper support.
* **Windows paths**: the FFmpeg filter parser doesn’t like drive letters (`D:`) and backslashes. This app writes transcripts using **relative POSIX‑style paths** (slashes `/`) to avoid that.
* **Microphone quirks / first words missing**: increase **Pre‑roll** (e.g., 300–800 ms). Keep **VAD off** unless you really need it. The app also auto‑converts inputs to **WAV 16 kHz mono** for consistency.
* **Only partial audio transcribed**: try disabling VAD, increase `queue (ms)`, and ensure your input isn’t corrupted.
* **LM Studio errors (404/connection refused)**: make sure LM Studio’s local server is running and that `LMSTUDIO_BASE_URL` and `LMSTUDIO_API_PATH` match your version.
* **Slow on CPU**: prefer `large‑v3‑turbo` or `small`; quantized variants can help.

---

## Roadmap (ideas)

* Export **DOCX/PDF** for the final minutes.
* **Diarization** / speaker labels.
* Multi‑language post‑processing and translation.
* Batch mode and watch folders.

---

## License

MIT
