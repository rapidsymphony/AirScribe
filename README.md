# 🎙️ AirScribe

**The free, hackable, dev-first alternative to Wispr Flow.**

Hold a hotkey, speak, release — your words are transcribed by a local
Whisper model, polished by a local LLM, and pasted straight into whatever
app you're using. No cloud. No API keys. No subscription. No audio ever
leaves your machine.

Tools like Wispr Flow and Superwhisper are great, but they're closed-source,
subscription-gated, and give you zero say in which models actually power
your dictation. AirScribe is a couple hundred lines of readable Python that
does the same job — and because you own the whole pipeline, you get to
open [`config.py`](config.py) and swap the Whisper size, the Ollama model,
the temperature, even the hotkey, and see the effect on your very next
sentence. It's built to be used out of the box in under five minutes, and
to be torn apart and rebuilt just as easily.

| | AirScribe | Wispr Flow / Superwhisper |
|---|---|---|
| **Cost** | $0, forever | Monthly subscription |
| **Where audio goes** | Never leaves your machine | Cloud APIs (usually) |
| **Model choice** | Any Whisper size, any Ollama model — one line each | Fixed, vendor-chosen |
| **Source** | Fully open, ~7 readable modules | Closed |
| **Offline use** | 100% after setup | Requires network |
| **Hackability** | Edit the prompt, the pipeline, the UI — it's your code | None |

---

## ✨ Features

- **🎛️ Bring your own models** — this is the feature that makes AirScribe
  yours. Not happy with transcription accuracy? Bump `WHISPER_MODEL` from
  `base.en` to `small.en`. Want snappier cleanup or a specific model you
  already have pulled? Point `OLLAMA_MODEL` at `llama3.2`, `phi3`,
  `mistral` — anything `ollama list` shows you. Two lines in
  [`config.py`](config.py), no code surgery required.
- **💸 Free, forever** — no subscription, no usage tier, no "pro" paywall.
  You already own the hardware; this just uses it.
- **🔒 100% local & private** — audio and text never touch a network beyond
  `localhost`. Works completely offline once models are downloaded.
- **⚡ Push-to-talk workflow** — hold the hotkey to record, release to paste.
  No windows to click, no apps to switch to.
- **🧠 Two-stage AI pipeline** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (quantised, low-latency) for transcription, then a local
  [Ollama](https://ollama.com) LLM removes filler words, fixes grammar, and
  resolves spoken self-corrections (*"meet at two, actually three"* → *"meet at three"*).
- **🌊 Live sound-wave overlay** — a sleek, frameless waveform appears at the
  bottom of your screen while you hold the hotkey, with bars that dance to
  your actual voice level. Non-activating: it never steals keyboard focus
  from the app you're dictating into. Pure tkinter, zero extra dependencies.
- **🪟 Cross-platform window awareness** — detects your active window
  (macOS / Windows / Linux) so the LLM can format for the destination:
  code-friendly output in IDEs, formal prose in email clients.
- **🧵 Multi-threaded & responsive** — hotkey listening stays on the main
  thread; ASR and LLM inference run on a worker thread behind a thread-safe
  queue. The app never freezes while models are thinking.
- **🛡️ Clipboard defense** — your existing clipboard contents are saved,
  the cleaned text is pasted via the native shortcut, and your original
  clipboard is restored — all under a thread lock.
- **📉 Graceful degradation** — Ollama offline? You still get the raw
  Whisper transcript. Short utterance? The LLM pass is skipped for speed.

---

## 📋 Prerequisites

1. **Python 3.11+**
2. **[Ollama](https://ollama.com/download)** installed and running:
   ```bash
   ollama pull qwen2.5:3b  # fast, leak-free default; llama3.2 also works well
   ollama serve            # usually started automatically
   ```
3. **System dependencies**
   | OS | Requirement |
   |---|---|
   | macOS | Grant **Microphone** + **Accessibility** permissions to your terminal (System Settings → Privacy & Security). PortAudio ships with the `sounddevice` wheel. |
   | Linux | `sudo apt install libportaudio2 xdotool python3-tk` (X11; on Wayland window titles may be unavailable — AirScribe degrades gracefully). |
   | Windows | No extra packages — `pygetwindow` is installed automatically. |

> The first run downloads the Whisper model (~75 MB for `base.en`) and then
> works fully offline.

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/rapidsymphony/AirScribe.git
cd airscribe

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

Hold the hotkey (default: **Right Alt / Right Option**), speak, release —
your polished words appear in the focused app.

---

## ⚙️ Configuration — make it yours

This is the whole point of running your own dictation tool: every knob
below is a plain variable in [`config.py`](config.py). No settings UI to
dig through, no account to log into — change a line, save, restart
`python main.py`, and your next utterance uses it.

| Setting | Default | Description |
|---|---|---|
| `HOTKEY` | `Key.alt_r` | Push-to-talk key (any `pynput` key object). |
| `WHISPER_MODEL` | `base.en` | Whisper size — try `small.en` or `medium.en` for more accuracy at the cost of latency. |
| `WHISPER_COMPUTE_TYPE` | `int8` | Quantisation; keeps CPU inference fast. |
| `WHISPER_DEVICE` | `cpu` | Set to `cuda` for NVIDIA GPUs. |
| `OLLAMA_MODEL` | `qwen2.5:3b` | **Any** local Ollama model — `ollama pull llama3.2`, `phi3`, `mistral`, your own fine-tune, and point this at it. |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint. |
| `OLLAMA_TEMPERATURE` | `0.1` | Near-deterministic cleanup; curbs chatty output. |
| `MIN_CHARS_FOR_LLM` | `15` | Transcripts shorter than this skip the LLM pass. |
| `OVERLAY_ENABLED` | `True` | Show the animated sound-wave while recording. |
| `SAMPLE_RATE` | `16000` | Whisper's native sample rate — leave as-is. |

Want to go further? The system prompt that drives cleanup lives right at
the top of [`ai_pipeline.py`](ai_pipeline.py) as a plain string — edit it to
change tone, enforce a house style, or teach it your team's shorthand.

---

## 🏗️ Architecture

```
 MAIN THREAD                                WORKER THREAD (daemon)
┌───────────────────────────┐             ┌────────────────────────────────┐
│  pynput Global Listener   │             │         AI PIPELINE            │
│                           │             │                                │
│  hotkey press             │             │  ┌──────────────────────────┐  │
│    └─► AudioCapture.start │             │  │ Stage 1: faster-whisper  │  │
│        (sounddevice,      │             │  │  audio ─► raw transcript │  │
│         16 kHz mono f32)  │             │  └────────────┬─────────────┘  │
│                           │             │               ▼                │
│  hotkey release           │  queue.Queue│  ┌──────────────────────────┐  │
│    └─► AudioCapture.stop  │ ═══════════►│  │ Stage 2: Ollama cleanup  │  │
│        + active window    │ (audio,     │  │  fillers, grammar,       │  │
│          title captured   │  title)     │  │  self-corrections        │  │
│                           │             │  └────────────┬─────────────┘  │
│  Ctrl+C ─► graceful       │             │               ▼                │
│            shutdown       │             │  ┌──────────────────────────┐  │
│                           │             │  │ Injector (thread-locked) │  │
│                           │             │  │  save clip ► copy ►      │  │
│                           │             │  │  ⌘V / Ctrl+V ► restore   │  │
│                           │             │  └──────────────────────────┘  │
└───────────────────────────┘             └────────────────────────────────┘
```

**Module map**

| File | Responsibility |
|---|---|
| [`main.py`](main.py) | Orchestration, hotkey listener, graceful shutdown. |
| [`config.py`](config.py) | All user-tunable settings. |
| [`audio_capture.py`](audio_capture.py) | Push-to-talk microphone recording. |
| [`window_context.py`](window_context.py) | Cross-platform active-window title. |
| [`ai_pipeline.py`](ai_pipeline.py) | Worker thread: Whisper ASR + Ollama cleanup + output sanitizer. |
| [`injector.py`](injector.py) | Defensive clipboard paste with restore. |
| [`wave_overlay.py`](wave_overlay.py) | Animated sound-wave window shown while recording. |

> The Tk overlay owns the main thread (a macOS requirement); hotkey events
> arrive on pynput's listener thread and only flip lightweight flags. The
> LLM's reply is passed through a sanitizer that strips chatty wrappers
> ("Here is the processed text:", trailing "Note: …" paragraphs, code fences,
> quotes) so only your words are ever pasted.

---

## 🧪 Testing

A dependency-free test suite (heavy libraries are stubbed) verifies the
pipeline logic, clipboard defense, and cross-platform fallbacks:

```bash
python test_airscribe.py
```

---

## 📄 License

MIT — do whatever you like, just keep the notice.
