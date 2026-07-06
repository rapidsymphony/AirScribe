"""
AirScribe — Central Configuration
=================================

All user-tunable settings live in this module. Every other module imports
from here so behaviour can be changed in exactly one place.
"""

from pynput import keyboard

# ---------------------------------------------------------------------------
# Hotkey
# ---------------------------------------------------------------------------
# The push-to-talk key. Hold it to record, release it to transcribe & paste.
# Any pynput Key object (or a KeyCode for character keys) is valid, e.g.:
#   keyboard.Key.alt_r        — right Alt / Option
#   keyboard.Key.f8           — F8
#   keyboard.KeyCode.from_char('`')
HOTKEY = keyboard.Key.alt_r

# ---------------------------------------------------------------------------
# Speech-to-Text (faster-whisper)
# ---------------------------------------------------------------------------
# Model size. "base.en" is a good latency/accuracy trade-off on CPU;
# "small.en" is noticeably more accurate at the cost of speed.
WHISPER_MODEL = "base.en"

# int8 quantisation keeps memory low and inference fast on CPU.
WHISPER_COMPUTE_TYPE = "int8"

# "cpu" is the safe cross-platform default. Set to "cuda" for NVIDIA GPUs.
WHISPER_DEVICE = "cpu"

# ---------------------------------------------------------------------------
# LLM Cleanup (Ollama)
# ---------------------------------------------------------------------------
# qwen2.5:3b benchmarked leak-free (no "Here is the cleaned text:" preambles)
# and fastest in testing; llama3.2 is a close second. Anything you've pulled
# with `ollama pull <model>` works.
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Seconds to wait for the Ollama HTTP API before falling back to raw text.
OLLAMA_TIMEOUT = 30

# Near-zero temperature: cleanup is deterministic editing, not creative
# writing. Also curbs chatty preambles like "Here is the processed text:".
OLLAMA_TEMPERATURE = 0.1

# Transcripts shorter than this (in characters) skip the LLM pass entirely —
# a 3-word utterance rarely needs cleanup and the round-trip costs latency.
MIN_CHARS_FOR_LLM = 15

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000      # Hz — whisper models are trained on 16 kHz audio.
CHANNELS = 1             # Mono.
AUDIO_DTYPE = "float32"  # faster-whisper expects float32 PCM in [-1, 1].

# ---------------------------------------------------------------------------
# Wave overlay
# ---------------------------------------------------------------------------
# Show the animated sound-wave window while the hotkey is held. Requires
# tkinter (bundled with most Python installs); degrades gracefully without it.
OVERLAY_ENABLED = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
