"""
AirScribe — AI Pipeline (Worker Thread)
=======================================

The heart of AirScribe. A daemon thread pulls finished recordings off the
thread-safe queue and runs a two-stage pipeline:

    Stage 1 — Transcription : faster-whisper (local, quantised)
    Stage 2 — Cleanup       : Ollama LLM (local HTTP API)

The Whisper model is loaded exactly once at startup and kept resident so
per-utterance latency is dominated by inference, not model loading. If the
Ollama server is unreachable or errors, the raw transcript is used instead —
degraded output always beats no output.
"""

import json
import logging
import queue
import re
import threading
import time
from typing import Callable, Optional

import requests

import config

logger = logging.getLogger("ai_pipeline")

# Exact system prompt for the dictation post-processor (do not edit lightly —
# the "output only the cleaned text" contract is what keeps pasting safe).
SYSTEM_PROMPT = (
    "You are a specialized dictation post-processor. Your sole function is "
    "to receive raw speech-to-text output and return clean, formatted text "
    "ready to be pasted into an application. Rules: 1. Remove all filler "
    "words (e.g., um, uh, like). 2. Correct grammatical errors and "
    "punctuation. 3. Resolve spoken self-corrections (e.g., if the input is "
    "'meet at two actually three', output 'meet at three'). 4. Output ONLY "
    "the cleaned text. Do not include any conversational preamble, XML tags, "
    "or commentary. If the active window is provided, adapt formatting "
    "appropriately (e.g., code syntax for IDEs, formal paragraphs for email)."
)

# Sentinel pushed onto the queue to ask the worker thread to exit.
SHUTDOWN = object()

# Chatty preambles some models prepend despite the system prompt, e.g.
# "Here is the processed text:" or "Cleaned text:". A colon and a content
# noun (text/output/transcript/…) are both required, so genuine dictation
# like "Here is what we need: milk, eggs" is never eaten.
_PREAMBLE_RE = re.compile(
    r"""^(?:sure[,.!]?\s+)?(?:okay[,.!]?\s+)?
        (?:
            here(?:'s|\s+is)\s+(?:the\s+|your\s+)?(?:[\w'-]+\s+){0,4}?
            (?:text|output|version|transcript|result)\s*:
          | (?:the\s+|your\s+)?
            (?:processed|clean(?:ed)?(?:\s*-?\s*up)?|corrected|formatted|
               final|polished|revised)\s+
            (?:text|output|version|transcript|result)\s*(?:is)?\s*[:\-]
        )\s*""",
    re.IGNORECASE | re.VERBOSE,
)

# Reasoning models (e.g. deepseek-r1) emit <think>…</think> blocks.
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

# Trailing commentary paragraphs like "Note: I've assumed …". Only ever
# dropped when they FOLLOW real content, so dictating a note is unaffected.
_TRAILING_NOTE_RE = re.compile(
    r"^\(?\s*(?:note|n\.b\.|p\.s\.|disclaimer)\b", re.IGNORECASE
)

# Catch-all for label lines the preamble regex misses ("Here is the cleaned
# transcript:", "Processed output:", …): a short first line that ends with a
# colon AND mentions post-processing. Requires content below it, so dictation
# like "Here is what we need:" followed by a list is never touched (no
# processing cue), nor is a lone "Note to self:" line (nothing follows).
_LABEL_LINE_RE = re.compile(
    r"(?:clean|process|correct|format|revis|polish|transcri|dictat)\w*",
    re.IGNORECASE,
)

_CODE_FENCE_RE = re.compile(r"^```[\w+-]*\s*\n?(.*?)\n?```$", re.DOTALL)

_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’"}


def sanitize_llm_output(text: str) -> str:
    """Strip conversational wrappers the LLM adds despite instructions.

    Removes reasoning tags, trailing "Note: …" commentary paragraphs, a
    wrapping code fence, a leading "Here is the …:" style preamble, and one
    pair of wrapping quotes. Returns "" if nothing survives (the caller then
    falls back to the raw transcript).
    """
    cleaned = _THINK_RE.sub("", text).strip()

    paragraphs = re.split(r"\n\s*\n", cleaned)
    while len(paragraphs) > 1 and _TRAILING_NOTE_RE.match(paragraphs[-1].strip()):
        paragraphs.pop()
    cleaned = "\n\n".join(paragraphs).strip()

    fence = _CODE_FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    cleaned = _PREAMBLE_RE.sub("", cleaned, count=1).strip()

    first_line, newline, rest = cleaned.partition("\n")
    if (
        newline
        and rest.strip()
        and len(first_line) <= 80
        and first_line.rstrip().endswith(":")
        and _LABEL_LINE_RE.search(first_line)
    ):
        cleaned = rest.strip()

    if len(cleaned) >= 2 and _QUOTE_PAIRS.get(cleaned[0]) == cleaned[-1]:
        cleaned = cleaned[1:-1].strip()

    return cleaned


class AIPipeline:
    """Queue-driven transcription + cleanup worker."""

    def __init__(
        self,
        audio_queue: "queue.Queue",
        on_result: Callable[[str], None],
    ) -> None:
        """
        Args:
            audio_queue: Queue of ``(np.ndarray, window_title)`` tuples
                produced by :class:`audio_capture.AudioCapture`.
            on_result: Callback invoked with the final cleaned text
                (typically ``injector.inject_text``).
        """
        self._queue = audio_queue
        self._on_result = on_result
        self._model = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Load the ASR model and start the worker daemon thread."""
        self._load_model()
        self._thread = threading.Thread(
            target=self._run, name="ai-pipeline", daemon=True
        )
        self._thread.start()
        logger.info("AI pipeline worker thread started.")

    def shutdown(self, timeout: float = 5.0) -> None:
        """Ask the worker to exit and wait briefly for it."""
        self._queue.put(SHUTDOWN)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _load_model(self) -> None:
        from faster_whisper import WhisperModel  # Heavy import; defer to start.

        t0 = time.perf_counter()
        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s)…",
            config.WHISPER_MODEL,
            config.WHISPER_DEVICE,
            config.WHISPER_COMPUTE_TYPE,
        )
        self._model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        logger.info("Whisper model ready in %.2fs.", time.perf_counter() - t0)

    # ------------------------------------------------------------------ #
    # Worker loop
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is SHUTDOWN:
                    logger.info("AI pipeline shutting down.")
                    return
                audio, window_title = item
                text = self.process(audio, window_title)
                if text:
                    self._on_result(text)
                else:
                    logger.info("Empty transcription; nothing to paste.")
            except Exception:
                # Never let one bad utterance kill the worker thread.
                logger.exception("Unhandled error while processing utterance.")
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ #
    # Two-stage pipeline
    # ------------------------------------------------------------------ #
    def process(self, audio, window_title: str = "") -> str:
        """Run one utterance through transcription + cleanup."""
        raw = self.transcribe(audio)
        if not raw:
            return ""
        return self.cleanup(raw, window_title)

    def transcribe(self, audio) -> str:
        """Stage 1: faster-whisper ASR on a float32 NumPy array."""
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(audio, beam_size=5)
        raw = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed = time.perf_counter() - t0
        logger.info(
            "Transcribed %.2fs of audio in %.2fs: %r",
            getattr(info, "duration", 0.0),
            elapsed,
            raw,
        )
        return raw

    def cleanup(self, raw_text: str, window_title: str = "") -> str:
        """Stage 2: semantic cleanup via the local Ollama API.

        Very short utterances bypass the LLM entirely — the round-trip
        would cost more latency than the cleanup is worth. On any network
        or parsing failure the raw transcript is returned unchanged.
        """
        if len(raw_text.strip()) < config.MIN_CHARS_FOR_LLM:
            logger.info("Transcript under %d chars; skipping LLM cleanup.",
                        config.MIN_CHARS_FOR_LLM)
            return raw_text.strip()

        user_prompt = (
            f"Active Window: {window_title or 'Unknown'}\n"
            f"Raw Transcript: {raw_text}\n\n"
            "Return ONLY the cleaned text — no label, no preamble, "
            "no commentary."
        )
        payload = {
            "model": config.OLLAMA_MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            # Low temperature: cleanup is deterministic editing, not writing.
            "options": {"temperature": config.OLLAMA_TEMPERATURE},
        }

        t0 = time.perf_counter()
        try:
            response = requests.post(
                config.OLLAMA_URL,
                json=payload,
                timeout=config.OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            cleaned = sanitize_llm_output(
                json.loads(response.text).get("response", "")
            )
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Ollama server unreachable at %s — is `ollama serve` running? "
                "Falling back to the raw transcript.",
                config.OLLAMA_URL,
            )
            return raw_text
        except Exception:
            logger.exception(
                "Ollama cleanup failed; falling back to the raw transcript."
            )
            return raw_text

        elapsed = time.perf_counter() - t0
        if not cleaned:
            logger.warning("Ollama returned an empty response; using raw text.")
            return raw_text

        logger.info("LLM cleanup done in %.2fs: %r", elapsed, cleaned)
        return cleaned
