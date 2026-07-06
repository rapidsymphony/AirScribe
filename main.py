"""
AirScribe — Orchestrator
========================

Wires the whole application together:

    Main thread     : tkinter wave-overlay loop (tkinter must own the main
                      thread on macOS); falls back to a plain wait when
                      tkinter is unavailable or the overlay is disabled.
    Listener thread : pynput global hotkey events (press → record + show
                      wave, release → enqueue + hide wave).
    Worker thread   : AIPipeline (faster-whisper → Ollama → injector).

Run with:  python main.py
Stop with: Ctrl+C
"""

import logging
import queue
import sys

from pynput import keyboard

import config
from ai_pipeline import AIPipeline
from audio_capture import AudioCapture
from injector import inject_text
from wave_overlay import WaveOverlay
from window_context import get_active_window_title

logger = logging.getLogger("main")


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format=config.LOG_FORMAT,
    )


def main() -> int:
    _setup_logging()
    logger.info("AirScribe starting — 100%% local dictation.")

    audio_queue: queue.Queue = queue.Queue()
    recorder = AudioCapture(audio_queue, context_provider=get_active_window_title)
    pipeline = AIPipeline(audio_queue, on_result=inject_text)
    overlay = WaveOverlay(level_provider=lambda: recorder.level)

    # Load the ASR model and start the background worker before listening,
    # so the first utterance isn't stalled behind model loading.
    try:
        pipeline.start()
    except Exception:
        logger.exception(
            "Failed to initialise the AI pipeline. Is faster-whisper "
            "installed and the model downloadable/cached?"
        )
        return 1

    # ------------------------------------------------------------------ #
    # Hotkey handling (runs on pynput's listener thread; both handlers
    # only flip lightweight recorder state, so no frames are dropped).
    # ------------------------------------------------------------------ #
    def on_press(key) -> None:
        if key == config.HOTKEY and not recorder.is_recording:
            recorder.start()
            if recorder.is_recording:  # only show the wave if the mic opened
                overlay.show()

    def on_release(key) -> None:
        if key == config.HOTKEY:
            overlay.hide()
            recorder.stop()

    logger.info(
        "Ready. Hold %s to dictate; release to transcribe and paste. "
        "Press Ctrl+C to quit.",
        config.HOTKEY,
    )

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        # The overlay's Tk mainloop must own the main thread (macOS rule).
        # If tkinter is missing or the overlay is disabled, block on the
        # listener instead — the app works identically, just without a wave.
        gui_ran = overlay.run() if config.OVERLAY_ENABLED else False
        if not gui_ran:
            listener.join()
    except KeyboardInterrupt:
        logger.info("Shutdown requested (Ctrl+C).")
    finally:
        overlay.stop()
        listener.stop()
        recorder.close()
        pipeline.shutdown()
        logger.info("AirScribe stopped. Goodbye.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
