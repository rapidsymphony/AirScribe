"""
AirScribe — Audio Capture
=========================

Real-time microphone capture driven by the push-to-talk hotkey.

While the hotkey is held, 16 kHz mono float32 frames are appended to an
in-memory buffer from the sounddevice callback thread. On release the
stream is closed, the buffer is flattened into a single NumPy array and
handed to the worker thread via a thread-safe queue.
"""

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

import config

logger = logging.getLogger("audio_capture")


class AudioCapture:
    """Push-to-talk microphone recorder.

    The stream is opened on :meth:`start` and fully closed on :meth:`stop`
    so no OS audio handle is held while idle. All buffer mutation is guarded
    by a lock because the sounddevice callback runs on a PortAudio thread.
    """

    def __init__(
        self,
        audio_queue: "queue.Queue",
        context_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        """
        Args:
            audio_queue: Thread-safe queue the finished recording is pushed to
                as a ``(np.ndarray, window_title)`` tuple.
            context_provider: Optional zero-arg callable returning the active
                window title, captured at the moment recording stops.
        """
        self._queue = audio_queue
        self._context_provider = context_provider
        self._buffer: list = []
        self._stream: Optional[sd.InputStream] = None
        self._recording = False
        self._lock = threading.Lock()
        self._started_at = 0.0
        self._level = 0.0  # RMS of the latest chunk, feeds the wave overlay

    # ------------------------------------------------------------------ #
    # Stream callback (runs on the PortAudio thread)
    # ------------------------------------------------------------------ #
    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning("Audio stream status: %s", status)
        with self._lock:
            if self._recording:
                # Copy — PortAudio reuses the buffer after the callback returns.
                self._buffer.append(indata.copy())
                try:
                    self._level = float(np.sqrt(np.mean(np.square(indata))))
                except Exception:
                    self._level = 0.0  # metering is cosmetic; never fail capture

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def level(self) -> float:
        """Latest microphone RMS level (0.0 while idle)."""
        return self._level if self._recording else 0.0

    def start(self) -> None:
        """Open the input stream and begin buffering audio."""
        if self._recording:
            return  # Key auto-repeat fires press events; ignore them.

        with self._lock:
            self._buffer = []
            self._recording = True

        try:
            self._stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype=config.AUDIO_DTYPE,
                callback=self._callback,
            )
            self._stream.start()
            self._started_at = time.perf_counter()
            logger.info("Recording started.")
        except Exception:
            # Microphone missing, permission denied, device busy, etc.
            logger.exception(
                "Could not open the microphone. Check that a device is "
                "connected and that AirScribe has microphone permission."
            )
            with self._lock:
                self._recording = False
            self._close_stream()

    def stop(self) -> None:
        """Stop recording, flatten the buffer, and enqueue the result."""
        if not self._recording:
            return

        with self._lock:
            self._recording = False
            chunks = self._buffer
            self._buffer = []

        self._close_stream()

        if not chunks:
            logger.warning("Recording stopped but no audio was captured.")
            return

        audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
        duration = len(audio) / config.SAMPLE_RATE
        logger.info(
            "Recording stopped: %.2fs of audio (%d samples).",
            duration,
            len(audio),
        )

        window_title = ""
        if self._context_provider is not None:
            try:
                window_title = self._context_provider() or ""
            except Exception:
                logger.exception("Window-context lookup failed; continuing.")

        self._queue.put((audio, window_title))

    def close(self) -> None:
        """Release all audio resources (used during shutdown)."""
        with self._lock:
            self._recording = False
            self._buffer = []
        self._close_stream()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _close_stream(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            logger.exception("Error while closing the audio stream.")
        finally:
            self._stream = None
