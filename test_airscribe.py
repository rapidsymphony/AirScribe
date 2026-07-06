"""
AirScribe — Test Suite
======================

Dependency-free verification of the AirScribe codebase. Heavy third-party
libraries (sounddevice, faster-whisper, pynput, pyautogui, pyperclip,
requests) are stubbed in ``sys.modules`` so the suite runs on any stock
Python 3.11+ interpreter — no microphone, model download, or Ollama server
required.

Run with:  python test_airscribe.py
"""

import importlib
import os
import py_compile
import queue
import sys
import threading
import types
import unittest
from unittest import mock

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

REQUIRED_FILES = [
    "config.py",
    "audio_capture.py",
    "window_context.py",
    "ai_pipeline.py",
    "injector.py",
    "wave_overlay.py",
    "main.py",
    "requirements.txt",
    ".gitignore",
    "README.md",
]

PYTHON_FILES = [f for f in REQUIRED_FILES if f.endswith(".py")]


# --------------------------------------------------------------------------- #
# Third-party stubs (installed before AirScribe modules are imported)
# --------------------------------------------------------------------------- #
def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_stubs() -> None:
    # ---- numpy (real one preferred; minimal stub otherwise) ---------------
    try:
        import numpy  # noqa: F401
    except ImportError:
        np = _module("numpy")

        class _FakeArray(list):
            def flatten(self):
                return _FakeArray(self)

            def astype(self, _dtype):
                return _FakeArray(self)

            def copy(self):
                return _FakeArray(self)

        np.float32 = "float32"
        np.ndarray = _FakeArray
        np.concatenate = lambda chunks, axis=0: _FakeArray(
            [x for c in chunks for x in c]
        )

    # ---- pynput ------------------------------------------------------------
    pynput = _module("pynput")
    kb = _module("pynput.keyboard")
    pynput.keyboard = kb

    class _Key:
        alt = "alt"
        alt_r = "alt_r"
        f8 = "f8"

    class _KeyCode:
        @staticmethod
        def from_char(char):
            return f"keycode:{char}"

    class _Listener:
        def __init__(self, on_press=None, on_release=None):
            self.on_press, self.on_release = on_press, on_release

        def start(self):
            pass

        def stop(self):
            pass

        def join(self):
            pass

    kb.Key, kb.KeyCode, kb.Listener = _Key, _KeyCode, _Listener

    # ---- sounddevice ---------------------------------------------------------
    sd = _module("sounddevice")

    class _InputStream:
        instances: list = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.callback = kwargs.get("callback")
            self.started = False
            self.closed = False
            _InputStream.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def close(self):
            self.closed = True

    sd.InputStream = _InputStream

    # ---- requests ------------------------------------------------------------
    req = _module("requests")
    exc = _module("requests.exceptions")

    class _ConnectionError(Exception):
        pass

    class _HTTPError(Exception):
        pass

    exc.ConnectionError = _ConnectionError
    exc.HTTPError = _HTTPError
    exc.Timeout = type("Timeout", (Exception,), {})
    req.exceptions = exc
    req.post = mock.MagicMock(name="requests.post")

    # ---- faster_whisper --------------------------------------------------------
    fw = _module("faster_whisper")

    class _WhisperModel:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs

        def transcribe(self, audio, **kwargs):
            seg = types.SimpleNamespace(text=" stub transcript ")
            info = types.SimpleNamespace(duration=1.0)
            return [seg], info

    fw.WhisperModel = _WhisperModel

    # ---- pyautogui / pyperclip / pygetwindow ------------------------------------
    pag = _module("pyautogui")
    pag.hotkey = mock.MagicMock(name="pyautogui.hotkey")

    pyc = _module("pyperclip")
    pyc.copy = mock.MagicMock(name="pyperclip.copy")
    pyc.paste = mock.MagicMock(name="pyperclip.paste", return_value="")

    pgw = _module("pygetwindow")
    pgw.getActiveWindow = mock.MagicMock(return_value=None)


_install_stubs()

import numpy as np  # noqa: E402  (real or stub, installed above)

import ai_pipeline  # noqa: E402
import audio_capture  # noqa: E402
import config  # noqa: E402
import injector  # noqa: E402
import wave_overlay  # noqa: E402
import window_context  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Repository structure
# --------------------------------------------------------------------------- #
class TestRepositoryStructure(unittest.TestCase):
    def test_all_required_files_exist(self):
        for name in REQUIRED_FILES:
            path = os.path.join(PROJECT_DIR, name)
            self.assertTrue(os.path.isfile(path), f"Missing file: {name}")

    def test_all_python_files_compile(self):
        for name in PYTHON_FILES:
            py_compile.compile(os.path.join(PROJECT_DIR, name), doraise=True)

    def test_gitignore_covers_essentials(self):
        with open(os.path.join(PROJECT_DIR, ".gitignore")) as fh:
            content = fh.read()
        for pattern in ("__pycache__", ".venv", "*.log", "*.wav"):
            self.assertIn(pattern, content)

    def test_readme_has_required_sections(self):
        with open(os.path.join(PROJECT_DIR, "README.md")) as fh:
            content = fh.read().lower()
        for keyword in (
            "features", "prerequisites", "installation",
            "configuration", "architecture", "ollama",
        ):
            self.assertIn(keyword, content)

    def test_requirements_txt_lists_stack(self):
        with open(os.path.join(PROJECT_DIR, "requirements.txt")) as fh:
            content = fh.read()
        for pkg in (
            "faster-whisper", "sounddevice", "numpy",
            "pynput", "pyautogui", "pyperclip", "requests",
        ):
            self.assertIn(pkg, content)


# --------------------------------------------------------------------------- #
# 2. Configuration
# --------------------------------------------------------------------------- #
class TestConfig(unittest.TestCase):
    def test_core_values(self):
        self.assertEqual(config.SAMPLE_RATE, 16000)
        self.assertEqual(config.OLLAMA_URL, "http://localhost:11434/api/generate")
        self.assertEqual(config.WHISPER_COMPUTE_TYPE, "int8")
        self.assertIn(config.WHISPER_MODEL, ("base.en", "small.en"))
        self.assertIsInstance(config.OLLAMA_MODEL, str)
        self.assertTrue(config.OLLAMA_MODEL)
        self.assertLessEqual(config.OLLAMA_TEMPERATURE, 0.3)
        self.assertIsInstance(config.OVERLAY_ENABLED, bool)
        self.assertEqual(config.CHANNELS, 1)
        self.assertEqual(config.MIN_CHARS_FOR_LLM, 15)
        self.assertIsNotNone(config.HOTKEY)


# --------------------------------------------------------------------------- #
# 3. Window context
# --------------------------------------------------------------------------- #
class TestWindowContext(unittest.TestCase):
    def test_macos_title(self):
        fake = mock.MagicMock(returncode=0, stdout="Visual Studio Code\n")
        with mock.patch("window_context.platform.system", return_value="Darwin"), \
             mock.patch("window_context.subprocess.run", return_value=fake):
            self.assertEqual(
                window_context.get_active_window_title(), "Visual Studio Code"
            )

    def test_linux_title(self):
        fake = mock.MagicMock(returncode=0, stdout="Terminal\n")
        with mock.patch("window_context.platform.system", return_value="Linux"), \
             mock.patch("window_context.subprocess.run", return_value=fake):
            self.assertEqual(window_context.get_active_window_title(), "Terminal")

    def test_missing_tool_returns_empty_string(self):
        with mock.patch("window_context.platform.system", return_value="Linux"), \
             mock.patch(
                 "window_context.subprocess.run",
                 side_effect=FileNotFoundError("xdotool not installed"),
             ):
            self.assertEqual(window_context.get_active_window_title(), "")

    def test_unknown_platform_returns_empty_string(self):
        with mock.patch("window_context.platform.system", return_value="Plan9"):
            self.assertEqual(window_context.get_active_window_title(), "")


# --------------------------------------------------------------------------- #
# 4. AI pipeline
# --------------------------------------------------------------------------- #
class TestAIPipeline(unittest.TestCase):
    def setUp(self):
        import requests
        self.requests = requests
        self.requests.post.reset_mock(return_value=True, side_effect=True)
        self.pipeline = ai_pipeline.AIPipeline(queue.Queue(), on_result=lambda t: None)

    def test_system_prompt_is_exact_contract(self):
        for phrase in (
            "specialized dictation post-processor",
            "Remove all filler words",
            "meet at three",
            "Output ONLY the cleaned text",
            "adapt formatting appropriately",
        ):
            self.assertIn(phrase, ai_pipeline.SYSTEM_PROMPT)

    def test_short_transcript_bypasses_llm(self):
        result = self.pipeline.cleanup("hi there")  # < 15 chars
        self.assertEqual(result, "hi there")
        self.requests.post.assert_not_called()

    def test_long_transcript_calls_ollama(self):
        response = mock.MagicMock(
            text='{"response": "This is the cleaned sentence."}'
        )
        self.requests.post.return_value = response

        result = self.pipeline.cleanup(
            "um so this is like a long raw transcript", "VS Code"
        )
        self.assertEqual(result, "This is the cleaned sentence.")

        _, kwargs = self.requests.post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["model"], config.OLLAMA_MODEL)
        self.assertEqual(payload["system"], ai_pipeline.SYSTEM_PROMPT)
        self.assertFalse(payload["stream"])
        self.assertIn("VS Code", payload["prompt"])
        self.assertIn("um so this is like a long raw transcript", payload["prompt"])
        self.assertLessEqual(payload["options"]["temperature"], 0.3)

    def test_ollama_offline_falls_back_to_raw_text(self):
        self.requests.post.side_effect = self.requests.exceptions.ConnectionError()
        raw = "this transcript should survive an outage"
        self.assertEqual(self.pipeline.cleanup(raw, "Mail"), raw)

    def test_malformed_ollama_response_falls_back(self):
        self.requests.post.side_effect = None
        self.requests.post.return_value = mock.MagicMock(text="not json at all")
        raw = "another sufficiently long transcript here"
        self.assertEqual(self.pipeline.cleanup(raw), raw)

    def test_transcribe_joins_segments(self):
        seg1 = types.SimpleNamespace(text=" Hello world. ")
        seg2 = types.SimpleNamespace(text=" Second segment. ")
        info = types.SimpleNamespace(duration=2.0)
        fake_model = mock.MagicMock()
        fake_model.transcribe.return_value = ([seg1, seg2], info)
        self.pipeline._model = fake_model

        self.assertEqual(
            self.pipeline.transcribe(object()), "Hello world. Second segment."
        )

    def test_worker_thread_processes_queue_and_shuts_down(self):
        results = []
        audio_queue = queue.Queue()
        pipeline = ai_pipeline.AIPipeline(audio_queue, on_result=results.append)
        pipeline.start()  # Uses stubbed WhisperModel.

        self.requests.post.reset_mock(return_value=True, side_effect=True)
        self.requests.post.return_value = mock.MagicMock(
            text='{"response": "clean"}'
        )
        audio_queue.put((object(), "Some Window"))
        audio_queue.join()  # Wait for the worker to finish the item.

        pipeline.shutdown()
        self.assertEqual(results, ["clean"])
        self.assertFalse(pipeline._thread.is_alive())


# --------------------------------------------------------------------------- #
# 4b. LLM output sanitizer (the "Here is the processed text:" fix)
# --------------------------------------------------------------------------- #
class TestSanitizer(unittest.TestCase):
    def test_strips_here_is_preamble(self):
        self.assertEqual(
            ai_pipeline.sanitize_llm_output(
                "Here is the processed text: Meet me at three."
            ),
            "Meet me at three.",
        )

    def test_strips_preamble_variants(self):
        cases = {
            "Here's the cleaned text: Hello world.": "Hello world.",
            "Here is your corrected transcript: Hello world.": "Hello world.",
            "Cleaned text: Hello world.": "Hello world.",
            "The formatted output is: Hello world.": "Hello world.",
            "Sure, here is the cleaned-up version: Hello world.": "Hello world.",
        }
        for raw, expected in cases.items():
            self.assertEqual(
                ai_pipeline.sanitize_llm_output(raw), expected, msg=raw
            )

    def test_strips_the_exact_leak_reported_by_the_user(self):
        self.assertEqual(
            ai_pipeline.sanitize_llm_output(
                "Here is the cleaned transcript:\n\n"
                "I want you to go to folders and check every document."
            ),
            "I want you to go to folders and check every document.",
        )

    def test_label_line_heuristic_catches_novel_preambles(self):
        self.assertEqual(
            ai_pipeline.sanitize_llm_output(
                "Below is the cleaned transcript:\nMeet me at three."
            ),
            "Meet me at three.",
        )

    def test_dictated_here_is_list_is_not_eaten(self):
        text = "Here is what we need: milk, eggs, and oat milk."
        self.assertEqual(ai_pipeline.sanitize_llm_output(text), text)

    def test_strips_wrapping_quotes(self):
        self.assertEqual(
            ai_pipeline.sanitize_llm_output('"Meet me at three."'),
            "Meet me at three.",
        )

    def test_strips_code_fence(self):
        self.assertEqual(
            ai_pipeline.sanitize_llm_output("```\nprint('hi')\n```"),
            "print('hi')",
        )

    def test_strips_think_blocks(self):
        self.assertEqual(
            ai_pipeline.sanitize_llm_output(
                "<think>the user wants cleanup</think>Meet me at three."
            ),
            "Meet me at three.",
        )

    def test_strips_trailing_note_paragraph_after_code_fence(self):
        raw = (
            "```python\ndef get_user():\n    return current_user\n```\n\n"
            "Note: I've assumed `current_user` is defined elsewhere."
        )
        self.assertEqual(
            ai_pipeline.sanitize_llm_output(raw),
            "def get_user():\n    return current_user",
        )

    def test_dictated_note_alone_is_not_stripped(self):
        text = "Note: the deadline moved to Friday."
        self.assertEqual(ai_pipeline.sanitize_llm_output(text), text)

    def test_clean_text_passes_through_untouched(self):
        text = "The meeting is at three, not two. Bring the report."
        self.assertEqual(ai_pipeline.sanitize_llm_output(text), text)

    def test_legit_sentence_starting_with_the_text_is_kept(self):
        text = "The text you sent me yesterday was great."
        self.assertEqual(ai_pipeline.sanitize_llm_output(text), text)

    def test_cleanup_applies_sanitizer_to_ollama_response(self):
        import requests
        requests.post.reset_mock(return_value=True, side_effect=True)
        requests.post.return_value = mock.MagicMock(
            text='{"response": "Here is the processed text: All fixed now."}'
        )
        pipeline = ai_pipeline.AIPipeline(queue.Queue(), on_result=lambda t: None)
        result = pipeline.cleanup("um a sufficiently long raw transcript", "Notes")
        self.assertEqual(result, "All fixed now.")


# --------------------------------------------------------------------------- #
# 4c. Wave overlay
# --------------------------------------------------------------------------- #
class TestWaveOverlay(unittest.TestCase):
    def test_bar_heights_shape_and_bounds(self):
        heights = wave_overlay.bar_heights(0.5, phase=1.0)
        self.assertEqual(len(heights), wave_overlay.BAR_COUNT)
        for h in heights:
            self.assertGreaterEqual(h, 2.0)
            self.assertLessEqual(h, wave_overlay.HEIGHT - 16)

    def test_bar_heights_scale_with_level(self):
        quiet = sum(wave_overlay.bar_heights(0.0, phase=0.3))
        loud = sum(wave_overlay.bar_heights(1.0, phase=0.3))
        self.assertGreater(loud, quiet)

    def test_bar_heights_clamps_out_of_range_level(self):
        for level in (-1.0, 5.0):
            for h in wave_overlay.bar_heights(level, phase=0.0):
                self.assertLessEqual(h, wave_overlay.HEIGHT - 16)

    def test_show_hide_stop_flip_flags_thread_safely(self):
        overlay = wave_overlay.WaveOverlay(level_provider=lambda: 0.1)
        overlay.show()
        self.assertTrue(overlay._want_visible)
        overlay.hide()
        self.assertFalse(overlay._want_visible)
        overlay.stop()
        self.assertTrue(overlay._want_quit)

    def test_run_returns_false_without_tkinter(self):
        overlay = wave_overlay.WaveOverlay(level_provider=lambda: 0.0)
        with mock.patch.dict(sys.modules, {"tkinter": None}):
            self.assertFalse(overlay.run())

    def test_stop_before_run_is_safe(self):
        overlay = wave_overlay.WaveOverlay(level_provider=lambda: 0.0)
        overlay.stop()  # must not raise even though no window exists


# --------------------------------------------------------------------------- #
# 5. Injector
# --------------------------------------------------------------------------- #
class TestInjector(unittest.TestCase):
    def _run_inject(self, system: str, text: str, original: str = "old clip"):
        events = []

        def fake_paste():
            events.append(("paste",))
            return original

        def fake_copy(value):
            events.append(("copy", value))

        def fake_hotkey(*keys):
            events.append(("hotkey", keys))

        with mock.patch("platform.system", return_value=system):
            importlib.reload(injector)
        with mock.patch.object(injector.pyperclip, "paste", fake_paste), \
             mock.patch.object(injector.pyperclip, "copy", fake_copy), \
             mock.patch.object(injector.pyautogui, "hotkey", fake_hotkey), \
             mock.patch("injector.time.sleep"):
            injector.inject_text(text)
        return events

    def tearDown(self):
        importlib.reload(injector)  # Restore real platform detection.

    def test_macos_flow_saves_pastes_and_restores(self):
        events = self._run_inject("Darwin", "new text")
        self.assertEqual(
            events,
            [
                ("paste",),                     # save original clipboard
                ("copy", ""),                   # clear
                ("copy", "new text"),           # write cleaned text
                ("hotkey", ("command", "v")),   # macOS paste
                ("copy", "old clip"),           # restore original
            ],
        )

    def test_windows_uses_ctrl_v(self):
        events = self._run_inject("Windows", "hello")
        self.assertIn(("hotkey", ("ctrl", "v")), events)

    def test_empty_text_is_a_noop(self):
        events = self._run_inject("Darwin", "")
        self.assertEqual(events, [])

    def test_clipboard_read_failure_still_pastes(self):
        with mock.patch.object(
            injector.pyperclip, "paste", side_effect=RuntimeError("locked")
        ), mock.patch.object(injector.pyperclip, "copy") as copy_mock, \
             mock.patch.object(injector.pyautogui, "hotkey") as hotkey_mock, \
             mock.patch("injector.time.sleep"):
            injector.inject_text("resilient")
        copy_mock.assert_any_call("resilient")
        hotkey_mock.assert_called_once()

    def test_injection_is_lock_guarded(self):
        self.assertIsInstance(
            injector._INJECT_LOCK, type(threading.Lock())
        )


# --------------------------------------------------------------------------- #
# 6. Audio capture
# --------------------------------------------------------------------------- #
class TestAudioCapture(unittest.TestCase):
    def _make_chunk(self):
        try:
            return np.zeros((160, 1), dtype=np.float32)
        except TypeError:  # minimal stub numpy
            return np.ndarray([0.0] * 160)

    def test_record_cycle_enqueues_audio_and_context(self):
        q: queue.Queue = queue.Queue()
        recorder = audio_capture.AudioCapture(
            q, context_provider=lambda: "Notes App"
        )

        recorder.start()
        self.assertTrue(recorder.is_recording)
        stream = recorder._stream
        self.assertEqual(stream.kwargs["samplerate"], config.SAMPLE_RATE)
        self.assertEqual(stream.kwargs["channels"], 1)
        self.assertTrue(stream.started)

        # Simulate three PortAudio callbacks while the key is held.
        for _ in range(3):
            recorder._callback(self._make_chunk(), 160, None, None)

        recorder.stop()
        self.assertFalse(recorder.is_recording)
        self.assertTrue(stream.closed, "Stream must be closed to avoid leaks")

        audio, title = q.get_nowait()
        self.assertEqual(len(audio), 480)
        self.assertEqual(title, "Notes App")

    def test_stop_without_audio_enqueues_nothing(self):
        q: queue.Queue = queue.Queue()
        recorder = audio_capture.AudioCapture(q)
        recorder.start()
        recorder.stop()
        self.assertTrue(q.empty())

    def test_duplicate_start_from_key_autorepeat_is_ignored(self):
        q: queue.Queue = queue.Queue()
        recorder = audio_capture.AudioCapture(q)
        recorder.start()
        first_stream = recorder._stream
        recorder.start()  # auto-repeat press
        self.assertIs(recorder._stream, first_stream)
        recorder.close()

    def test_callback_ignores_frames_when_not_recording(self):
        q: queue.Queue = queue.Queue()
        recorder = audio_capture.AudioCapture(q)
        recorder._callback(self._make_chunk(), 160, None, None)
        self.assertEqual(recorder._buffer, [])

    def test_level_tracks_recording_state(self):
        q: queue.Queue = queue.Queue()
        recorder = audio_capture.AudioCapture(q)
        self.assertEqual(recorder.level, 0.0)  # idle → 0 for the overlay

        recorder.start()
        recorder._callback(self._make_chunk(), 160, None, None)
        self.assertGreaterEqual(recorder.level, 0.0)

        if hasattr(np, "__version__"):  # real numpy: check actual RMS math
            loud = np.full((160, 1), 0.5, dtype=np.float32)
            recorder._callback(loud, 160, None, None)
            self.assertAlmostEqual(recorder.level, 0.5, places=3)

        recorder.stop()
        self.assertEqual(recorder.level, 0.0)


# --------------------------------------------------------------------------- #
# 7. Orchestrator wiring
# --------------------------------------------------------------------------- #
class TestMainModule(unittest.TestCase):
    def test_main_imports_and_exposes_entrypoint(self):
        import main
        self.assertTrue(callable(main.main))


if __name__ == "__main__":
    print("=" * 70)
    print("AirScribe test suite (third-party dependencies stubbed)")
    print("=" * 70)
    unittest.main(verbosity=2)
