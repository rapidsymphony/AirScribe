"""
AirScribe — Active Window Context
=================================

Cross-platform lookup of the currently focused window title. The title is
passed to the LLM cleanup stage so formatting can adapt to the destination
(e.g. code style inside an IDE, prose inside an email client).

Every path is wrapped defensively: on any failure — missing tooling,
sandboxing, denied accessibility permissions — an empty string is returned
and the pipeline continues without context.
"""

import logging
import platform
import subprocess

logger = logging.getLogger("window_context")

_OSASCRIPT_SNIPPET = (
    'tell application "System Events" to get name of first application '
    "process whose frontmost is true"
)


def _macos_title() -> str:
    result = subprocess.run(
        ["osascript", "-e", _OSASCRIPT_SNIPPET],
        capture_output=True,
        text=True,
        timeout=2,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _windows_title() -> str:
    import pygetwindow  # Imported lazily; only installed/usable on Windows.

    window = pygetwindow.getActiveWindow()
    return window.title if window and window.title else ""


def _linux_title() -> str:
    result = subprocess.run(
        ["xdotool", "getactivewindow", "getwindowname"],
        capture_output=True,
        text=True,
        timeout=2,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def get_active_window_title() -> str:
    """Return the focused window's title, or "" if it cannot be determined."""
    system = platform.system()
    try:
        if system == "Darwin":
            return _macos_title()
        if system == "Windows":
            return _windows_title()
        if system == "Linux":
            return _linux_title()
        logger.debug("Unsupported platform for window context: %s", system)
        return ""
    except Exception as exc:
        # FileNotFoundError (xdotool/osascript missing), TimeoutExpired,
        # permission errors, Wayland restrictions — all non-fatal.
        logger.debug("Active-window lookup failed: %s", exc)
        return ""
