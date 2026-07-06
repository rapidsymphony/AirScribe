"""
AirScribe — Clipboard & Injection Manager
=========================================

Pastes the final cleaned text into whatever window currently has focus.

Clipboard injection (copy → simulate paste shortcut → restore) is used
instead of keystroke simulation: it is orders of magnitude faster, immune
to keyboard-layout issues, and cannot mangle unicode. The user's original
clipboard is defensively saved and restored around the operation, and a
module-level lock serialises concurrent injections.
"""

import logging
import platform
import threading
import time

import pyautogui
import pyperclip

logger = logging.getLogger("injector")

# Serialise injections — two utterances finishing close together must not
# interleave their clipboard save/restore cycles.
_INJECT_LOCK = threading.Lock()

# Delay (seconds) for the OS to register clipboard writes / paste events.
_SETTLE_DELAY = 0.05

_IS_MACOS = platform.system() == "Darwin"


def inject_text(text: str) -> None:
    """Paste ``text`` into the active window via defensive clipboard swap."""
    if not text:
        return

    with _INJECT_LOCK:
        t0 = time.perf_counter()

        # 1. Save whatever the user currently has on the clipboard.
        original_clipboard = None
        try:
            original_clipboard = pyperclip.paste()
        except Exception:
            # Clipboard locked by another process or holding non-text data.
            logger.warning(
                "Could not read the current clipboard; it will not be restored."
            )

        try:
            # 2. Clear first so a failed copy can't paste stale content.
            pyperclip.copy("")
            # 3. Write the cleaned text.
            pyperclip.copy(text)
            # 4. Give the OS clipboard a moment to register the change.
            time.sleep(_SETTLE_DELAY)

            # 5. Fire the platform-correct paste shortcut.
            if _IS_MACOS:
                pyautogui.hotkey("command", "v")
            else:
                pyautogui.hotkey("ctrl", "v")

            # 6. Let the target application complete the paste event.
            time.sleep(_SETTLE_DELAY)
            logger.info(
                "Injected %d chars in %.2fs.",
                len(text),
                time.perf_counter() - t0,
            )
        except Exception:
            logger.exception("Text injection failed.")
        finally:
            # 7. Restore the user's original clipboard contents.
            if original_clipboard is not None:
                try:
                    pyperclip.copy(original_clipboard)
                except Exception:
                    logger.warning("Could not restore the original clipboard.")
