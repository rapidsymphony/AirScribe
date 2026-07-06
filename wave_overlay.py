"""
AirScribe — Waveform Overlay
============================

A small, frameless, always-on-top window that appears while the push-to-talk
key is held, showing an animated sound-wave whose bars react to the live
microphone level. Pure tkinter (stdlib) — nothing leaves the machine.

Threading model: tkinter must run on the main thread (hard requirement on
macOS), so :meth:`WaveOverlay.run` owns the main thread with ``mainloop()``.
Other threads (the pynput listener) only flip atomic flags via
:meth:`show` / :meth:`hide` / :meth:`stop`; a 30 ms ``after()`` tick on the
Tk thread applies them and redraws the bars.

If tkinter is unavailable (headless server, stripped-down Python), ``run()``
returns ``False`` and the app falls back to running without a visualiser.

Focus safety: the window must NEVER take keyboard focus — stealing focus
would move the text cursor out of the app the user is dictating into and
the paste would go nowhere. So the window is created exactly once as a
non-activating panel (macOS "help" window style) and "shown"/"hidden" by
moving it on/off screen and toggling opacity — never ``deiconify``/``lift``,
both of which activate the window on macOS.
"""

import logging
import math
import time
from typing import Callable, List, Optional

logger = logging.getLogger("wave_overlay")

# Window geometry / look ------------------------------------------------------
WIDTH = 280
HEIGHT = 64
BOTTOM_MARGIN = 90          # px above the bottom edge of the screen
BAR_COUNT = 24
BAR_WIDTH = 6
BG_COLOR = "#101418"
BAR_COLOR = "#4fd1c5"       # teal
BAR_COLOR_HOT = "#f6ad55"   # amber tips when speaking loudly
ALPHA = 0.92
TICK_MS = 30

# Level shaping ---------------------------------------------------------------
LEVEL_GAIN = 9.0            # typical speech RMS is ~0.02–0.15; scale to 0–1
LEVEL_SMOOTHING = 0.65      # 0 = jumpy, 1 = frozen
IDLE_PULSE = 0.12           # minimum bar life so the overlay never looks dead


def bar_heights(level: float, phase: float, n: int = BAR_COUNT,
                max_height: float = HEIGHT - 16) -> List[float]:
    """Pure function mapping a mic level (0..1) to ``n`` animated bar heights.

    A centre-weighted envelope gives the classic "voice wave" silhouette and
    per-bar sine wobble keeps the motion organic. Kept side-effect free so it
    can be unit-tested without a display.
    """
    level = max(0.0, min(1.0, level))
    heights = []
    for i in range(n):
        envelope = math.sin(math.pi * (i + 0.5) / n)          # 0..1, peak mid
        wobble = 0.55 + 0.45 * math.sin(phase + i * 0.9)      # 0.1..1
        energy = IDLE_PULSE + (1.0 - IDLE_PULSE) * level * wobble
        heights.append(max(2.0, max_height * envelope * energy))
    return heights


class WaveOverlay:
    """Thread-safe controller around the tkinter waveform window."""

    def __init__(self, level_provider: Callable[[], float]) -> None:
        """
        Args:
            level_provider: Zero-arg callable returning the current mic RMS
                level (>= 0). Called on the Tk thread every tick.
        """
        self._level_provider = level_provider
        self._want_visible = False   # written by listener thread, read by Tk
        self._want_quit = False
        self._is_visible = False
        self._smoothed = 0.0
        self._root = None
        self._canvas = None
        self._started = time.perf_counter()
        self._x_on = 0
        self._y_on = 0
        self._y_off = 0

    # ------------------------------------------------------------------ #
    # Cross-thread controls (safe to call from any thread)
    # ------------------------------------------------------------------ #
    def show(self) -> None:
        self._want_visible = True

    def hide(self) -> None:
        self._want_visible = False

    def stop(self) -> None:
        self._want_quit = True

    # ------------------------------------------------------------------ #
    # Main-thread loop
    # ------------------------------------------------------------------ #
    def run(self) -> bool:
        """Build the window and block in ``mainloop()``.

        Returns True if the GUI ran (and has now exited), False if tkinter
        is unusable and the caller should fall back to a headless wait.
        """
        try:
            import tkinter  # noqa: F401 — probe availability first
        except Exception:
            logger.warning(
                "tkinter is not available; running without the wave overlay."
            )
            return False

        try:
            self._build()
        except Exception:
            logger.exception(
                "Could not create the overlay window; running without it."
            )
            return False

        logger.info("Wave overlay ready.")
        try:
            self._root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self._destroy()
        return True

    # ------------------------------------------------------------------ #
    # Internals (Tk thread only)
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        import tkinter as tk

        root = tk.Tk()

        # macOS: "help"-class windows are frameless, float above everything
        # and — critically — never activate, so showing the overlay cannot
        # pull the text cursor out of the app being dictated into.
        styled = False
        try:
            root.tk.call(
                "::tk::unsupported::MacWindowStyle",
                "style", root._w, "help", "noActivates",
            )
            styled = True
        except tk.TclError:
            pass  # not macOS (or old Tk) — fall through to overrideredirect
        if not styled:
            root.overrideredirect(True)      # frameless on Windows/Linux

        root.attributes("-topmost", True)

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        self._x_on = (screen_w - WIDTH) // 2
        self._y_on = screen_h - HEIGHT - BOTTOM_MARGIN
        self._y_off = screen_h + 200         # parked below the screen edge

        # Start "hidden": parked off-screen and fully transparent. The
        # window stays mapped forever; visibility is only ever geometry +
        # alpha, which never touches keyboard focus.
        try:
            root.attributes("-alpha", 0.0)
        except Exception:
            pass                             # some window managers lack alpha
        root.geometry(f"{WIDTH}x{HEIGHT}+{self._x_on}+{self._y_off}")

        canvas = tk.Canvas(
            root, width=WIDTH, height=HEIGHT,
            bg=BG_COLOR, highlightthickness=0,
        )
        canvas.pack(fill="both", expand=True)

        self._root = root
        self._canvas = canvas
        root.after(TICK_MS, self._tick)

    def _tick(self) -> None:
        if self._want_quit:
            self._destroy()
            return

        if self._want_visible != self._is_visible:
            self._set_visible(self._want_visible)

        if self._is_visible:
            self._draw()

        self._root.after(TICK_MS, self._tick)

    def _set_visible(self, visible: bool) -> None:
        """Show/hide via geometry + alpha only — never deiconify/lift,
        which would activate the window and steal keyboard focus."""
        root = self._root
        if visible:
            self._smoothed = 0.0
            root.geometry(f"+{self._x_on}+{self._y_on}")
            try:
                root.attributes("-alpha", ALPHA)
            except Exception:
                pass
        else:
            try:
                root.attributes("-alpha", 0.0)
            except Exception:
                pass
            root.geometry(f"+{self._x_on}+{self._y_off}")
        self._is_visible = visible

    def _draw(self) -> None:
        try:
            raw_level = float(self._level_provider() or 0.0)
        except Exception:
            raw_level = 0.0
        target = min(1.0, raw_level * LEVEL_GAIN)
        self._smoothed = (
            LEVEL_SMOOTHING * self._smoothed + (1 - LEVEL_SMOOTHING) * target
        )

        phase = (time.perf_counter() - self._started) * 8.0
        heights = bar_heights(self._smoothed, phase)

        canvas = self._canvas
        canvas.delete("all")
        total_span = len(heights) * BAR_WIDTH * 2 - BAR_WIDTH
        x = (WIDTH - total_span) / 2
        mid = HEIGHT / 2
        for h in heights:
            color = BAR_COLOR_HOT if h > (HEIGHT - 16) * 0.75 else BAR_COLOR
            canvas.create_rectangle(
                x, mid - h / 2, x + BAR_WIDTH, mid + h / 2,
                fill=color, outline="",
            )
            x += BAR_WIDTH * 2

    def _destroy(self) -> None:
        if self._root is None:
            return
        try:
            self._root.destroy()
        except Exception:
            pass
        finally:
            self._root = None
            self._canvas = None
