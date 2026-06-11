"""
led_pwm.py — Python wrapper around the led_pwm.so C shared library.

The real driver logic lives in led_pwm.c (hardware PWM via /dev/mem mmap).
This module auto-compiles the C source on first import if led_pwm.so is
missing or outdated, then loads it with ctypes.

Public API is identical to the previous pure-Python implementation so
server.py requires no changes:

    pwm = get_pwm()
    pwm.set_brightness(75)   # 0–100 %
    pwm.off()
    pwm.available            # bool — False if driver init failed

Falls back gracefully on any failure (permission denied, not Pi hardware,
gcc not found) — the app continues without LED control.
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_C_SRC = _HERE / "led_pwm.c"
_SO    = _HERE / "led_pwm.so"


# ── Compile ───────────────────────────────────────────────────────────────────

def _compile() -> bool:
    """
    Compile led_pwm.c → led_pwm.so if the .so is missing or the .c is newer.
    Returns True on success, False on failure.
    """
    need_build = (
        not _SO.exists()
        or (_C_SRC.exists() and _C_SRC.stat().st_mtime > _SO.stat().st_mtime)
    )
    if not need_build:
        return True

    log.info("LED PWM: compiling led_pwm.c …")
    result = subprocess.run(
        ["gcc", "-O2", "-shared", "-fPIC", "-o", str(_SO), str(_C_SRC)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.warning("LED PWM: gcc failed — %s", result.stderr.strip())
        return False

    log.info("LED PWM: compiled successfully → %s", _SO)
    return True


# ── Load shared library ───────────────────────────────────────────────────────

def _load() -> Optional[ctypes.CDLL]:
    """
    Load led_pwm.so and declare the C function signatures so ctypes can
    marshal arguments and return values correctly.
    Returns the loaded library or None on failure.
    """
    if not _compile():
        return None

    try:
        lib = ctypes.CDLL(str(_SO))
    except OSError as e:
        log.warning("LED PWM: cannot load led_pwm.so — %s", e)
        return None

    # Declare types so ctypes marshals correctly
    lib.pwm_init.restype          = ctypes.c_int
    lib.pwm_init.argtypes         = []

    lib.pwm_set_brightness.restype  = None
    lib.pwm_set_brightness.argtypes = [ctypes.c_float]

    lib.pwm_get_brightness.restype  = ctypes.c_float
    lib.pwm_get_brightness.argtypes = []

    lib.pwm_off.restype    = None
    lib.pwm_off.argtypes   = []

    lib.pwm_is_available.restype  = ctypes.c_int
    lib.pwm_is_available.argtypes = []

    lib.pwm_cleanup.restype  = None
    lib.pwm_cleanup.argtypes = []

    return lib


# ── HardwarePWM class ─────────────────────────────────────────────────────────

class HardwarePWM:
    """
    Thin Python wrapper around the C PWM driver.

    Presents the same interface as the previous pure-Python version so
    nothing else in the codebase needs to change.

    If the C library fails to load or initialise, `available` is False
    and all method calls are no-ops.
    """

    def __init__(self) -> None:
        self.available   = False
        self._lib: Optional[ctypes.CDLL] = None
        self._init()

    def _init(self) -> None:
        """Load the shared library and call pwm_init()."""
        lib = _load()
        if lib is None:
            return

        try:
            ok = lib.pwm_init()
        except Exception as e:
            log.warning("LED PWM: pwm_init() raised %s — LED disabled", e)
            return

        if ok:
            self._lib      = lib
            self.available = True
            log.info("LED PWM: C driver initialised")
        else:
            log.warning("LED PWM: pwm_init() returned 0 — LED disabled")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_brightness(self, percent: float) -> None:
        """Set LED brightness 0–100 %. Clamps silently to range."""
        if self._lib:
            self._lib.pwm_set_brightness(float(percent))

    @property
    def brightness(self) -> float:
        """Return the last brightness value set (0–100)."""
        if self._lib:
            return float(self._lib.pwm_get_brightness())
        return 0.0

    def off(self) -> None:
        """Turn LED off (brightness = 0)."""
        if self._lib:
            self._lib.pwm_off()

    def __del__(self) -> None:
        """Clean up mmap regions on garbage collection."""
        try:
            if self._lib:
                self._lib.pwm_cleanup()
        except Exception:
            pass


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: Optional[HardwarePWM] = None


def get_pwm() -> HardwarePWM:
    """Return the module-level HardwarePWM singleton."""
    global _instance
    if _instance is None:
        _instance = HardwarePWM()
    return _instance
