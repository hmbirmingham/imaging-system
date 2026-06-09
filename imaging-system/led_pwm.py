"""
led_pwm.py — Hardware PWM for LED backlight on Raspberry Pi 5.

Directly accesses the RP1 PWM peripheral via mmap on /dev/mem.
Falls back gracefully on any failure — the app continues without LED control.

Hardware target: Pi 5, GPIO 12, PWM0_CHAN0, RP1 peripheral.
RP1 PWM base physical address: 0x1f00098000
RP1 GPIO base physical address: 0x1f000d0000
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
from typing import Optional

log = logging.getLogger(__name__)

# ── RP1 physical addresses (Pi 5) ─────────────────────────────────────────────
_RP1_PWM0_PHYS  = 0x1f00098000   # PWM0 peripheral
_RP1_GPIO_PHYS  = 0x1f000d0000   # GPIO peripheral

# PWM register offsets (matches BCM2835 layout, compatible with RP1)
_PWM_CTL  = 0x00   # Control
_PWM_STA  = 0x04   # Status
_PWM_RNG1 = 0x10   # Range  (period), channel 1
_PWM_DAT1 = 0x14   # Data   (duty),   channel 1

# GPIO_CTRL offset within per-pin block (8 bytes per GPIO in RP1)
_GPIO_CTRL = 0x04

# PWM clock on RP1 is fixed at 25 MHz.
# Range = 25_000_000 / target_freq_hz. 25 kHz gives smooth LED dimming.
_PWM_RANGE = 1000          # 25 MHz / 25 kHz
_PWM_GPIO  = 12            # GPIO pin for PWM0_CHAN0
_PWM_FUNCSEL = 4           # ALT function value for PWM on GPIO 12 in RP1

# CTL register bits
_CTL_PWEN1 = 1 << 0        # Enable channel 1
_CTL_MSEN1 = 1 << 7        # M/S mode (true PWM duty cycle, not dithered)


class HardwarePWM:
    """
    Direct mmap PWM driver for Pi 5 RP1 peripheral.

    Usage
    -----
        pwm = HardwarePWM()
        pwm.set_brightness(75)   # 75 %
        pwm.off()

    If hardware init fails, `available` is False and all calls are no-ops.
    """

    def __init__(self, gpio: int = _PWM_GPIO) -> None:
        self.available  = False
        self._gpio      = gpio
        self._brightness = 0.0
        self._pwm_map: Optional[mmap.mmap] = None
        self._gpio_map: Optional[mmap.mmap] = None
        self._init()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init(self) -> None:
        try:
            fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        except PermissionError:
            log.warning("LED PWM: /dev/mem access denied (need root or video group) — LED disabled")
            return
        except FileNotFoundError:
            log.warning("LED PWM: /dev/mem not found — not running on Pi hardware")
            return
        except OSError as e:
            log.warning("LED PWM: cannot open /dev/mem: %s — LED disabled", e)
            return

        try:
            page = mmap.PAGESIZE

            # Map PWM registers
            pwm_aligned = _RP1_PWM0_PHYS & ~(page - 1)
            self._pwm_map = mmap.mmap(
                fd, page,
                mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                offset=pwm_aligned,
            )

            # Map GPIO registers
            gpio_aligned = _RP1_GPIO_PHYS & ~(page - 1)
            self._gpio_map = mmap.mmap(
                fd, page,
                mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                offset=gpio_aligned,
            )

            os.close(fd)

            self._set_gpio_function()
            self._write_pwm(_PWM_RNG1, _PWM_RANGE)
            self._write_pwm(_PWM_DAT1, 0)
            self._write_pwm(_PWM_CTL, _CTL_MSEN1 | _CTL_PWEN1)

            self.available = True
            log.info("LED PWM: RP1 hardware PWM initialised on GPIO %d", self._gpio)

        except Exception as e:
            log.warning("LED PWM: init error (%s) — LED disabled", e)
            self._cleanup_maps()

    def _set_gpio_function(self) -> None:
        """Set GPIO_CTRL FUNCSEL to route PWM to the pin."""
        offset = self._gpio * 8 + _GPIO_CTRL     # 8 bytes per pin in RP1
        self._gpio_map.seek(offset)
        val = struct.unpack("<I", self._gpio_map.read(4))[0]
        val = (val & ~0x1F) | (_PWM_FUNCSEL & 0x1F)
        self._gpio_map.seek(offset)
        self._gpio_map.write(struct.pack("<I", val))

    # ── Register helpers ──────────────────────────────────────────────────────

    def _write_pwm(self, reg: int, val: int) -> None:
        self._pwm_map.seek(reg)
        self._pwm_map.write(struct.pack("<I", val & 0xFFFF_FFFF))

    # ── Public API ────────────────────────────────────────────────────────────

    def set_brightness(self, percent: float) -> None:
        """Set brightness 0–100. Clamps silently to range."""
        percent = max(0.0, min(100.0, float(percent)))
        self._brightness = percent
        if not self.available:
            return
        duty = int(_PWM_RANGE * percent / 100.0)
        self._write_pwm(_PWM_DAT1, duty)

    @property
    def brightness(self) -> float:
        return self._brightness

    def off(self) -> None:
        self.set_brightness(0)

    def _cleanup_maps(self) -> None:
        for m in (self._pwm_map, self._gpio_map):
            if m:
                try:
                    m.close()
                except Exception:
                    pass

    def __del__(self) -> None:
        try:
            self.off()
        except Exception:
            pass
        self._cleanup_maps()


# ── Module-level singleton ─────────────────────────────────────────────────────
_instance: Optional[HardwarePWM] = None


def get_pwm() -> HardwarePWM:
    global _instance
    if _instance is None:
        _instance = HardwarePWM()
    return _instance
