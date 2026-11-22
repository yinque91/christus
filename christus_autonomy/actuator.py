"""Windows SendInput actuator with permission checks and audit logging.

This module provides real mouse/keyboard control, but every action passes through
SafetyBroker first. It is intentionally Windows-only.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import time
from ctypes import wintypes
from pathlib import Path

from .perception import foreground, list_windows, virtual_screen
from .safety import SafetyBroker, SafetyError, WindowInfo

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

ULONG_PTR = ctypes.c_size_t
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)


def normalize_point(x: int, y: int) -> tuple[int, int]:
    """Map virtual-screen coordinates to SendInput's 0..65535 absolute space."""
    vx, vy, vw, vh = virtual_screen()
    vw = max(1, vw - 1)
    vh = max(1, vh - 1)
    return (int(round((x - vx) * 65535 / vw)), int(round((y - vy) * 65535 / vh)))


VK = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "return": 0x0D,
    "shift": 0x10, "ctrl": 0x11, "control": 0x11, "alt": 0x12,
    "pause": 0x13, "capslock": 0x14, "esc": 0x1B, "escape": 0x1B,
    "space": 0x20, "pageup": 0x21, "pagedown": 0x22, "end": 0x23,
    "home": 0x24, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "insert": 0x2D, "delete": 0x2E, "win": 0x5B, "apps": 0x5D,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74,
    "f6": 0x75, "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79,
    "f11": 0x7A, "f12": 0x7B,
}
for _c in "abcdefghijklmnopqrstuvwxyz":
    VK[_c] = ord(_c.upper())
for _c in "0123456789":
    VK[_c] = ord(_c)


def _input_mouse(dx: int, dy: int, data: int, flags: int) -> INPUT:
    return INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dx, dy, data, flags, 0, 0))


def _input_key(vk: int, scan: int, flags: int) -> INPUT:
    return INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(vk, scan, flags, 0, 0))


def _send(inp: INPUT):
    if user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
        raise ctypes.WinError(ctypes.get_last_error())


class DesktopActuator:
    def __init__(self, broker: SafetyBroker | None = None, *, dry_run: bool = False):
        self.broker = broker or SafetyBroker()
        self.dry_run = dry_run

    # --------------------------------------------------------------- helpers
    def _check(self, action: str, **extra):
        self.broker.check(action, foreground=foreground(), **extra)

    def _move_raw(self, x: int, y: int):
        if self.dry_run:
            return
        nx, ny = normalize_point(int(x), int(y))
        _send(_input_mouse(nx, ny, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK))

    # --------------------------------------------------------------- actions
    def move(self, x: int, y: int, duration: float = 0.0):
        self._check("move", x=x, y=y)
        if duration <= 0:
            self._move_raw(x, y)
            return
        # Use current cursor as the interpolation origin.
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        sx, sy = int(pt.x), int(pt.y)
        steps = max(1, int(duration * 60))
        for i in range(1, steps + 1):
            t = i / steps
            self._move_raw(int(sx + (x - sx) * t), int(sy + (y - sy) * t))
            time.sleep(duration / steps)

    def click(self, x: int | None = None, y: int | None = None, *,
              button: str = "left", double: bool = False):
        self._check("doubleclick" if double else "click", x=x, y=y)
        if x is not None and y is not None:
            self._move_raw(x, y)
            time.sleep(0.03)
        down, up = ((MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP) if button == "left"
                    else (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP))
        count = 2 if double else 1
        for _ in range(count):
            if not self.dry_run:
                _send(_input_mouse(0, 0, 0, down))
                _send(_input_mouse(0, 0, 0, up))
            time.sleep(0.05)

    def scroll(self, amount: int):
        self._check("scroll", amount=amount)
        if not self.dry_run:
            _send(_input_mouse(0, 0, int(amount), MOUSEEVENTF_WHEEL))

    def type_text(self, text: str):
        self._check("type", chars=len(text))
        if self.dry_run:
            return
        for unit in text.encode("utf-16-le"):
            pass
        units = [int.from_bytes(text.encode("utf-16-le")[i:i + 2], "little")
                 for i in range(0, len(text.encode("utf-16-le")), 2)]
        for unit in units:
            _send(_input_key(0, unit, KEYEVENTF_UNICODE))
            _send(_input_key(0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
            time.sleep(0.002)

    def key(self, name: str, presses: int = 1):
        self._check("key", key=name)
        vk = VK.get(name.casefold())
        if vk is None:
            raise ValueError(f"unknown key: {name}")
        if self.dry_run:
            return
        for _ in range(max(1, presses)):
            _send(_input_key(vk, 0, 0))
            _send(_input_key(vk, 0, KEYEVENTF_KEYUP))

    def hotkey(self, combo: str):
        self._check("hotkey", shortcut=combo)
        parts = [p.strip().casefold() for p in combo.replace("-", "+").split("+") if p.strip()]
        if not parts:
            raise ValueError("empty hotkey")
        codes = []
        for p in parts:
            if p not in VK:
                raise ValueError(f"unknown key in hotkey: {p}")
            codes.append(VK[p])
        if self.dry_run:
            return
        for vk in codes[:-1]:
            _send(_input_key(vk, 0, 0))
        _send(_input_key(codes[-1], 0, 0))
        _send(_input_key(codes[-1], 0, KEYEVENTF_KEYUP))
        for vk in reversed(codes[:-1]):
            _send(_input_key(vk, 0, KEYEVENTF_KEYUP))

    def activate(self, title: str) -> bool:
        needle = title.casefold()
        for win in list_windows():
            if needle in (win.title or "").casefold():
                user32.ShowWindow(win.hwnd, 9)
                user32.SetForegroundWindow(win.hwnd)
                time.sleep(0.15)
                return True
        return False

    def launch(self, executable: str):
        path = Path(executable)
        name = path.name
        self.broker.check("launch", executable=name)
        if not path.is_absolute():
            found = shutil.which(name)
            if found:
                path = Path(found)
        if not path.is_file():
            raise FileNotFoundError(executable)
        if self.dry_run:
            return None
        return subprocess.Popen([str(path)])
