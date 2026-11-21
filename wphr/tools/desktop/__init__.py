"""Desktop GUI automation via Win32 (ctypes): windows, screenshot, click, keys."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import time
from pathlib import Path

from wphr.tools.policy import Audit, Policy

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
KEYEVENTF_KEYUP = 0x0002


class Desktop:
    def __init__(self, policy: Policy | None = None, audit: Audit | None = None):
        self.policy = policy or Policy()
        self.audit = audit or Audit()

    def windows(self, limit: int = 40) -> list[dict]:
        out: list[dict] = []

        @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        def cb(hwnd, lparam):
            if user32.IsWindowVisible(hwnd):
                n = user32.GetWindowTextLengthW(hwnd)
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if buf.value:
                    out.append({"hwnd": int(hwnd), "title": buf.value})
            return True

        user32.EnumWindows(cb, 0)
        self.audit.log("desktop.windows", "EnumWindows", f"{len(out)} visible", ok=True)
        return out[:limit]

    def active_window(self) -> dict:
        hwnd = user32.GetForegroundWindow()
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        return {"hwnd": int(hwnd), "title": buf.value}

    def screenshot(self, out: str, hwnd: int | None = None) -> dict:
        from PIL import ImageGrab
        p = Path(out).resolve() if Path(out).is_absolute() else (self.policy.workspace / out).resolve()
        if not self.policy.inside(p):
            raise PermissionError("screenshot must stay inside workspace")
        p.parent.mkdir(parents=True, exist_ok=True)
        if hwnd:
            rect = wt.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            box = (rect.left, rect.top, rect.right, rect.bottom)
        else:
            box = None
        img = ImageGrab.grab(bbox=box)
        img.save(p)
        self.audit.log("desktop.screenshot", str(hwnd or "screen"), str(p), ok=True,
                       artifacts=[str(p)])
        return {"ok": True, "path": str(p), "size": img.size}

    def click(self, x: int, y: int) -> dict:
        self.policy.check("proc.python")
        user32.SetCursorPos(int(x), int(y))
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        self.audit.log("desktop.click", f"{x},{y}", "clicked", ok=True)
        return {"ok": True, "x": x, "y": y}

    def type_text(self, text: str, interval: float = 0.01) -> dict:
        self.policy.check("proc.python")
        for ch in text:
            vk = user32.VkKeyScanW(ord(ch))
            code = vk & 0xFF
            user32.keybd_event(code, 0, 0, 0)
            user32.keybd_event(code, 0, KEYEVENTF_KEYUP, 0)
            time.sleep(interval)
        self.audit.log("desktop.type", f"{len(text)} chars", "typed", ok=True)
        return {"ok": True, "chars": len(text)}
