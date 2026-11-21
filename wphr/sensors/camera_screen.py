"""Capture a real camera frame via the Windows Camera app window (user-visible preview).

DirectShow/OpenCV cannot bind the device in this environment, but the Camera app
shows a live feed. We screenshot that window; the result is a real camera image
(screen-derived), explicitly tagged provenance=SCREEN_OF_CAMERA_APP.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import time
from pathlib import Path

from PIL import ImageGrab

TITLE = "\u76f8\u673a"  # 相机


def find_camera_window():
    u = ctypes.windll.user32
    hits = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(h, l):
        n = u.GetWindowTextLengthW(h)
        b = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(h, b, n + 1)
        if TITLE in b.value:
            hits.append(int(h))
        return True

    u.EnumWindows(cb, 0)
    return hits[0] if hits else 0


def capture(out: str, crop: float = 0.12) -> dict:
    u = ctypes.windll.user32
    hwnd = find_camera_window()
    if not hwnd:
        return {"ok": False, "reason": "camera_app_window_not_found"}
    u.SetForegroundWindow(hwnd)
    time.sleep(1.0)
    r = wt.RECT()
    u.GetWindowRect(hwnd, ctypes.byref(r))
    img = ImageGrab.grab(bbox=(r.left, r.top, r.right, r.bottom))
    w, h = img.size
    box = (int(w * crop), int(h * crop), int(w * (1 - crop)), int(h * (1 - crop)))
    preview = img.crop(box)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    preview.save(p)
    meta = {"ok": True, "path": str(p), "size": preview.size,
            "provenance": "SCREEN_OF_CAMERA_APP", "window": hwnd,
            "ts": time.time()}
    p.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "camera_preview.png"
    print(json.dumps(capture(out), ensure_ascii=False))
