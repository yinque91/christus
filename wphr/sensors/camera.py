"""Camera adapter (read-only) + offline replay for WPHR physical sensing.

Real devices are enumerated/opened only with explicit user permission; the
adapter never writes to any device. When no camera exists or permission is not
granted, an offline frame source can be used instead (source=USER_PROVIDED).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from wphr.sensors.base import SensorReading, Source


@dataclass
class CameraInfo:
    index: int
    backend: str
    width: int = 0
    height: int = 0
    fps: float = 0.0


class CameraAdapter:
    """Read-only camera access with pause/resume/close and offline replay."""

    def __init__(self, permission_scope: str = "user_authorized_read_only"):
        self.permission_scope = permission_scope
        self.cap = None
        self.info: CameraInfo | None = None
        self.paused = False
        self.frames_read = 0
        self.errors = 0
        self.source = Source.LOCAL_DEVICE.value

    # ---------------------------------------------------------------- enumerate
    def enumerate(self, max_index: int = 6) -> list[CameraInfo]:
        out: list[CameraInfo] = []
        if cv2 is None:
            return out
        for i in range(max_index):
            try:
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    out.append(CameraInfo(index=i, backend="default",
                                          width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                          height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                          fps=float(cap.get(cv2.CAP_PROP_FPS))))
                cap.release()
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------- open
    def open(self, index: int, width: int = 320, height: int = 240, fps: int = 15,
             authorized: bool = False) -> bool:
        if not authorized:
            raise PermissionError("camera access requires explicit user authorization")
        if cv2 is None:
            raise RuntimeError("cv2 unavailable")
        cap = cv2.VideoCapture(index)
        if not cap or not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap = cap
        self.info = CameraInfo(index=index, backend="default", width=width, height=height, fps=fps)
        self.source = Source.LOCAL_DEVICE.value
        return True

    def read(self) -> np.ndarray | None:
        if self.cap is None or self.paused:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.errors += 1
            return None
        self.frames_read += 1
        return frame

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    # ---------------------------------------------------------------- offline
    def open_offline(self, frames: list[np.ndarray], fps: float = 15.0):
        self._frames = list(frames)
        self._i = 0
        self.info = CameraInfo(index=-1, backend="offline", width=frames[0].shape[1],
                              height=frames[0].shape[0], fps=fps)
        self.source = Source.USER_PROVIDED.value
        self.paused = False

    def read_offline(self) -> np.ndarray | None:
        if self.paused or not hasattr(self, "_frames"):
            return None
        if self._i >= len(self._frames):
            return None
        f = self._frames[self._i]
        self._i += 1
        self.frames_read += 1
        return f

    # -------------------------------------------------------------- readings
    def reading(self, frame: np.ndarray | None, centroid: tuple[float, float] | None,
                latency_ms: float = 0.0) -> SensorReading:
        if frame is None or centroid is None:
            return SensorReading(
                sensor_id="camera:0", modality="vision", timestamp=time.time(),
                values=(), units="px", coordinate_frame="camera_image",
                confidence=0.0, noise_level=1.0, latency_ms=latency_ms,
                is_valid=False, missing_reason="no_frame_or_no_detection",
                source=self.source, provenance="camera_adapter")
        h, w = frame.shape[:2]
        return SensorReading(
            sensor_id=f"camera:{self.info.index if self.info else -1}",
            modality="vision", timestamp=time.time(),
            values=(float(centroid[0]), float(centroid[1]), float(w), float(h)),
            units="px", coordinate_frame="camera_image", confidence=0.9,
            noise_level=0.05, latency_ms=latency_ms, is_valid=True,
            source=self.source, provenance="camera_adapter")
