# =============================================================================
# HapticGuide — Low-Latency Camera Capture (V4L2 + OpenCV)
# =============================================================================

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class CameraConfig(BaseModel):
    device_index: int = 0
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    fourcc: str = "MJPG"
    buffer_size: int = 1  # 1 = minimal latency


# ---------------------------------------------------------------------------
# Camera Capture
# ---------------------------------------------------------------------------

class CameraCapture:
    """
    Low-latency camera capture using OpenCV V4L2 backend.

    Features:
    - Minimal buffer (CAP_PROP_BUFFERSIZE=1) for lowest latency
    - MJPG format for USB cameras (faster than YUYV decompression)
    - Async capture thread to decouple capture from inference
    - Frame skipping: always provides the LATEST frame, never stale
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._capture_thread: threading.Thread | None = None
        self._frame_count = 0
        self._fps_actual = 0.0
        self._fps_timer = time.perf_counter()
        self._fps_frame_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open camera with V4L2 backend."""
        logger.info(
            f"Opening camera: device={self.config.device_index}, "
            f"resolution={self.config.resolution}, fps={self.config.fps}"
        )

        self._cap = cv2.VideoCapture(self.config.device_index, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {self.config.device_index}. "
                f"Check: ls /dev/video*"
            )

        # Configure camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        # Set MJPG format for USB cameras
        fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc)
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Verify settings
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Camera opened: {actual_w}x{actual_h} @ {actual_fps} FPS "
            f"(requested: {self.config.resolution} @ {self.config.fps})"
        )

    def start_capture(self) -> None:
        """Start async capture thread."""
        if self._running:
            logger.warning("Capture already running")
            return

        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True,
        )
        self._capture_thread.start()
        logger.info("Camera capture thread started")

    def stop_capture(self) -> None:
        """Stop async capture thread."""
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        logger.info("Camera capture stopped")

    def close(self) -> None:
        """Release camera resources."""
        self.stop_capture()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    # ------------------------------------------------------------------
    # Frame Access
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> np.ndarray | None:
        """
        Get the most recent frame (thread-safe).

        Returns None if no frame is available yet.
        Always returns the LATEST frame — never stale.
        """
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def fps_actual(self) -> float:
        """Actual capture FPS (measured)."""
        return self._fps_actual

    # ------------------------------------------------------------------
    # Capture Loop (runs in background thread)
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background capture loop — always keeps latest frame."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                logger.error("Camera disconnected")
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame grab failed — retrying")
                time.sleep(0.001)
                continue

            with self._frame_lock:
                self._latest_frame = frame

            self._frame_count += 1
            self._fps_frame_count += 1

            # Update FPS measurement every second
            now = time.perf_counter()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._fps_actual = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._fps_timer = now

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    @staticmethod
    def list_devices() -> list[dict]:
        """List available V4L2 video devices."""
        devices = []
        for index in range(10):
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                devices.append({
                    "index": index,
                    "resolution": f"{w}x{h}",
                    "fps": fps,
                })
                cap.release()
        return devices
