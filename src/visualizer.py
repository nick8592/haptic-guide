# =============================================================================
# HapticGuide — Real-Time Detection Visualizer
# OpenCV-based overlay for debugging + demo (Windows/Linux)
# =============================================================================

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .detector import Detection, DetectionResult
from .feedback_engine import FeedbackMode

if TYPE_CHECKING:
    from .feedback_engine import FeedbackSignal


# ---------------------------------------------------------------------------
# Color Palette (BGR)
# ---------------------------------------------------------------------------

COLORS = {
    "target_box": (0, 255, 0),       # Green
    "other_box": (128, 128, 128),    # Gray
    "text_bg": (0, 0, 0),            # Black
    "text_fg": (255, 255, 255),      # White
    "proximity_far": (0, 0, 255),    # Red
    "proximity_near": (0, 255, 255), # Yellow
    "proximity_locked": (0, 255, 0), # Green
    "mode_scanning": (0, 165, 255),  # Orange
    "mode_tracking": (0, 255, 255),  # Yellow
    "mode_locked": (0, 255, 0),      # Green
    "crosshair": (0, 255, 0),        # Green
}


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class DetectionVisualizer:
    """
    Renders detection results + feedback state onto a frame for display.

    Shows:
    - Bounding boxes with class labels + confidence
    - Target highlight (green) vs other detections (gray)
    - Proximity indicator bar (left side)
    - Mode indicator (top-left)
    - Crosshair at frame center
    - FPS counter
    - "Metal detector" style proximity ring around target
    """

    def __init__(self, target_class: str = "cell phone") -> None:
        self.target_class = target_class

    def render(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        signal: FeedbackSignal,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Render detection overlay on frame. Returns modified frame.

        Args:
            frame: BGR image (H, W, 3)
            result: Detection results
            signal: Current feedback signal
            fps: Current FPS measurement
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        self._draw_crosshair(vis)
        self._draw_detections(vis, result, signal)
        self._draw_proximity_bar(vis, signal)
        self._draw_mode_indicator(vis, signal)
        self._draw_fps(vis, fps)
        self._draw_target_label(vis, signal)

        return vis

    def show(self, frame: np.ndarray, window_name: str = "HapticGuide") -> bool:
        """
        Display frame and check for quit key.

        Returns True if window should stay open, False if user pressed 'q'.
        """
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            return False
        return True

    @staticmethod
    def close() -> None:
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Drawing Helpers
    # ------------------------------------------------------------------

    def _draw_crosshair(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = 15
        color = COLORS["crosshair"]
        thickness = 1

        cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
        cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)

    def _draw_detections(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        signal: FeedbackSignal,
    ) -> None:
        for det in result.detections:
            is_target = det.class_name.lower() == self.target_class.lower()

            if is_target:
                color = self._proximity_color(signal)
                thickness = 3

                # Draw proximity ring around target
                cx, cy = int(det.center[0]), int(det.center[1])
                bw = det.bbox[2] - det.bbox[0]
                bh = det.bbox[3] - det.bbox[1]
                radius = int(max(bw, bh) * 0.6)

                if signal.mode == FeedbackMode.LOCKED:
                    cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 3)
                    cv2.circle(frame, (cx, cy), radius + 8, (0, 255, 0), 1)
                elif signal.mode == FeedbackMode.TRACKING:
                    cv2.circle(frame, (cx, cy), radius, color, 2)
            else:
                color = COLORS["other_box"]
                thickness = 1

            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"{det.class_name} {det.confidence:.0%}"
            self._draw_label(frame, label, x1, y1 - 5, color)

    def _draw_proximity_bar(self, frame: np.ndarray, signal: FeedbackSignal) -> None:
        h, w = frame.shape[:2]
        bar_w = 20
        bar_h = h - 100
        bar_x = 20
        bar_y = 50

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)

        # Fill based on proximity
        if signal.proximity > 0:
            fill_h = int(bar_h * signal.proximity)
            color = self._proximity_color(signal)
            y_top = bar_y + bar_h - fill_h
            cv2.rectangle(frame, (bar_x + 2, y_top), (bar_x + bar_w - 2, bar_y + bar_h), color, -1)

        # Threshold markers
        for threshold, label in [(0.3, "NEAR"), (0.9, "LOCK")]:
            y = bar_y + bar_h - int(bar_h * threshold)
            cv2.line(frame, (bar_x, y), (bar_x + bar_w, y), (200, 200, 200), 1)
            cv2.putText(frame, label, (bar_x + bar_w + 4, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def _draw_mode_indicator(self, frame: np.ndarray, signal: FeedbackSignal) -> None:
        h, w = frame.shape[:2]

        mode_text = signal.mode.value.upper()
        color = {
            FeedbackMode.IDLE: (100, 100, 100),
            FeedbackMode.SCANNING: COLORS["mode_scanning"],
            FeedbackMode.TRACKING: COLORS["mode_tracking"],
            FeedbackMode.LOCKED: COLORS["mode_locked"],
        }.get(signal.mode, (200, 200, 200))

        bg_w = 160
        bg_h = 30
        cv2.rectangle(frame, (w - bg_w - 10, 10), (w - 10, bg_h + 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - bg_w - 10, 10), (w - 10, bg_h + 10), color, 2)

        text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = w - 10 - bg_w // 2 - text_size[0] // 2
        text_y = 10 + bg_h // 2 + text_size[1] // 2
        cv2.putText(frame, mode_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        text = f"{fps:.1f} FPS"
        cv2.putText(frame, text, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _draw_target_label(self, frame: np.ndarray, signal: FeedbackSignal) -> None:
        text = f"Target: {self.target_class}"
        cv2.putText(frame, text, (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text_fg"], 2)
        cv2.putText(frame, text, (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)

        if signal.mode != FeedbackMode.IDLE and signal.proximity > 0:
            dir_text = signal.direction.value.replace("_", " ").upper()
            prox_text = f"  {signal.proximity:.0%} {dir_text}"
            cv2.putText(frame, prox_text, (50, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self._proximity_color(signal), 1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _proximity_color(signal: FeedbackSignal) -> tuple[int, int, int]:
        if signal.mode == FeedbackMode.LOCKED:
            return COLORS["proximity_locked"]
        elif signal.mode == FeedbackMode.TRACKING:
            return COLORS["proximity_near"]
        return COLORS["proximity_far"]

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 2), COLORS["text_bg"], -1)
        cv2.putText(frame, text, (x + 2, y - 2), font, font_scale, COLORS["text_fg"], thickness)
