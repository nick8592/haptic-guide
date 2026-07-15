# =============================================================================
# HapticGuide — Spatial Feedback Engine
# "Metal Detector" Metaphor: proximity + direction → haptic + audio signal
# =============================================================================

from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass

from loguru import logger
from pydantic import BaseModel

from .detector import Detection, DetectionResult


# ---------------------------------------------------------------------------
# Feedback Signal Types
# ---------------------------------------------------------------------------

class FeedbackMode(str, Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    TRACKING = "tracking"
    LOCKED = "locked"


class Direction(str, Enum):
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    UP_LEFT = "up_left"
    UP_RIGHT = "up_right"
    DOWN_LEFT = "down_left"
    DOWN_RIGHT = "down_right"
    ABOVE = "above"
    BELOW = "below"


@dataclass(frozen=True)
class FeedbackSignal:
    """
    Complete feedback signal for a single frame.

    Encodes proximity (distance to target) and direction as:
    - vibration: frequency + intensity (for GPIO haptic motor)
    - audio:    pitch + beat_rate + stereo_pan (for earphones)
    - mode:     current tracking state
    - direction: verbal direction for TTS announcement
    """
    mode: FeedbackMode = FeedbackMode.IDLE

    # Vibration (GPIO haptic motor)
    vibration_freq_hz: float = 0.0    # 0 = silent, 10-80 Hz
    vibration_intensity: float = 0.0  # 0.0-1.0

    # Audio (earphones)
    audio_pitch_hz: float = 0.0      # 200-800 Hz
    audio_beat_bpm: float = 0.0      # 60-360 BPM
    audio_pan: float = 0.0           # -1.0 (left) to +1.0 (right)
    audio_locked: bool = False       # "Found it!" earcon trigger

    # Direction (for TTS)
    direction: Direction = Direction.CENTER
    proximity: float = 0.0           # 0.0 (edge) to 1.0 (centered)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class FeedbackConfig(BaseModel):
    """Tunable feedback mapping parameters."""
    # Proximity thresholds (0.0 = frame edge, 1.0 = frame center)
    far_threshold: float = 0.7
    near_threshold: float = 0.3
    locked_threshold: float = 0.1

    # Vibration mapping
    vib_far_hz: float = 10.0
    vib_near_hz: float = 40.0
    vib_locked_hz: float = 80.0
    vib_far_intensity: float = 0.2
    vib_near_intensity: float = 0.6
    vib_locked_intensity: float = 1.0

    # Audio mapping
    audio_far_pitch_hz: float = 200.0
    audio_near_pitch_hz: float = 500.0
    audio_locked_pitch_hz: float = 800.0
    audio_far_bpm: float = 60.0
    audio_near_bpm: float = 180.0
    audio_locked_bpm: float = 360.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SpatialFeedbackEngine:
    """
    Converts detection results into feedback signals.

    The core innovation: instead of text descriptions, we produce
    continuous analog signals (vibration + audio) that encode spatial
    information in a way the visually impaired can perceive instantly.

    The "metal detector" metaphor:
      - Far from target  → slow, low-pitched, quiet signal
      - Getting closer   → faster, higher-pitched, louder
      - Centered (locked) → continuous strong signal + "found it" earcon
      - Direction         → stereo panning (left/right ear)
    """

    def __init__(self, config: FeedbackConfig) -> None:
        self.config = config

    def compute(
        self,
        result: DetectionResult,
        target_class: str,
    ) -> FeedbackSignal:
        """
        Compute feedback signal for current frame.

        Args:
            result: Detection results from the current frame
            target_class: The class name we're searching for

        Returns:
            FeedbackSignal with vibration + audio parameters
        """
        # Find the best matching target
        target = self._find_target(result.detections, target_class)

        if target is None:
            return FeedbackSignal(mode=FeedbackMode.SCANNING)

        # Compute proximity (0.0 = edge, 1.0 = center)
        proximity = self._compute_proximity(target, result)
        direction = self._compute_direction(target, result)

        # Determine mode based on proximity
        if proximity >= (1.0 - self.config.locked_threshold):
            mode = FeedbackMode.LOCKED
        elif proximity >= (1.0 - self.config.near_threshold):
            mode = FeedbackMode.TRACKING
        else:
            mode = FeedbackMode.SCANNING

        # Interpolate feedback parameters based on proximity
        vib_hz = self._lerp(
            self.config.vib_far_hz,
            self.config.vib_locked_hz,
            proximity,
        )
        vib_intensity = self._lerp(
            self.config.vib_far_intensity,
            self.config.vib_locked_intensity,
            proximity,
        )
        audio_pitch = self._lerp(
            self.config.audio_far_pitch_hz,
            self.config.audio_locked_pitch_hz,
            proximity,
        )
        audio_bpm = self._lerp(
            self.config.audio_far_bpm,
            self.config.audio_locked_bpm,
            proximity,
        )

        # Stereo pan based on horizontal offset
        pan = self._compute_pan(target, result)

        return FeedbackSignal(
            mode=mode,
            vibration_freq_hz=vib_hz,
            vibration_intensity=vib_intensity,
            audio_pitch_hz=audio_pitch,
            audio_beat_bpm=audio_bpm,
            audio_pan=pan,
            audio_locked=(mode == FeedbackMode.LOCKED),
            direction=direction,
            proximity=proximity,
        )

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _find_target(
        self,
        detections: list[Detection],
        target_class: str,
    ) -> Detection | None:
        """Find the highest-confidence detection matching target class."""
        matches = [
            d for d in detections
            if d.class_name.lower() == target_class.lower()
        ]
        if not matches:
            # Fuzzy match: check if target is substring of class name
            matches = [
                d for d in detections
                if target_class.lower() in d.class_name.lower()
                or d.class_name.lower() in target_class.lower()
            ]
        if not matches:
            return None
        return max(matches, key=lambda d: d.confidence)

    def _compute_proximity(
        self,
        target: Detection,
        result: DetectionResult,
    ) -> float:
        """
        Compute how close the target is to frame center.

        Returns 0.0 (at edge) to 1.0 (perfectly centered).
        Uses normalized Euclidean distance.
        """
        frame_cx, frame_cy = result.frame_center
        frame_h, frame_w = result.frame_shape

        # Half-diagonal as max possible distance
        max_dist = math.sqrt((frame_w / 2) ** 2 + (frame_h / 2) ** 2)

        dx = target.center[0] - frame_cx
        dy = target.center[1] - frame_cy
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # Normalize: 0 = centered, 1 = corner
        normalized_dist = dist / max_dist if max_dist > 0 else 0.0

        # Invert: proximity = 1 means centered
        return 1.0 - min(normalized_dist, 1.0)

    def _compute_direction(
        self,
        target: Detection,
        result: DetectionResult,
    ) -> Direction:
        """Determine verbal direction of target relative to frame center."""
        frame_cx, frame_cy = result.frame_center
        dx = target.center[0] - frame_cx
        dy = target.center[1] - frame_cy

        h, w = result.frame_shape
        # Normalize offsets
        nx = dx / (w / 2)  # -1 to +1
        ny = dy / (h / 2)  # -1 to +1 (positive = below center)

        # Direction zones
        threshold = 0.2  # 20% of frame = "center"

        if abs(nx) < threshold and abs(ny) < threshold:
            return Direction.CENTER

        # Determine primary direction
        is_left = nx < -threshold
        is_right = nx > threshold
        is_up = ny < -threshold
        is_down = ny > threshold

        if is_left and is_up:
            return Direction.UP_LEFT
        if is_right and is_up:
            return Direction.UP_RIGHT
        if is_left and is_down:
            return Direction.DOWN_LEFT
        if is_right and is_down:
            return Direction.DOWN_RIGHT
        if is_left:
            return Direction.LEFT
        if is_right:
            return Direction.RIGHT
        if is_up:
            return Direction.ABOVE
        if is_down:
            return Direction.BELOW

        return Direction.CENTER

    def _compute_pan(
        self,
        target: Detection,
        result: DetectionResult,
    ) -> float:
        """
        Compute stereo pan value.

        Returns -1.0 (fully left) to +1.0 (fully right).
        0.0 = centered.
        """
        frame_cx = result.frame_center[0]
        frame_w = result.frame_shape[1]

        dx = target.center[0] - frame_cx
        # Normalize to [-1, +1]
        pan = dx / (frame_w / 2)
        return max(-1.0, min(1.0, pan))

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation: a at t=0, b at t=1."""
        return a + (b - a) * t
