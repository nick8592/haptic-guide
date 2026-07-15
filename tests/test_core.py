# =============================================================================
# HapticGuide — Tests
# =============================================================================

import numpy as np
import pytest
import time

from src.detector import Detection, DetectionResult, DetectorConfig, InferenceBackend
from src.feedback_engine import (
    Direction,
    FeedbackConfig,
    FeedbackMode,
    FeedbackSignal,
    SpatialFeedbackEngine,
)
from src.tracker import IoUCentroidTracker, Track


# ---------------------------------------------------------------------------
# Detector Tests (unit-level, no model needed)
# ---------------------------------------------------------------------------

class TestDetection:
    def test_detection_creation(self):
        det = Detection(
            class_id=67,
            class_name="cell phone",
            confidence=0.92,
            bbox=[100.0, 200.0, 300.0, 400.0],
            center=(200.0, 300.0),
        )
        assert det.class_name == "cell phone"
        assert det.confidence == 0.92
        assert det.center == (200.0, 300.0)

    def test_detection_result_center(self):
        result = DetectionResult(
            detections=[],
            frame_shape=(480, 640),
            inference_ms=15.0,
            timestamp=time.time(),
        )
        assert result.frame_center == (320.0, 240.0)


# ---------------------------------------------------------------------------
# Feedback Engine Tests
# ---------------------------------------------------------------------------

class TestSpatialFeedbackEngine:
    @pytest.fixture
    def engine(self):
        return SpatialFeedbackEngine(FeedbackConfig())

    @pytest.fixture
    def sample_result(self):
        """Detection result with a cell phone at center."""
        return DetectionResult(
            detections=[
                Detection(
                    class_id=67,
                    class_name="cell phone",
                    confidence=0.9,
                    bbox=[260.0, 200.0, 380.0, 340.0],
                    center=(320.0, 270.0),  # Near center of 640x480
                )
            ],
            frame_shape=(480, 640),
            inference_ms=10.0,
            timestamp=time.time(),
        )

    def test_no_target_returns_scanning(self, engine, sample_result):
        signal = engine.compute(sample_result, "keys")
        assert signal.mode == FeedbackMode.SCANNING

    def test_target_at_center_returns_locked(self, engine):
        """Target perfectly at center should be LOCKED."""
        result = DetectionResult(
            detections=[
                Detection(
                    class_id=67,
                    class_name="cell phone",
                    confidence=0.9,
                    bbox=[280.0, 220.0, 360.0, 260.0],
                    center=(320.0, 240.0),  # Exact center
                )
            ],
            frame_shape=(480, 640),
            inference_ms=5.0,
            timestamp=time.time(),
        )
        signal = engine.compute(result, "cell phone")
        assert signal.mode == FeedbackMode.LOCKED
        assert signal.proximity > 0.9

    def test_target_on_left_edge(self, engine):
        """Target at left edge should pan left."""
        result = DetectionResult(
            detections=[
                Detection(
                    class_id=67,
                    class_name="cell phone",
                    confidence=0.8,
                    bbox=[10.0, 220.0, 50.0, 260.0],
                    center=(30.0, 240.0),  # Far left
                )
            ],
            frame_shape=(480, 640),
            inference_ms=8.0,
            timestamp=time.time(),
        )
        signal = engine.compute(result, "cell phone")
        assert signal.mode == FeedbackMode.SCANNING
        assert signal.audio_pan < -0.5
        assert signal.direction == Direction.LEFT

    def test_vibration_intensity_increases_with_proximity(self, engine):
        """Closer target → stronger vibration."""
        # Far target
        far_result = DetectionResult(
            detections=[
                Detection(
                    class_id=67,
                    class_name="cell phone",
                    confidence=0.8,
                    bbox=[10.0, 10.0, 50.0, 50.0],
                    center=(30.0, 30.0),
                )
            ],
            frame_shape=(480, 640),
            inference_ms=5.0,
            timestamp=time.time(),
        )
        far_signal = engine.compute(far_result, "cell phone")

        # Near target
        near_result = DetectionResult(
            detections=[
                Detection(
                    class_id=67,
                    class_name="cell phone",
                    confidence=0.9,
                    bbox=[290.0, 220.0, 350.0, 260.0],
                    center=(320.0, 240.0),
                )
            ],
            frame_shape=(480, 640),
            inference_ms=5.0,
            timestamp=time.time(),
        )
        near_signal = engine.compute(near_result, "cell phone")

        assert near_signal.vibration_intensity > far_signal.vibration_intensity
        assert near_signal.audio_pitch_hz > far_signal.audio_pitch_hz

    def test_audio_pan_direction(self, engine):
        """Left target → negative pan, right target → positive pan."""
        # Right side
        right_result = DetectionResult(
            detections=[
                Detection(
                    class_id=67,
                    class_name="cell phone",
                    confidence=0.8,
                    bbox=[500.0, 220.0, 540.0, 260.0],
                    center=(520.0, 240.0),
                )
            ],
            frame_shape=(480, 640),
            inference_ms=5.0,
            timestamp=time.time(),
        )
        signal = engine.compute(right_result, "cell phone")
        assert signal.audio_pan > 0.5
        assert signal.direction == Direction.RIGHT


# ---------------------------------------------------------------------------
# Tracker Tests
# ---------------------------------------------------------------------------

class TestIoUCentroidTracker:
    @pytest.fixture
    def tracker(self):
        return IoUCentroidTracker(max_lost_frames=5, min_iou_match=0.3)

    def test_first_frame_creates_tracks(self, tracker):
        dets = [
            Detection(class_id=0, class_name="person", confidence=0.9,
                      bbox=[100, 100, 200, 200], center=(150, 150)),
        ]
        tracks = tracker.update(dets)
        assert len(tracks) == 1
        assert tracks[0].track_id == 0
        assert tracks[0].class_name == "person"

    def test_matching_across_frames(self, tracker):
        det1 = Detection(class_id=0, class_name="person", confidence=0.9,
                         bbox=[100, 100, 200, 200], center=(150, 150))
        tracker.update([det1])

        # Same location, slight shift
        det2 = Detection(class_id=0, class_name="person", confidence=0.85,
                         bbox=[105, 105, 205, 205], center=(155, 155))
        tracks = tracker.update([det2])
        assert len(tracks) == 1
        assert tracks[0].track_id == 0  # Same ID

    def test_lost_tracks_removed(self, tracker):
        det = Detection(class_id=0, class_name="person", confidence=0.9,
                        bbox=[100, 100, 200, 200], center=(150, 150))
        tracker.update([det])

        # No detections for max_lost_frames
        for _ in range(10):
            tracker.update([])

        tracks = tracker.tracks
        assert len(tracks) == 0  # Track should be removed

    def test_new_detection_gets_new_id(self, tracker):
        det1 = Detection(class_id=0, class_name="person", confidence=0.9,
                         bbox=[100, 100, 200, 200], center=(150, 150))
        tracker.update([det1])

        # Completely different location
        det2 = Detection(class_id=67, class_name="cell phone", confidence=0.8,
                         bbox=[400, 300, 500, 400], center=(450, 350))
        tracks = tracker.update([det2])
        assert len(tracks) == 2  # Both tracked
        ids = [t.track_id for t in tracks]
        assert 0 in ids
        assert 1 in ids


# ---------------------------------------------------------------------------
# Integration Test (requires model — marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestDetectorIntegration:
    """Integration tests that load actual YOLO26 model."""

    def test_yolo26_pytorch_inference(self):
        """Test PyTorch backend inference on a dummy frame."""
        from src.detector import YOLO26Detector, DetectorConfig, InferenceBackend

        config = DetectorConfig(
            variant="yolo26n",
            backend=InferenceBackend.PYTORCH,
            confidence_threshold=0.5,
        )
        detector = YOLO26Detector(config)
        detector.load()

        # Create a test frame (blank — no detections expected, but should not crash)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result.inference_ms > 0
        assert result.frame_shape == (480, 640)

        detector.unload()
