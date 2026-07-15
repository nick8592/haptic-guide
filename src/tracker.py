# =============================================================================
# HapticGuide — IOU Centroid Object Tracker
# Maintains object identity across frames for stable feedback
# =============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from .detector import Detection


# ---------------------------------------------------------------------------
# Track Data
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """A tracked object with persistent identity."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    center: tuple[float, float]
    bbox: list[float]
    last_seen: float
    lost_frames: int = 0
    age: int = 0  # Number of frames this track has existed


# ---------------------------------------------------------------------------
# IoU Centroid Tracker
# ---------------------------------------------------------------------------

class IoUCentroidTracker:
    """
    Lightweight multi-object tracker using IoU + centroid distance.

    Why not DeepSORT/ByteTrack?
    - We need <1ms tracking overhead per frame
    - DeepSORT's ReID model adds 5-10ms per frame
    - For haptic feedback, smooth identity is nice but not critical
    - Simple IoU matching is sufficient for single-target "find X" use case

    Algorithm per frame:
    1. Compute IoU between all existing tracks and new detections
    2. Hungarian algorithm for optimal matching
    3. Unmatched tracks → increment lost_frames
    4. Unmatched detections → create new tracks
    5. Remove tracks exceeding max_lost_frames
    """

    def __init__(
        self,
        max_lost_frames: int = 15,
        min_iou_match: float = 0.3,
        max_centroid_dist: float = 100.0,
    ) -> None:
        self.max_lost_frames = max_lost_frames
        self.min_iou_match = min_iou_match
        self.max_centroid_dist = max_centroid_dist

        self._tracks: list[Track] = []
        self._next_id: int = 0

    @property
    def tracks(self) -> list[Track]:
        """Current active tracks."""
        return [t for t in self._tracks if t.lost_frames < self.max_lost_frames]

    def update(self, detections: list[Detection]) -> list[Track]:
        """
        Update tracks with new detections.

        Args:
            detections: Raw detections from current frame

        Returns:
            Updated list of active tracks
        """
        now = time.time()

        if not detections:
            # No detections — age all tracks
            for track in self._tracks:
                track.lost_frames += 1
            return self.tracks

        if not self._tracks:
            # First frame — create tracks for all detections
            for det in detections:
                self._create_track(det, now)
            return self.tracks

        # Compute cost matrix: IoU between all track-detection pairs
        n_tracks = len(self._tracks)
        n_dets = len(detections)

        cost_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for i, track in enumerate(self._tracks):
            for j, det in enumerate(detections):
                iou = self._compute_iou(track.bbox, det.bbox)
                # Also consider centroid distance as secondary cost
                centroid_dist = self._centroid_distance(track.center, det.center)
                # Combined cost: prioritize IoU, penalize large centroid jumps
                if iou > 0:
                    cost_matrix[i, j] = iou - (centroid_dist / self.max_centroid_dist) * 0.1
                else:
                    cost_matrix[i, j] = 0

        # Hungarian matching
        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = \
            self._hungarian_match(cost_matrix)

        # Update matched tracks
        for ti, di in zip(matched_tracks, matched_dets):
            det = detections[di]
            track = self._tracks[ti]
            track.center = det.center
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.last_seen = now
            track.lost_frames = 0
            track.age += 1

        # Age unmatched tracks
        for ti in unmatched_tracks:
            self._tracks[ti].lost_frames += 1

        # Create new tracks for unmatched detections
        for di in unmatched_dets:
            self._create_track(detections[di], now)

        # Prune dead tracks
        self._tracks = [
            t for t in self._tracks
            if t.lost_frames < self.max_lost_frames
        ]

        return self.tracks

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _hungarian_match(
        self, cost_matrix: np.ndarray
    ) -> tuple[list, list, list, list]:
        """
        Greedy matching (simpler than scipy.optimize.linear_sum_assignment).

        For our use case (typically <10 objects), greedy is fast enough.
        """
        from scipy.optimize import linear_sum_assignment

        n_tracks, n_dets = cost_matrix.shape

        if n_tracks == 0 or n_dets == 0:
            return [], [], list(range(n_tracks)), list(range(n_dets))

        # Negate because linear_sum_assignment minimizes cost
        row_indices, col_indices = linear_sum_assignment(-cost_matrix)

        matched_tracks = []
        matched_dets = []
        unmatched_tracks = list(range(n_tracks))
        unmatched_dets = list(range(n_dets))

        for ri, ci in zip(row_indices, col_indices):
            if cost_matrix[ri, ci] >= self.min_iou_match:
                matched_tracks.append(ri)
                matched_dets.append(ci)
                unmatched_tracks.remove(ri)
                unmatched_dets.remove(ci)

        return matched_tracks, matched_dets, unmatched_tracks, unmatched_dets

    # ------------------------------------------------------------------
    # Track Management
    # ------------------------------------------------------------------

    def _create_track(self, detection: Detection, timestamp: float) -> Track:
        track = Track(
            track_id=self._next_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            center=detection.center,
            bbox=detection.bbox,
            last_seen=timestamp,
        )
        self._tracks.append(track)
        self._next_id += 1
        return track

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _centroid_distance(
        center_a: tuple[float, float],
        center_b: tuple[float, float],
    ) -> float:
        """Euclidean distance between two centroids."""
        return ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5
