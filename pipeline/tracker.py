"""ByteTrack-style association for cardboard-box detections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import config
from pipeline.detector import DetectionResult
from utils.helpers import BBox, Point, bbox_area, bbox_center
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TrackResult:
    """Current state of one tracked cardboard box."""

    track_id: int
    bbox: BBox
    confidence: float
    class_name: str
    center: Point
    area: int
    trajectory: tuple[Point, ...]
    is_occluded: bool
    missed_frames: int = 0


@dataclass
class _TrackState:
    """Mutable internal state used by the lightweight ByteTrack association."""

    track_id: int
    bbox: BBox
    confidence: float
    class_name: str
    trajectory: list[Point] = field(default_factory=list)
    missed_frames: int = 0
    previous_area: int = 0
    is_occluded: bool = False


class Tracker:
    """Associate detections using high/low confidence IoU matching.

    This is a dependency-free ByteTrack-style implementation: high-confidence
    detections create tracks, while lower-confidence detections may recover an
    existing track. It keeps the stage independently testable from YOLO.
    """

    def __init__(self) -> None:
        """Initialize an empty tracker and its monotonic ID counter."""
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    def update(self, detections: Iterable[DetectionResult]) -> list[TrackResult]:
        """Update active tracks from one frame's detections.

        Args:
            detections: Detection results from the detector stage.

        Returns:
            Active track results for this frame.
        """
        all_detections = list(detections)
        high_confidence = [
            detection
            for detection in all_detections
            if detection.confidence >= config.TRACK_HIGH_CONFIDENCE_THRESHOLD
        ]
        low_confidence = [
            detection
            for detection in all_detections
            if config.TRACK_LOW_CONFIDENCE_THRESHOLD
            <= detection.confidence
            < config.TRACK_HIGH_CONFIDENCE_THRESHOLD
        ]
        unmatched_track_ids = set(self._tracks)
        self._associate(high_confidence, unmatched_track_ids)
        self._associate(low_confidence, unmatched_track_ids)
        self._age_unmatched(unmatched_track_ids)
        self._create_new_tracks(high_confidence)
        return [
            self._to_result(track)
            for track in self._tracks.values()
            if track.missed_frames == 0
        ]

    def _associate(
        self, detections: list[DetectionResult], unmatched_track_ids: set[int]
    ) -> None:
        """Greedily associate detections with the best available IoU."""
        candidates = sorted(detections, key=lambda item: item.confidence, reverse=True)
        for detection in candidates:
            best_track_id: int | None = None
            best_iou = config.TRACK_IOU_MATCH_THRESHOLD
            for track_id in unmatched_track_ids:
                track = self._tracks[track_id]
                if track.class_name != detection.class_name:
                    continue
                overlap = self._iou(track.bbox, detection.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track_id = track_id
            if best_track_id is None:
                continue
            self._update_track(self._tracks[best_track_id], detection)
            unmatched_track_ids.remove(best_track_id)

    def _update_track(self, track: _TrackState, detection: DetectionResult) -> None:
        """Update one track and flag a sudden area drop as occlusion."""
        new_area = bbox_area(detection.bbox)
        area_drop = (
            track.previous_area > 0
            and new_area
            < track.previous_area * (1.0 - config.OCCLUSION_AREA_DROP_RATIO)
        )
        track.is_occluded = area_drop
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.missed_frames = 0
        track.previous_area = new_area
        track.trajectory.append(bbox_center(detection.bbox))
        del track.trajectory[: -config.TRACK_TRAJECTORY_LENGTH]

    def _age_unmatched(self, unmatched_track_ids: set[int]) -> None:
        """Age and retire tracks absent for too many frames."""
        for track_id in list(unmatched_track_ids):
            track = self._tracks[track_id]
            track.missed_frames += 1
            track.is_occluded = False
            if track.missed_frames > config.TRACK_MAX_MISSED_FRAMES:
                del self._tracks[track_id]

    def _create_new_tracks(self, detections: list[DetectionResult]) -> None:
        """Create tracks for high-confidence detections left unmatched."""
        matched_boxes = [self._tracks[track_id].bbox for track_id in self._tracks]
        for detection in detections:
            if any(
                self._iou(detection.bbox, box) > config.TRACK_IOU_MATCH_THRESHOLD
                for box in matched_boxes
            ):
                continue
            center = bbox_center(detection.bbox)
            self._tracks[self._next_track_id] = _TrackState(
                track_id=self._next_track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_name=detection.class_name,
                trajectory=[center],
                previous_area=bbox_area(detection.bbox),
            )
            self._next_track_id += 1
            matched_boxes.append(detection.bbox)

    @staticmethod
    def _iou(first: BBox, second: BBox) -> float:
        """Compute intersection-over-union for two boxes."""
        first_x1, first_y1, first_x2, first_y2 = first
        second_x1, second_y1, second_x2, second_y2 = second
        intersection_x1 = max(first_x1, second_x1)
        intersection_y1 = max(first_y1, second_y1)
        intersection_x2 = min(first_x2, second_x2)
        intersection_y2 = min(first_y2, second_y2)
        intersection = bbox_area(
            (intersection_x1, intersection_y1, intersection_x2, intersection_y2)
        )
        union = bbox_area(first) + bbox_area(second) - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _to_result(track: _TrackState) -> TrackResult:
        """Convert mutable state to an immutable stage output."""
        return TrackResult(
            track_id=track.track_id,
            bbox=track.bbox,
            confidence=track.confidence,
            class_name=track.class_name,
            center=bbox_center(track.bbox),
            area=bbox_area(track.bbox),
            trajectory=tuple(track.trajectory),
            is_occluded=track.is_occluded,
            missed_frames=track.missed_frames,
        )
