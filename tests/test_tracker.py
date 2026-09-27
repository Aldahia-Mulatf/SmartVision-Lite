"""Unit tests for ByteTrack-style association."""

from pipeline.detector import DetectionResult
from pipeline.tracker import Tracker


def _detection(
    bbox: tuple[int, int, int, int], confidence: float = 0.9
) -> DetectionResult:
    """Create one cardboard detection for a tracker test."""
    return DetectionResult(bbox, confidence, 0, "cardboard_box")


def test_tracker_preserves_id_for_overlapping_detections() -> None:
    """Verify a moving box keeps its track ID."""
    tracker = Tracker()
    first = tracker.update([_detection((100, 100, 200, 200))])
    second = tracker.update([_detection((110, 105, 210, 205))])
    assert first[0].track_id == second[0].track_id
    assert len(second[0].trajectory) == 2


def test_tracker_marks_large_area_drop_as_occlusion() -> None:
    """Verify a matched area drop is flagged instead of becoming a defect."""
    tracker = Tracker()
    tracker.update([_detection((100, 100, 200, 200))])
    result = tracker.update([_detection((115, 115, 185, 185))])
    assert result[0].is_occluded is True
