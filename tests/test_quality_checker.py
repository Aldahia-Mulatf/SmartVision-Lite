"""Unit tests for frame-aware quality checks."""

import numpy as np

import config
from pipeline.orientation import OrientationResult
from pipeline.quality_checker import QualityChecker
from pipeline.tracker import TrackResult


def _track() -> TrackResult:
    """Create a large square track eligible for label inspection."""
    return TrackResult(
        track_id=1,
        bbox=(100, 100, 300, 300),
        confidence=0.9,
        class_name="cardboard_box",
        center=(200, 200),
        area=40000,
        trajectory=((200, 200),),
        is_occluded=False,
    )


def _orientation() -> OrientationResult:
    """Create an aligned orientation result."""
    return OrientationResult(1, 0.0, 0.0, True, (200, 200), (), 1.0)


def test_blurry_frame_does_not_add_quality_sample() -> None:
    """Verify blur skips quality evidence while returning a safe result."""
    checker = QualityChecker()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = checker.update(_track(), frame, _orientation(), 0.0, True)
    assert result.sample_count == 0
    assert result.status == "OK"


def test_missing_label_is_detected_over_clear_frames() -> None:
    """Verify repeated absent labels create the specified defect reason."""
    checker = QualityChecker()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(3):
        checker.update(
            _track(),
            frame,
            _orientation(),
            config.BLUR_THRESHOLD + 1.0,
            False,
        )
    result = checker.finalize(1)
    assert result.status == "DEFECTIVE"
    assert "missing_label" in result.reasons
