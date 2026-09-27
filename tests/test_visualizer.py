"""Unit tests for non-mutating visualization."""

import numpy as np

from pipeline.counter import CountResult
from pipeline.orientation import OrientationResult
from pipeline.quality_checker import QualityResult
from pipeline.tracker import TrackResult
from pipeline.visualizer import Visualizer


def test_visualizer_returns_annotated_copy() -> None:
    """Verify overlays do not alter the source frame in place."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    original = frame.copy()
    track = TrackResult(
        1,
        (100, 100, 300, 300),
        0.9,
        "cardboard_box",
        (200, 200),
        40000,
        ((200, 200),),
        False,
    )
    orientation = OrientationResult(1, 0.0, 0.0, True, (200, 200), (), 1.0)
    quality = QualityResult(1, "OK", (), 1.0, 0.0, None, False, 1)
    counts = CountResult(0, 0, 0, frozenset())
    annotated = Visualizer().draw_frame(
        frame,
        [track],
        {1: orientation},
        {1: quality},
        counts,
        30.0,
    )
    assert annotated.shape == frame.shape
    assert np.array_equal(frame, original)
    assert not np.array_equal(annotated, original)
