"""Unit tests for minAreaRect orientation estimation."""

import cv2
import numpy as np

from pipeline.orientation import OrientationEstimator


def test_horizontal_cardboard_is_aligned() -> None:
    """Verify a horizontal kraft rectangle aligns with a zero-degree axis."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (500, 320), (42, 84, 150), -1)
    result = OrientationEstimator().estimate(frame, (200, 200, 500, 320), 1)
    assert result.track_id == 1
    assert 0.0 <= result.angle_deg < 90.0
    assert result.is_aligned is True
    assert result.deviation_deg <= 15.0
