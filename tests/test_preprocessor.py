"""Unit tests for classical preprocessing."""

import cv2
import numpy as np

from pipeline.preprocessor import Preprocessor


def test_preprocessor_resizes_without_mutating_input() -> None:
    """Verify output dimensions and preservation of the original frame."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[20:80, 30:100] = (42, 84, 150)
    original = frame.copy()
    result = Preprocessor().process(frame)
    assert result.frame.shape == (720, 1280, 3)
    assert np.array_equal(frame, original)
    assert result.blur_score >= 0.0


def test_preprocessor_handles_sharp_edges() -> None:
    """Verify that a structured image produces a measurable focus score."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (900, 500), (80, 120, 180), -1)
    result = Preprocessor().process(frame)
    assert result.frame.shape[:2] == (720, 1280)
    assert isinstance(result.is_blurry, bool)
