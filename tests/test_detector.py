"""Unit tests for strict detector class filtering."""

import numpy as np

from pipeline.detector import Detector


class FakeBoxes:
    """Minimal Ultralytics Boxes replacement for detector tests."""

    def __init__(self) -> None:
        """Create box, confidence, and class arrays."""
        self.xyxy = np.array(
            [[10, 10, 80, 80], [100, 10, 160, 80], [200, 10, 260, 80]],
            dtype=np.float32,
        )
        self.conf = np.array([0.95, 0.90, 0.88], dtype=np.float32)
        self.cls = np.array([0, 1, 2], dtype=np.float32)


class FakePrediction:
    """Minimal result object containing three classes."""

    def __init__(self) -> None:
        """Create names for product, worker, and another object."""
        self.boxes = FakeBoxes()
        self.names = {0: "cardboard_box", 1: "person", 2: "forklift"}


class FakeModel:
    """Minimal model exposing the predict method used by Detector."""

    def predict(
        self, source: np.ndarray, conf: float, verbose: bool
    ) -> list[FakePrediction]:
        """Return one deterministic prediction."""
        return [FakePrediction()]


def test_custom_detector_keeps_only_cardboard_boxes() -> None:
    """Verify person and unrelated classes never leave the detector stage."""
    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    results = Detector(FakeModel(), product_mode="custom").detect(frame)
    assert len(results) == 1
    assert results[0].class_name == "cardboard_box"
    assert results[0].bbox == (10, 10, 80, 80)


def test_agnostic_detector_maps_surviving_objects_to_product_name() -> None:
    """Verify agnostic mode remains useful for preliminary object tests."""
    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    results = Detector(FakeModel(), product_mode="agnostic").detect(frame)
    assert len(results) == 2
    assert all(result.class_name == "cardboard_box" for result in results)
