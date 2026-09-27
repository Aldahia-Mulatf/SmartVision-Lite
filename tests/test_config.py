"""Tests for the central SmartVision Lite configuration."""

from pathlib import Path

import config


def test_core_configuration_values() -> None:
    """Verify the product mode and model path required by the final demo."""
    assert config.PRODUCT_NAME == "cardboard_box"
    assert config.PRODUCT_MODE == "custom"
    assert config.CUSTOM_MODEL_PATH == Path("models/box_detector.pt")
    assert config.IGNORED_CLASSES == ("person",)


def test_quality_safety_thresholds() -> None:
    """Verify the occlusion and size-check guardrails are configured."""
    assert config.MIN_SAMPLES_FOR_SIZE_CHECK == 5
    assert config.OCCLUSION_AREA_DROP_RATIO == 0.50
    assert config.MIN_LABEL_RATIO == 0.02
    assert config.BLUR_THRESHOLD > 0.0
