"""Classical image preprocessing for conveyor-belt frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config
from utils.helpers import validate_frame
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class PreprocessedFrame:
    """Preprocessed frame and its focus-quality measurements."""

    frame: np.ndarray
    blur_score: float
    is_blurry: bool


class Preprocessor:
    """Resize, denoise, illuminate, and score one BGR video frame."""

    def __init__(self) -> None:
        """Initialize reusable CLAHE state."""
        self._clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID_SIZE,
        )

    def process(self, frame: np.ndarray) -> PreprocessedFrame:
        """Preprocess a frame without changing the caller's array.

        Args:
            frame: Original BGR frame.

        Returns:
            Resized, blurred, CLAHE-enhanced frame and blur metadata.

        Raises:
            ValueError: If ``frame`` is invalid.
        """
        validate_frame(frame)
        working_frame = cv2.resize(
            frame.copy(),
            (config.FRAME_WIDTH, config.FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        smoothed = cv2.GaussianBlur(
            working_frame,
            config.PREPROCESS_GAUSSIAN_KERNEL,
            config.PREPROCESS_GAUSSIAN_SIGMA,
        )
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        enhanced_lightness = self._clahe.apply(lightness)
        enhanced = cv2.merge((enhanced_lightness, channel_a, channel_b))
        processed = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = blur_score < config.BLUR_THRESHOLD
        return PreprocessedFrame(processed, blur_score, is_blurry)
