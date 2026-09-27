"""Small dependency-free helpers shared by pipeline stages."""

from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np

import config


BBox = tuple[int, int, int, int]
Point = tuple[int, int]


def validate_frame(frame: np.ndarray) -> None:
    """Validate an OpenCV frame before it enters a pipeline stage.

    Args:
        frame: BGR image represented as a NumPy array.

    Raises:
        ValueError: If the frame is missing, empty, or has an unsupported shape.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame must be a non-empty NumPy array")
    if frame.ndim not in (2, 3):
        raise ValueError("frame must have two or three dimensions")


def sanitize_bbox(bbox: Sequence[float], frame_width: int, frame_height: int) -> BBox:
    """Clamp a floating-point bounding box to image boundaries.

    Args:
        bbox: Bounding box in ``x1, y1, x2, y2`` format.
        frame_width: Image width in pixels.
        frame_height: Image height in pixels.

    Returns:
        A non-negative integer bounding box.

    Raises:
        ValueError: If the box has the wrong shape or no positive area.
    """
    if len(bbox) != 4:
        raise ValueError("bbox must contain x1, y1, x2, y2")
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive area")
    return x1, y1, x2, y2


def bbox_area(bbox: BBox) -> int:
    """Return the area of an ``x1, y1, x2, y2`` bounding box."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center(bbox: BBox) -> Point:
    """Return the integer center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def crop_frame(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    """Return a copied crop for a bounding box without modifying the frame."""
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("bbox does not intersect the frame")
    return crop.copy()


def safe_median(values: Iterable[float]) -> float | None:
    """Return a median or ``None`` when no numeric values are available."""
    materialized = [float(value) for value in values]
    return float(median(materialized)) if materialized else None


def normalize_rect_angle(angle_deg: float) -> float:
    """Normalize a rectangle angle to the half-open interval ``[0, 90)``."""
    normalized = float(angle_deg) % 90.0
    return 0.0 if abs(normalized - 90.0) < config.EPSILON else normalized


def angular_deviation(angle_deg: float, axis_deg: float) -> float:
    """Return the smallest rectangle-orientation deviation in degrees."""
    difference = abs((float(angle_deg) - float(axis_deg)) % 90.0)
    return min(difference, 90.0 - difference)


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamped_output_path(directory: Path, prefix: str, suffix: str) -> Path:
    """Build a timestamped output path using the configured naming convention."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"{prefix}_{timestamp}{suffix}"


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to RGB without mutating the input."""
    validate_frame(frame)
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
