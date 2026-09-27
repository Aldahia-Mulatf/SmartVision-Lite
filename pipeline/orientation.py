"""Cardboard-box orientation estimation using classical OpenCV geometry."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config
from utils.helpers import (
    BBox,
    Point,
    angular_deviation,
    crop_frame,
    normalize_rect_angle,
)
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class OrientationResult:
    """Orientation and conveyor-axis alignment for one tracked box."""

    track_id: int
    angle_deg: float
    deviation_deg: float
    is_aligned: bool
    center: Point
    polygon: tuple[Point, ...]
    aspect_ratio: float


class OrientationEstimator:
    """Estimate the dominant rectangle angle inside a detection crop."""

    def estimate(
        self, frame: np.ndarray, bbox: BBox, track_id: int
    ) -> OrientationResult:
        """Estimate orientation from cardboard-colored contours.

        Args:
            frame: Preprocessed BGR frame.
            bbox: Detection bounding box.
            track_id: Track identifier associated with the box.

        Returns:
            Normalized angle, deviation, alignment flag, and rotated polygon.

        Raises:
            ValueError: If the frame or bounding box is invalid.
        """
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")
        crop = crop_frame(frame, bbox)
        mask = self._cardboard_mask(crop)
        contour = self._largest_contour(mask)
        offset_x, offset_y = bbox[0], bbox[1]
        if contour is None:
            polygon = self._bbox_polygon(bbox)
            angle_deg = 0.0
            aspect_ratio = self._bbox_aspect_ratio(bbox)
        else:
            rectangle = cv2.minAreaRect(contour)
            points = cv2.boxPoints(rectangle).astype(int)
            polygon = tuple((int(x + offset_x), int(y + offset_y)) for x, y in points)
            raw_angle = float(rectangle[2])
            width, height = rectangle[1]
            if width < height:
                raw_angle += 90.0
            angle_deg = normalize_rect_angle(raw_angle)
            long_side = max(float(width), float(height))
            short_side = min(float(width), float(height))
            aspect_ratio = long_side / max(short_side, config.EPSILON)
        deviation = angular_deviation(angle_deg, config.CONVEYOR_AXIS_ANGLE)
        center = (
            int((bbox[0] + bbox[2]) / 2),
            int((bbox[1] + bbox[3]) / 2),
        )
        return OrientationResult(
            track_id=track_id,
            angle_deg=angle_deg,
            deviation_deg=deviation,
            is_aligned=deviation <= config.ORIENTATION_TOLERANCE_DEG,
            center=center,
            polygon=polygon,
            aspect_ratio=aspect_ratio,
        )

    @staticmethod
    def _cardboard_mask(crop: np.ndarray) -> np.ndarray:
        """Create a cleaned HSV mask for kraft-colored cardboard."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower = np.array(config.CARDBOARD_HSV_LOWER, dtype=np.uint8)
        upper = np.array(config.CARDBOARD_HSV_UPPER, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones(
            (config.MORPHOLOGY_KERNEL_SIZE, config.MORPHOLOGY_KERNEL_SIZE),
            dtype=np.uint8,
        )
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
        """Return the largest external contour above the configured area."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= config.MIN_CONTOUR_AREA
        ]
        return max(valid, key=cv2.contourArea) if valid else None

    @staticmethod
    def _bbox_polygon(bbox: BBox) -> tuple[Point, ...]:
        """Return an axis-aligned polygon fallback for an empty color mask."""
        x1, y1, x2, y2 = bbox
        return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))

    @staticmethod
    def _bbox_aspect_ratio(bbox: BBox) -> float:
        """Return the long-side to short-side ratio of a bounding box."""
        width = max(1, bbox[2] - bbox[0])
        height = max(1, bbox[3] - bbox[1])
        return max(width, height) / min(width, height)
