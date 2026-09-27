"""Classical quality checks for tracked cardboard boxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

import config
from pipeline.orientation import OrientationResult
from pipeline.tracker import TrackResult
from utils.helpers import crop_frame, safe_median
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class QualityResult:
    """Current or final quality decision for one track."""

    track_id: int
    status: str
    reasons: tuple[str, ...]
    solidity: float | None
    label_ratio: float | None
    size_deviation: float | None
    is_blurry: bool
    sample_count: int
    is_final: bool = False

    @property
    def display_text(self) -> str:
        """Return the compact label used by the visualizer."""
        if self.status == "OK":
            return "OK"
        return "DEFECT: " + ", ".join(self.reasons)


@dataclass
class _Observation:
    """Best-focus and frame-ratio measurements for one track."""

    blur_score: float = -1.0
    solidity: float | None = None
    label_ratio: float | None = None
    label_checked_frames: int = 0
    label_missing_frames: int = 0
    orientation_aligned: bool = True
    aspect_ratio: float | None = None
    sample_count: int = 0
    areas: list[int] = field(default_factory=list)


class QualityChecker:
    """Accumulate clear-frame evidence and finalize track-level defects."""

    def __init__(self) -> None:
        """Initialize per-track observations and completed-size history."""
        self._observations: dict[int, _Observation] = {}
        self._completed_areas: list[float] = []
        self._latest_results: dict[int, QualityResult] = {}

    def update(
        self,
        track: TrackResult,
        frame: np.ndarray,
        orientation: OrientationResult,
        blur_score: float,
        is_blurry: bool,
    ) -> QualityResult:
        """Evaluate a track without using blurred or occluded frames.

        Args:
            track: Current tracking result.
            frame: Preprocessed BGR frame.
            orientation: Current orientation result.
            blur_score: Laplacian variance for the frame.
            is_blurry: Whether the blur threshold was crossed.

        Returns:
            Provisional quality result. The final result is produced when the
            track is finalized.
        """
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")
        observation = self._observations.setdefault(track.track_id, _Observation())
        if is_blurry or track.is_occluded:
            result = self._build_result(track.track_id, observation, is_blurry, False)
            self._latest_results[track.track_id] = result
            return result
        observation.areas.append(track.area)
        observation.sample_count += 1
        observation.orientation_aligned = (
            observation.orientation_aligned and orientation.is_aligned
        )
        observation.areas = observation.areas[-config.MEDIAN_WINDOW_SIZE :]
        label_ratio: float | None = None
        if track.area >= config.MIN_AREA_FOR_LABEL_CHECK:
            crop = crop_frame(frame, track.bbox)
            label_ratio = self._label_ratio(crop)
            observation.label_checked_frames += 1
            if label_ratio < config.MIN_LABEL_RATIO:
                observation.label_missing_frames += 1
        if blur_score >= observation.blur_score:
            observation.blur_score = blur_score
            observation.solidity = self._solidity(frame, track)
            observation.label_ratio = label_ratio
            observation.aspect_ratio = orientation.aspect_ratio
        result = self._build_result(track.track_id, observation, False, False)
        self._latest_results[track.track_id] = result
        return result

    def finalize(self, track_id: int) -> QualityResult:
        """Finalize one track and add its stable area to the reference history."""
        observation = self._observations.setdefault(track_id, _Observation())
        result = self._build_result(track_id, observation, False, True)
        representative_area = safe_median(observation.areas)
        if representative_area is not None:
            reference = self._size_reference()
            if reference is not None:
                deviation = abs(representative_area - reference) / max(
                    reference, config.EPSILON
                )
                if deviation > config.SIZE_DEVIATION_MAX:
                    result = self._with_reason(result, "size_anomaly", deviation)
            self._completed_areas.append(representative_area)
            self._completed_areas = self._completed_areas[-config.MEDIAN_WINDOW_SIZE :]
        self._latest_results[track_id] = result
        return result

    def finalize_all(self) -> dict[int, QualityResult]:
        """Finalize every observed track and return all final results."""
        return {track_id: self.finalize(track_id) for track_id in self._observations}

    def _build_result(
        self,
        track_id: int,
        observation: _Observation,
        is_blurry: bool,
        is_final: bool,
    ) -> QualityResult:
        """Build a result from current evidence, excluding skipped frames."""
        reasons: list[str] = []
        aspect_out_of_range = (
            observation.aspect_ratio is not None
            and not (
                config.BOX_ASPECT_RANGE[0]
                <= observation.aspect_ratio
                <= config.BOX_ASPECT_RANGE[1]
            )
        )
        if (
            observation.solidity is not None
            and observation.solidity < config.SOLIDITY_MIN
        ) or aspect_out_of_range:
            reasons.append("crushed")
        if (
            observation.label_checked_frames > 0
            and observation.label_missing_frames / observation.label_checked_frames
            >= config.LABEL_DEFECT_FRAME_RATIO
        ):
            reasons.append("missing_label")
        if not observation.orientation_aligned:
            reasons.append("misaligned")
        unique_reasons = tuple(dict.fromkeys(reasons))
        return QualityResult(
            track_id=track_id,
            status="DEFECTIVE" if unique_reasons else "OK",
            reasons=unique_reasons,
            solidity=observation.solidity,
            label_ratio=observation.label_ratio,
            size_deviation=None,
            is_blurry=is_blurry,
            sample_count=observation.sample_count,
            is_final=is_final,
        )

    def _size_reference(self) -> float | None:
        """Return the median of the latest completed tracks after warm-up."""
        if len(self._completed_areas) < config.MIN_SAMPLES_FOR_SIZE_CHECK:
            return None
        return safe_median(self._completed_areas[-config.MEDIAN_WINDOW_SIZE :])

    @staticmethod
    def _label_ratio(crop: np.ndarray) -> float:
        """Measure the white-label fraction inside a detection crop."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower = np.array(config.LABEL_HSV_LOWER, dtype=np.uint8)
        upper = np.array(config.LABEL_HSV_UPPER, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        return float(np.count_nonzero(mask)) / max(mask.size, 1)

    @staticmethod
    def _solidity(frame: np.ndarray, track: TrackResult) -> float | None:
        """Compute largest-contour solidity inside a tracked crop."""
        crop = crop_frame(frame, track.bbox)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(config.CARDBOARD_HSV_LOWER, dtype=np.uint8),
            np.array(config.CARDBOARD_HSV_UPPER, dtype=np.uint8),
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if area < config.MIN_CONTOUR_AREA or hull_area <= config.EPSILON:
            return None
        return float(area / hull_area)

    @staticmethod
    def _with_reason(
        result: QualityResult, reason: str, size_deviation: float
    ) -> QualityResult:
        """Return a result with a newly discovered size defect."""
        reasons = tuple(dict.fromkeys((*result.reasons, reason)))
        return QualityResult(
            track_id=result.track_id,
            status="DEFECTIVE",
            reasons=reasons,
            solidity=result.solidity,
            label_ratio=result.label_ratio,
            size_deviation=size_deviation,
            is_blurry=result.is_blurry,
            sample_count=result.sample_count,
            is_final=result.is_final,
        )
