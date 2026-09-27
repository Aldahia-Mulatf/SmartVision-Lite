"""YOLO-based cardboard-box detection with strict class filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import config
from utils.helpers import BBox, sanitize_bbox, validate_frame
from utils.logger import get_logger


LOGGER = get_logger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the configured YOLO model cannot be loaded."""


@dataclass(frozen=True)
class DetectionResult:
    """One accepted product detection emitted by the detector stage."""

    bbox: BBox
    confidence: float
    class_id: int
    class_name: str


class Detector:
    """Run YOLO inference and reject people or unsupported classes."""

    def __init__(self, model: Any, product_mode: str = config.PRODUCT_MODE) -> None:
        """Create a detector around an already loaded YOLO model.

        Args:
            model: Ultralytics YOLO model or a compatible test double.
            product_mode: ``custom`` for the trained model or ``agnostic`` for
                preliminary moving-object tests.

        Raises:
            ValueError: If the product mode is unsupported.
        """
        if product_mode not in ("custom", "agnostic"):
            raise ValueError("product_mode must be 'custom' or 'agnostic'")
        self._model = model
        self._product_mode = product_mode

    @classmethod
    def from_config(cls) -> "Detector":
        """Load the configured YOLO model once for application use.

        Returns:
            A ready detector.

        Raises:
            ModelLoadError: If the configured model is absent or Ultralytics
                cannot be imported or initialized.
        """
        model_path = (
            config.CUSTOM_MODEL_PATH
            if config.PRODUCT_MODE == "custom"
            else Path(config.YOLO_MODEL_NAME)
        )
        resolved_path = model_path
        if not resolved_path.is_absolute():
            resolved_path = config.PROJECT_ROOT / resolved_path
        if config.PRODUCT_MODE == "custom" and not resolved_path.is_file():
            raise ModelLoadError(
                f"Custom model not found: {resolved_path}. "
                "Train it and copy it to models/box_detector.pt."
            )
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ModelLoadError(
                "Ultralytics is not installed. Run pip install -r requirements.txt."
            ) from error
        try:
            model = YOLO(str(resolved_path))
        except (OSError, RuntimeError, ValueError) as error:
            raise ModelLoadError(
                f"Unable to load YOLO model: {resolved_path}"
            ) from error
        LOGGER.info("Loaded YOLO model from %s", resolved_path)
        return cls(model, config.PRODUCT_MODE)

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Detect only permitted objects in one preprocessed frame.

        Args:
            frame: BGR frame.

        Returns:
            Accepted detections. People and unsupported custom classes are
            filtered before this method returns.

        Raises:
            ValueError: If ``frame`` is invalid.
            RuntimeError: If model inference fails.
        """
        validate_frame(frame)
        try:
            predictions = self._model.predict(
                source=frame,
                conf=config.CONFIDENCE_THRESHOLD,
                verbose=False,
            )
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.exception("YOLO inference failed")
            raise RuntimeError("YOLO inference failed") from error
        if not predictions:
            return []
        return self._parse_prediction(predictions[0], frame.shape[1], frame.shape[0])

    def _parse_prediction(
        self, prediction: Any, frame_width: int, frame_height: int
    ) -> list[DetectionResult]:
        """Convert one Ultralytics result into filtered dataclasses."""
        boxes = getattr(prediction, "boxes", None)
        if boxes is None:
            return []
        coordinates = self._to_numpy(getattr(boxes, "xyxy", None))
        confidences = self._to_numpy(getattr(boxes, "conf", None))
        class_ids = self._to_numpy(getattr(boxes, "cls", None))
        if coordinates is None or confidences is None or class_ids is None:
            return []
        names = getattr(prediction, "names", getattr(self._model, "names", {}))
        detections: list[DetectionResult] = []
        for raw_bbox, raw_confidence, raw_class_id in zip(
            coordinates, confidences, class_ids
        ):
            class_id = int(raw_class_id)
            class_name = self._class_name(names, class_id)
            if class_name in config.IGNORED_CLASSES:
                continue
            if self._product_mode == "custom" and not self._is_product_class(
                class_name, class_id, names
            ):
                continue
            try:
                bbox = sanitize_bbox(raw_bbox, frame_width, frame_height)
            except ValueError:
                continue
            detections.append(
                DetectionResult(
                    bbox=bbox,
                    confidence=float(raw_confidence),
                    class_id=class_id,
                    class_name=(
                        config.PRODUCT_NAME
                        if self._product_mode == "agnostic"
                        else class_name
                    ),
                )
            )
        return detections

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray | None:
        """Convert tensor-like values to NumPy arrays when present."""
        if value is None:
            return None
        detached = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        return np.asarray(detached)

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        """Resolve a model class ID from either a list or dictionary."""
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _is_product_class(class_name: str, class_id: int, names: Any) -> bool:
        """Accept the named product or a one-class custom model's class zero."""
        normalized = class_name.strip().lower().replace(" ", "_")
        if normalized == config.PRODUCT_NAME:
            return True
        class_count = len(names) if isinstance(names, (dict, list, tuple)) else 0
        return class_count == 1 and class_id == 0
