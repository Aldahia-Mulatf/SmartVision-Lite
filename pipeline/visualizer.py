"""OpenCV overlays for detections, tracking, quality, orientation, and counts."""

from __future__ import annotations

import math

import cv2
import numpy as np

import config
from pipeline.counter import CountResult
from pipeline.orientation import OrientationResult
from pipeline.quality_checker import QualityResult
from pipeline.tracker import TrackResult
from utils.helpers import validate_frame
from utils.logger import get_logger


LOGGER = get_logger(__name__)


class Visualizer:
    """Render one complete annotation layer on a copied frame."""

    def draw_frame(
        self,
        frame: np.ndarray,
        tracks: list[TrackResult],
        orientations: dict[int, OrientationResult],
        quality_results: dict[int, QualityResult],
        count_result: CountResult,
        fps: float,
    ) -> np.ndarray:
        """Draw the academic demonstration overlay.

        Args:
            frame: Preprocessed BGR frame.
            tracks: Current active tracks.
            orientations: Orientation results by track ID.
            quality_results: Quality results by track ID.
            count_result: Current counter snapshot.
            fps: Measured processing FPS.

        Returns:
            Annotated copy of the frame.
        """
        validate_frame(frame)
        annotated = frame.copy()
        line_y = min(max(config.COUNTING_LINE_Y, 0), annotated.shape[0] - 1)
        cv2.line(
            annotated,
            (0, line_y),
            (annotated.shape[1], line_y),
            config.COLORS["counting_line"],
            config.LINE_THICKNESS,
        )
        for track in tracks:
            orientation = orientations.get(track.track_id)
            quality = quality_results.get(track.track_id)
            if orientation is None or quality is None:
                continue
            self._draw_track(annotated, track, orientation, quality)
        self._draw_panel(annotated, count_result, fps)
        return annotated

    @staticmethod
    def _draw_track(
        frame: np.ndarray,
        track: TrackResult,
        orientation: OrientationResult,
        quality: QualityResult,
    ) -> None:
        """Draw one track's box, ID, quality badge, and angle arrow."""
        color = (
            config.COLORS["defective"]
            if quality.status == "DEFECTIVE"
            else config.COLORS["good"]
        )
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)
        if len(track.trajectory) > 1:
            points = np.array(track.trajectory, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame,
                [points],
                False,
                config.COLORS["tracking"],
                config.TRAJECTORY_THICKNESS,
            )
        if orientation.polygon:
            polygon = np.array(orientation.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame,
                [polygon],
                True,
                config.COLORS["orientation"],
                config.ORIENTATION_THICKNESS,
            )
        angle_radians = math.radians(orientation.angle_deg)
        end_point = (
            int(
                track.center[0]
                + config.ORIENTATION_ARROW_LENGTH * math.cos(angle_radians)
            ),
            int(
                track.center[1]
                - config.ORIENTATION_ARROW_LENGTH * math.sin(angle_radians)
            ),
        )
        cv2.arrowedLine(
            frame,
            track.center,
            end_point,
            config.COLORS["orientation"],
            config.ORIENTATION_THICKNESS,
            tipLength=0.2,
        )
        cv2.putText(
            frame,
            f"BOX #{track.track_id}",
            (x1, max(config.TEXT_THICKNESS, y1 - 28)),
            config.FONT_FACE,
            config.FONT_SCALE,
            color,
            config.TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            quality.display_text,
            (x1, max(config.TEXT_THICKNESS, y1 - 8)),
            config.FONT_FACE,
            config.FONT_SCALE,
            color,
            config.TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"angle: {orientation.angle_deg:.1f} deg",
            (x1, min(frame.shape[0] - config.TEXT_THICKNESS, y2 + 20)),
            config.FONT_FACE,
            config.FONT_SCALE,
            config.COLORS["text"],
            config.TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_panel(frame: np.ndarray, count_result: CountResult, fps: float) -> None:
        """Draw the four required statistics in the upper-left corner."""
        x, y = config.PANEL_ORIGIN
        lines = (
            f"Total: {count_result.total_count}",
            f"Good: {count_result.good_count}",
            f"Defective: {count_result.defective_count}",
            f"FPS: {fps:.1f}",
        )
        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (x, y + index * config.PANEL_LINE_HEIGHT),
                config.FONT_FACE,
                config.FONT_SCALE,
                config.COLORS["text"],
                config.TEXT_THICKNESS,
                cv2.LINE_AA,
            )
