"""Orchestration for complete-video processing before Streamlit display."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

import config
from pipeline.counter import CountResult, Counter
from pipeline.detector import Detector
from pipeline.orientation import OrientationEstimator
from pipeline.preprocessor import Preprocessor
from pipeline.quality_checker import QualityChecker, QualityResult
from pipeline.tracker import Tracker
from pipeline.visualizer import Visualizer
from utils.logger import get_logger
from utils.video_io import (
    create_video_writer,
    iter_video_frames,
    read_video_metadata,
    validate_video_path,
)


LOGGER = get_logger(__name__)
ProgressCallback = Callable[[int, int], None]


class ProcessingError(RuntimeError):
    """Raised when complete-video processing fails."""


@dataclass(frozen=True)
class ProcessingSummary:
    """Final metrics and output information for one processed video."""

    output_path: Path
    frames_processed: int
    total_count: int
    good_count: int
    defective_count: int
    average_fps: float
    quality_results: dict[int, QualityResult]


def process_video(
    input_path: Path,
    output_path: Path,
    detector: Detector,
    progress_callback: ProgressCallback | None = None,
) -> ProcessingSummary:
    """Process a complete video through all seven pipeline stages.

    Args:
        input_path: Source pre-recorded conveyor video.
        output_path: Destination annotated MP4.
        detector: Loaded detector resource.
        progress_callback: Optional callback receiving current and total frame
            counts after each written frame.

    Returns:
        Final processing summary.

    Raises:
        ProcessingError: If an OpenCV or pipeline error stops processing.
        ValueError: If input arguments are invalid.
    """
    source = validate_video_path(Path(input_path))
    if detector is None:
        raise ValueError("detector is required")
    metadata = read_video_metadata(source)
    writer = None
    preprocessor = Preprocessor()
    tracker = Tracker()
    orientation_estimator = OrientationEstimator()
    quality_checker = QualityChecker()
    counter = Counter()
    visualizer = Visualizer()
    frame_count = 0
    started_at = time.perf_counter()
    last_count = CountResult(0, 0, 0, frozenset())
    latest_quality_results: dict[int, QualityResult] = {}
    try:
        writer = create_video_writer(Path(output_path))
        for _, frame in iter_video_frames(source):
            processed = preprocessor.process(frame)
            detections = detector.detect(processed.frame)
            tracks = tracker.update(detections)
            orientations = {
                track.track_id: orientation_estimator.estimate(
                    processed.frame, track.bbox, track.track_id
                )
                for track in tracks
            }
            latest_quality_results = {
                track.track_id: quality_checker.update(
                    track,
                    processed.frame,
                    orientations[track.track_id],
                    processed.blur_score,
                    processed.is_blurry,
                )
                for track in tracks
            }
            last_count = counter.update(tracks, latest_quality_results)
            elapsed = time.perf_counter() - started_at
            measured_fps = (frame_count + 1) / max(elapsed, config.EPSILON)
            annotated = visualizer.draw_frame(
                processed.frame,
                tracks,
                orientations,
                latest_quality_results,
                last_count,
                measured_fps,
            )
            writer.write(annotated)
            frame_count += 1
            if progress_callback is not None:
                progress_callback(frame_count, metadata.frame_count)
        final_quality_results = quality_checker.finalize_all()
        last_count = counter.update_quality(final_quality_results)
        average_fps = frame_count / max(
            time.perf_counter() - started_at, config.EPSILON
        )
        LOGGER.info(
            "Completed video: frames=%s total=%s good=%s defective=%s",
            frame_count,
            last_count.total_count,
            last_count.good_count,
            last_count.defective_count,
        )
        return ProcessingSummary(
            output_path=Path(output_path),
            frames_processed=frame_count,
            total_count=last_count.total_count,
            good_count=last_count.good_count,
            defective_count=last_count.defective_count,
            average_fps=average_fps,
            quality_results=final_quality_results,
        )
    except (cv2.error, OSError, RuntimeError, ValueError) as error:
        LOGGER.exception("Video processing failed for %s", source)
        raise ProcessingError(f"Video processing failed: {error}") from error
    finally:
        if writer is not None:
            writer.release()
