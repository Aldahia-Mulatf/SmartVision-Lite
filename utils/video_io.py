"""Validated OpenCV video input and output helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

import config
from utils.helpers import ensure_directory
from utils.logger import get_logger


LOGGER = get_logger(__name__)


class VideoValidationError(ValueError):
    """Raised when a user-provided video fails validation."""


class VideoIOError(RuntimeError):
    """Raised when OpenCV cannot open, read, or write a video."""


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata required to process and write a video."""

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        """Return the estimated duration in seconds."""
        return self.frame_count / max(self.fps, config.EPSILON)


def validate_video_path(video_path: Path) -> Path:
    """Validate a local video path by extension, size, and OpenCV readability.

    Args:
        video_path: Candidate video path.

    Returns:
        The resolved path.

    Raises:
        VideoValidationError: If the path is invalid or unsupported.
    """
    path = Path(video_path)
    if not path.is_file():
        raise VideoValidationError(f"Video file does not exist: {path}")
    if path.suffix.lower() not in config.SUPPORTED_VIDEO_EXTENSIONS:
        extensions = ", ".join(config.SUPPORTED_VIDEO_EXTENSIONS)
        raise VideoValidationError(f"Unsupported video extension. Use: {extensions}")
    max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if path.stat().st_size > max_bytes:
        raise VideoValidationError(
            f"Video exceeds the {config.MAX_UPLOAD_SIZE_MB} MB size limit"
        )
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    capture.release()
    if not opened:
        raise VideoValidationError(f"OpenCV cannot open the video: {path}")
    return path.resolve()


def read_video_metadata(video_path: Path) -> VideoMetadata:
    """Read video dimensions, FPS, and frame count.

    Args:
        video_path: Validated video path.

    Returns:
        Video metadata.

    Raises:
        VideoIOError: If metadata cannot be read.
    """
    path = validate_video_path(video_path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoIOError(f"Unable to open video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise VideoIOError("Video has invalid dimensions")
        if fps <= 0:
            fps = float(config.OUTPUT_FPS)
        return VideoMetadata(width, height, fps, max(frame_count, 0))
    finally:
        capture.release()


def iter_video_frames(video_path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, frame)`` pairs and always release the capture.

    Args:
        video_path: Validated video path.

    Yields:
        BGR frames indexed from zero.

    Raises:
        VideoIOError: If a frame cannot be read after the stream starts.
    """
    path = validate_video_path(video_path)
    capture = cv2.VideoCapture(str(path))
    frame_index = 0
    try:
        if not capture.isOpened():
            raise VideoIOError(f"Unable to open video: {path}")
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None or frame.size == 0:
                raise VideoIOError(f"Empty frame at index {frame_index}")
            yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def create_video_writer(output_path: Path) -> cv2.VideoWriter:
    """Create an MP4 writer using the configured output dimensions and FPS.

    Args:
        output_path: Destination video path.

    Returns:
        An opened OpenCV video writer.

    Raises:
        VideoIOError: If the writer cannot be opened.
    """
    destination = Path(output_path)
    ensure_directory(destination.parent)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*config.VIDEO_CODEC),
        float(config.OUTPUT_FPS),
        (config.FRAME_WIDTH, config.FRAME_HEIGHT),
    )
    if not writer.isOpened():
        writer.release()
        raise VideoIOError(f"Unable to create output video: {destination}")
    return writer
