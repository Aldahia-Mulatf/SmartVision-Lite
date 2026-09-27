"""Unit tests for video input validation."""

from pathlib import Path

import pytest

from utils.video_io import VideoValidationError, validate_video_path


def test_missing_video_is_rejected(tmp_path: Path) -> None:
    """Verify missing inputs produce a user-facing validation error."""
    with pytest.raises(VideoValidationError):
        validate_video_path(tmp_path / "missing.mp4")


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    """Verify non-video extensions fail before OpenCV processing."""
    path = tmp_path / "sample.txt"
    path.write_text("not a video", encoding="utf-8")
    with pytest.raises(VideoValidationError):
        validate_video_path(path)
