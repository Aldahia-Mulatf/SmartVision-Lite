"""Small end-to-end integration test using a deterministic fake detector."""

from pathlib import Path

import cv2
import numpy as np

import config
from pipeline.detector import DetectionResult
from pipeline.runner import ProcessingSummary, process_video


class MovingFakeDetector:
    """Return one known detection whose center crosses the counting line."""

    def __init__(self) -> None:
        """Initialize the frame-call counter."""
        self._calls = 0

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Return overlapping boxes on two synthetic frames."""
        self._calls += 1
        y1 = 250 if self._calls == 1 else 350
        return [DetectionResult((200, y1, 400, y1 + 200), 0.95, 0, "cardboard_box")]


def _write_two_frame_video(path: Path) -> None:
    """Write a minimal MP4 accepted by OpenCV."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*config.VIDEO_CODEC),
        float(config.OUTPUT_FPS),
        (config.FRAME_WIDTH, config.FRAME_HEIGHT),
    )
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(
                np.zeros(
                    (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3),
                    dtype=np.uint8,
                )
            )
    finally:
        writer.release()


def test_pipeline_processes_video_and_writes_output(tmp_path: Path) -> None:
    """Verify all seven stages can run with a deterministic detector double."""
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "monitored.mp4"
    _write_two_frame_video(input_path)
    summary = process_video(
        input_path,
        output_path,
        MovingFakeDetector(),  # type: ignore[arg-type]
    )
    assert isinstance(summary, ProcessingSummary)
    assert summary.frames_processed == 2
    assert summary.total_count == 1
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
