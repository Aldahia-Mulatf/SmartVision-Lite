"""Streamlit entry point for SmartVision Lite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

import config
from pipeline.detector import Detector, ModelLoadError
from pipeline.runner import ProcessingError, ProcessingSummary, process_video
from utils.helpers import ensure_directory, timestamped_output_path
from utils.logger import get_logger
from utils.video_io import VideoValidationError, validate_video_path


LOGGER = get_logger(__name__)


@st.cache_resource(show_spinner="Loading cardboard-box detector...")
def load_cached_detector() -> Detector:
    """Load the YOLO detector once per Streamlit process."""
    return Detector.from_config()


def _validate_uploaded_file(uploaded_file: Any) -> None:
    """Validate an uploaded Streamlit file before writing it to disk.

    Args:
        uploaded_file: Streamlit UploadedFile-like object.

    Raises:
        VideoValidationError: If extension or byte size is invalid.
    """
    file_name = str(getattr(uploaded_file, "name", ""))
    extension = Path(file_name).suffix.lower()
    if extension not in config.SUPPORTED_VIDEO_EXTENSIONS:
        extensions = ", ".join(config.SUPPORTED_VIDEO_EXTENSIONS)
        raise VideoValidationError(f"Unsupported video extension. Use: {extensions}")
    file_size = int(getattr(uploaded_file, "size", 0) or 0)
    maximum = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size <= 0:
        raise VideoValidationError("The uploaded video is empty")
    if file_size > maximum:
        raise VideoValidationError(
            f"Video exceeds the {config.MAX_UPLOAD_SIZE_MB} MB size limit"
        )


def _save_uploaded_file(uploaded_file: Any) -> Path:
    """Save an already validated upload to the ignored outputs directory."""
    _validate_uploaded_file(uploaded_file)
    extension = Path(str(uploaded_file.name)).suffix.lower()
    destination = timestamped_output_path(
        config.OUTPUTS_DIR, "uploaded", extension
    )
    ensure_directory(destination.parent)
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def _persist_uploaded_file(uploaded_file: Any) -> Path:
    """Persist an upload across Streamlit reruns and return its validated path."""
    _validate_uploaded_file(uploaded_file)
    signature = (
        str(getattr(uploaded_file, "name", "")),
        int(getattr(uploaded_file, "size", 0) or 0),
    )
    stored_signature = st.session_state.get("uploaded_signature")
    stored_path = st.session_state.get("uploaded_video_path")
    if stored_signature == signature and stored_path:
        candidate = Path(str(stored_path))
        if candidate.is_file():
            return validate_video_path(candidate)
    destination = _save_uploaded_file(uploaded_file)
    st.session_state["uploaded_signature"] = signature
    st.session_state["uploaded_video_path"] = str(destination)
    return validate_video_path(destination)


def _render_summary(summary: ProcessingSummary) -> None:
    """Render the output video, four metrics, and download control."""
    st.subheader("Processed monitoring video")
    st.video(str(summary.output_path))
    columns = st.columns(4)
    columns[0].metric("Total", summary.total_count)
    columns[1].metric("Good", summary.good_count)
    columns[2].metric("Defective", summary.defective_count)
    columns[3].metric("FPS", f"{summary.average_fps:.1f}")
    video_bytes = summary.output_path.read_bytes()
    st.download_button(
        label="Download monitored video",
        data=video_bytes,
        file_name=summary.output_path.name,
        mime="video/mp4",
    )


def _select_input_video(uploaded_file: Any) -> Path | None:
    """Resolve an upload, persisted upload, or default sample video."""
    if uploaded_file is not None:
        return _persist_uploaded_file(uploaded_file)
    stored_path = st.session_state.get("uploaded_video_path")
    if stored_path:
        candidate = Path(str(stored_path))
        if candidate.is_file():
            return validate_video_path(candidate)
    default_path = config.ASSETS_DIR / "sample_video.mp4"
    if default_path.is_file():
        return validate_video_path(default_path)
    return None


def main() -> None:
    """Render the Streamlit dashboard and process only on button press."""
    st.set_page_config(page_title="SmartVision Lite", layout="wide")
    st.title("SmartVision Lite — Carton Box Monitoring")
    st.caption("Classical image processing + custom YOLOv8 + tracking")
    with st.sidebar:
        st.header("Video source")
        uploaded_file = st.file_uploader(
            "Upload a conveyor-belt video",
            type=[
                extension.lstrip(".")
                for extension in config.SUPPORTED_VIDEO_EXTENSIONS
            ],
            key="video_upload",
        )
        if uploaded_file is not None:
            try:
                selected_path = _persist_uploaded_file(uploaded_file)
                st.success(f"Selected: {selected_path.name}")
            except VideoValidationError as error:
                LOGGER.error("Upload validation error: %s", error)
                st.error(str(error))
        elif st.session_state.get("uploaded_video_path"):
            st.success("Uploaded video is ready. Press Run monitoring.")
        else:
            st.info("No upload selected. The app will use assets/sample_video.mp4.")
        run_monitoring = st.button("Run monitoring", type="primary")

    if run_monitoring:
        try:
            input_path = _select_input_video(uploaded_file)
            if input_path is None:
                raise VideoValidationError(
                    "No video selected and assets/sample_video.mp4 is missing"
                )
            output_path = timestamped_output_path(
                config.OUTPUTS_DIR,
                config.OUTPUT_FILENAME_PREFIX,
                ".mp4",
            )
            progress = st.progress(0.0, text="Preparing video...")

            def update_progress(current: int, total: int) -> None:
                """Update the progress bar after a completed frame."""
                fraction = current / total if total > 0 else 0.0
                progress.progress(
                    min(max(fraction, 0.0), 1.0),
                    text=f"Processing frame {current} of {total or '?'}",
                )

            detector = load_cached_detector()
            summary = process_video(
                input_path,
                output_path,
                detector,
                progress_callback=update_progress,
            )
            progress.progress(1.0, text="Processing complete")
            st.session_state["last_summary"] = summary
        except (VideoValidationError, ModelLoadError) as error:
            LOGGER.error("User or model validation error: %s", error)
            st.error(str(error))
        except ProcessingError as error:
            LOGGER.exception("Processing error")
            st.error(f"Processing failed: {error}")

    summary = st.session_state.get("last_summary")
    if isinstance(summary, ProcessingSummary):
        _render_summary(summary)


if __name__ == "__main__":
    main()
