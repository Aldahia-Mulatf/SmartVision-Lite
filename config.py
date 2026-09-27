"""Central configuration for SmartVision Lite.

All pipeline-wide constants live in this module so that calibration and
academic experiments can be changed without editing processing stages.
"""

from pathlib import Path


# Product configuration.
PRODUCT_NAME = "cardboard_box"
PRODUCT_MODE = "custom"
CUSTOM_MODEL_PATH = Path("models/box_detector.pt")
YOLO_MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45
IGNORED_CLASSES = ("person",)

# Conveyor configuration.
CONVEYOR_AXIS_ANGLE = 0.0
ORIENTATION_TOLERANCE_DEG = 15.0
COUNTING_LINE_Y = 400

# HSV ranges: OpenCV uses H in [0, 180] and S/V in [0, 255].
CARDBOARD_HSV_LOWER = (10, 40, 40)
CARDBOARD_HSV_UPPER = (30, 255, 255)
LABEL_HSV_LOWER = (0, 0, 180)
LABEL_HSV_UPPER = (180, 60, 255)

# Quality configuration.
SOLIDITY_MIN = 0.85
MIN_LABEL_RATIO = 0.02
LABEL_DEFECT_FRAME_RATIO = 0.70
MIN_AREA_FOR_LABEL_CHECK = 2000
SIZE_DEVIATION_MAX = 0.40
MEDIAN_WINDOW_SIZE = 20
MIN_SAMPLES_FOR_SIZE_CHECK = 5
BOX_ASPECT_RANGE = (0.4, 2.5)
BLUR_THRESHOLD = 100.0
OCCLUSION_AREA_DROP_RATIO = 0.50

# Video and UI configuration.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
OUTPUT_FPS = 30
MAX_UPLOAD_SIZE_MB = 500
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# Repository paths.
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOG_FILE_PATH = OUTPUTS_DIR / "processing.log"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "dataset"
ASSETS_DIR = PROJECT_ROOT / "assets"

# OpenCV BGR colors used by the visualizer.
COLORS = {
    "good": (0, 255, 0),
    "defective": (0, 0, 255),
    "tracking": (255, 165, 0),
    "counting_line": (0, 255, 255),
    "orientation": (255, 0, 0),
    "text": (255, 255, 255),
}
