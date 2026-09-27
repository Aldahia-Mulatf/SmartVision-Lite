# SmartVision Lite — Carton Box Monitoring

SmartVision Lite is an academic computer-vision pipeline for monitoring
cardboard shipping boxes moving on a conveyor belt. The system uses classical
image processing, a custom YOLOv8 detector, tracking, orientation estimation,
quality checks, exactly-once counting, and a Streamlit dashboard.

## Project status

The seven-stage pipeline, Streamlit runtime, tests, central configuration,
academic documentation, and runnable demo assets are implemented. The
repository now includes:

```text
models/box_detector.pt
assets/sample_video.mp4
```

The included demo weights are a custom one-class detector trained on the
bundled synthetic cardboard-box exercise data so the application can run
immediately. For a final academic accuracy claim, replace them with weights
trained on real conveyor frames. People are never counted or quality-checked.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset and model

See [`dataset/README.md`](dataset/README.md) for the required YOLOv8 layout,
annotation rules, and training command. After training, place the weights at:

```text
models/box_detector.pt
```

The final detector must contain only the `cardboard_box` class. The temporary
`agnostic` mode is available in the detector API for early tests but is not
acceptable for the final presentation.

## Run the dashboard

Place a conveyor video at `assets/sample_video.mp4`, or upload one from the
sidebar, then run:

```bash
streamlit run app.py
```

The app processes the complete video, writes
`outputs/monitored_<timestamp>.mp4`, displays it with `st.video`, shows four
metrics, and provides a download button.

## Run tests

```bash
pytest -q
```

## Repository map

```text
app.py                      Streamlit entry point
config.py                   Central constants and calibration
pipeline/preprocessor.py   Resize, CLAHE, blur score
pipeline/detector.py       YOLO detection and class filtering
pipeline/tracker.py        Track association and occlusion
pipeline/orientation.py    minAreaRect orientation
pipeline/quality_checker.py Track-level quality rules
pipeline/counter.py        Counting-line logic
pipeline/visualizer.py     OpenCV annotations
pipeline/runner.py         Complete-video orchestration
utils/                     Video I/O, logging, common helpers
tests/                     Unit and integration tests
docs/                      Report and presentation outline
```
