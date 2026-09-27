# SmartVision Lite — Carton Box Monitoring

SmartVision Lite is an academic computer-vision pipeline for monitoring
cardboard shipping boxes moving on a conveyor belt. The final system will use
classical image processing, a custom YOLOv8 detector, tracking, orientation
estimation, quality checks, counting, and a Streamlit dashboard.

## Current status

The repository scaffold and central configuration are ready. The final custom
model is **not** included yet because it must be trained on a real, single-class
`cardboard_box` dataset.

## First setup

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

The final detector must contain only the `cardboard_box` class. People are not
an analysis class and must never be counted or quality-checked.

## Planned runtime

The application entry point will be `app.py` and will process a complete
pre-recorded video before presenting the annotated output through Streamlit.
It will not use a live frame-by-frame `st.image` loop.
