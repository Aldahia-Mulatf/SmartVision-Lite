# SmartVision Lite — Academic Technical Report

## 1. Problem statement

SmartVision Lite monitors cardboard shipping boxes moving on a conveyor belt
using a recorded video. The system detects only the product class
`cardboard_box`, ignores workers, follows each product with a stable ID,
checks geometric and visual quality, counts line crossings once, and exports a
fully annotated video for academic demonstration.

The design intentionally uses a small modular pipeline rather than a database,
service mesh, authentication layer, or real-time streaming architecture.

## 2. Processing architecture

```text
Recorded MP4
   -> Preprocessor
   -> Detector
   -> Tracker
   -> OrientationEstimator
   -> QualityChecker
   -> Counter
   -> Visualizer
   -> Annotated MP4 + Streamlit metrics
```

Each stage receives the previous stage's output. The original input frame is
never modified; overlays are drawn on a copied frame.

## 3. Stage design

### 3.1 Preprocessing

`pipeline/preprocessor.py` resizes every frame to 1280 x 720, applies Gaussian
smoothing, enhances the L channel with CLAHE, and computes Laplacian variance.
Frames below `BLUR_THRESHOLD` are marked as blurry. This score controls the
quality stage only; detection and counting continue.

### 3.2 Detection

`pipeline/detector.py` loads the custom YOLO model once and filters results
before they leave the stage. In `custom` mode only `cardboard_box` (or class
zero in a verified one-class model) is accepted. `person` and every other
class are rejected. `agnostic` mode is retained only for early experiments.

### 3.3 Tracking

`pipeline/tracker.py` implements a dependency-free ByteTrack-style high/low
confidence association. High-confidence detections create and match tracks;
lower-confidence detections can recover an existing track. Track IDs are
monotonic, trajectories are retained, and a sudden area reduction of 50% or
more marks the current observation as occluded.

This lightweight implementation keeps the seven-stage boundary explicit and
avoids adding a second tracking dependency. It is suitable for this academic
pipeline; a production system would use a maintained Kalman/Hungarian
ByteTrack implementation.

### 3.4 Orientation

`pipeline/orientation.py` thresholds the cardboard HSV range, selects the
largest contour, and applies `cv2.minAreaRect`. The rectangle angle is
normalized to `[0, 90)`. Deviation from `CONVEYOR_AXIS_ANGLE` is compared with
`ORIENTATION_TOLERANCE_DEG`.

### 3.5 Quality

`pipeline/quality_checker.py` accumulates evidence at track level:

- `crushed`: solidity below `SOLIDITY_MIN` or an aspect ratio outside the
  configured range.
- `missing_label`: label ratio below `MIN_LABEL_RATIO` in at least 70% of
  eligible clear frames.
- `size_anomaly`: representative area differs by more than 40% from the
  median of the latest completed tracks, after five warm-up samples.
- `misaligned`: orientation deviation exceeds the conveyor tolerance.

Blurred and occluded observations are excluded from quality evidence. The
sharpest clear observation supplies the stable contour and label measurement.
The label check is skipped for boxes below `MIN_AREA_FOR_LABEL_CHECK`.

### 3.6 Counting

`pipeline/counter.py` stores the previous center for each track and detects a
crossing of the horizontal counting line. A set of track IDs prevents double
counting. Quality buckets can be corrected after a track is finalized.

### 3.7 Visualization

`pipeline/visualizer.py` renders the box, `BOX #id`, quality reason, angle
arrow, trajectory, counting line, and total/good/defective/FPS panel. Green
means provisional or final OK; red means defective.

## 4. Runtime behavior

The Streamlit app follows the required sequence:

1. Upload a video or use `assets/sample_video.mp4`.
2. Press **Run monitoring**.
3. Validate the file and load the cached detector.
4. Process the complete video with a progress bar.
5. Display the generated `outputs/monitored_<timestamp>.mp4`.
6. Show four metrics and provide a download button.

The app never performs processing merely because the page reran. Detector
loading uses `st.cache_resource`.

## 5. Dataset and training

The detector uses a one-class YOLOv8 dataset. See
[`dataset/README.md`](../dataset/README.md). The repository includes a small
custom demo model and a clean synthetic conveyor video for an immediately
runnable demonstration. Those assets are not a substitute for real evaluation
on the project's target camera.

For real training, use:

```bash
yolo task=detect mode=train model=yolov8n.pt \
    data=dataset/data.yaml epochs=30 imgsz=640
```

Copy the resulting weights to `models/box_detector.pt` and validate them on
conveyor frames containing lighting variation, blur, partial occlusion, empty
belt intervals, and worker distractors.

## 6. Testing strategy

The test suite covers:

- preprocessing without input mutation,
- strict detector class filtering,
- stable tracking IDs and occlusion flags,
- orientation normalization,
- blur skipping and missing-label evidence,
- exactly-once counting,
- non-mutating visualization,
- video validation, and
- a two-frame end-to-end pipeline run with a deterministic detector double.

## 7. Limitations and academic interpretation

The system uses axis-aligned YOLO boxes rather than segmentation masks. The
quality checks therefore estimate geometry from the visible contour inside the
box. Lighting and camera calibration affect HSV thresholds and orientation.
The conveyor angle and counting line must be calibrated for each video.

The included code is a correct, testable academic baseline. A trained model
and representative conveyor video are still required before claiming final
accuracy or presenting quantitative detection metrics.
