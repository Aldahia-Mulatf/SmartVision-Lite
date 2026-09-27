# SmartVision Lite — Presentation Outline

## Slide 1 — Title

**SmartVision Lite**

Cardboard Box Monitoring on a Conveyor Belt

- Computer Vision and Image Processing project
- Classical processing + custom YOLOv8
- Streamlit output dashboard

## Slide 2 — Motivation

- Manual counting is slow and inconsistent.
- Workers and conveyor hardware create visual distractors.
- A shipping box must be detected, tracked, checked, and counted once.
- The project focuses on explainable processing stages.

## Slide 3 — Product policy

- Target class: `cardboard_box`
- Ignored class: `person`
- No OCR: only label presence is checked.
- No database and no live camera dependency.
- Final demonstration uses a pre-recorded conveyor video.

## Slide 4 — Seven-stage pipeline

```text
Preprocessing -> Detection -> Tracking -> Orientation
      -> Quality -> Counting -> Visualization
```

Each stage has a separate module, typed dataclasses, logs, and tests.

## Slide 5 — Classical preprocessing

- Resize to a common working resolution.
- Gaussian smoothing reduces sensor noise.
- CLAHE compensates for uneven warehouse lighting.
- Laplacian variance measures motion blur.
- Blurry frames continue detection and counting but do not create quality
  evidence.

## Slide 6 — Custom detection

- COCO does not provide the required cardboard-box class.
- A one-class YOLOv8 model is trained using `cardboard_box` annotations.
- The detector filters workers before later stages.
- Model path: `models/box_detector.pt`.

## Slide 7 — Tracking and occlusion

- High-confidence detections establish tracks.
- Lower-confidence detections can recover a track.
- Every box receives one stable ID.
- A 50% area drop is marked `occluded`.
- Occluded observations do not generate quality defects.

## Slide 8 — Orientation

- Convert the box crop to an HSV cardboard mask.
- Extract the largest contour.
- Apply `cv2.minAreaRect`.
- Normalize the rectangle angle to `[0, 90)`.
- Compare with the calibrated conveyor axis.

## Slide 9 — Quality rules

| Rule | Decision |
|---|---|
| Solidity | `solidity < 0.85` → crushed |
| Label | Less than 2% in at least 70% of eligible frames → missing label |
| Size | More than 40% from median after five samples → size anomaly |
| Orientation | More than 15° from conveyor axis → misaligned |

Quality is track-level, not a single-frame decision.

## Slide 10 — Counting

- A horizontal line is calibrated at `COUNTING_LINE_Y`.
- The track center crossing the line increments the count.
- A `set[int]` prevents duplicate counting.
- Good and defective buckets are updated from final track quality.

## Slide 11 — Visual output

- Green bounding box: OK.
- Red bounding box: defective with reason.
- `BOX #id` remains stable during movement.
- Angle arrow and trajectory explain the decision.
- Dashboard shows total, good, defective, and FPS.

## Slide 12 — Runtime demonstration

1. Select or upload the MP4.
2. Press **Run monitoring**.
3. Watch the frame progress bar.
4. Play the generated annotated video.
5. Inspect metrics and download the result.

## Slide 13 — Verification

- Unit tests cover all core stages.
- Integration test processes a synthetic two-frame video.
- Frames are copied before drawing.
- People are filtered before tracking.
- Empty and blurry frames are handled without crashing.

## Slide 14 — Limitations and future academic experiments

- Accuracy depends on annotation quality and camera calibration.
- HSV thresholds may need recalibration for a different warehouse.
- Axis-aligned detections limit fine-grained deformation analysis.
- Future experiments can compare confidence thresholds, CLAHE settings, and
  quality rules using the same recorded video.

## Slide 15 — Conclusion

SmartVision Lite combines classical image processing with a custom detector in
a transparent seven-stage pipeline. The result is explainable, testable, and
appropriate for an academic computer-vision demonstration.
