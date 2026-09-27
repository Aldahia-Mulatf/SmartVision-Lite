# Dataset preparation

The final detector must be trained on one class only:

```text
cardboard_box
```

## Expected YOLOv8 layout

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each label file uses the YOLO format:

```text
class_id center_x center_y width height
```

All values except `class_id` are normalized to `[0, 1]`. The only valid
`class_id` in this project is `0` (`cardboard_box`). Do not include `person`
or any other class in the training labels.

## Recommended preparation process

1. Obtain a dataset containing cardboard shipping boxes, or extract 60–100
   representative frames from the project conveyor video.
2. Annotate every visible cardboard box with one rectangular bounding box.
3. Include variations in lighting, partial occlusion, blur, box size, and
   empty conveyor frames.
4. Split the data into training and validation sets.
5. Validate that every annotation uses class `0` only.
6. Train the detector from the repository root:

```bash
yolo task=detect mode=train model=yolov8n.pt \
    data=dataset/data.yaml epochs=30 imgsz=640
```

7. Copy the resulting weights to:

```text
models/box_detector.pt
```

The model file is intentionally not created by the scaffold. A custom model
must be produced from real annotations before the final demonstration.
