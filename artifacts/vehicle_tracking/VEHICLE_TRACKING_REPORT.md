# Vehicle Detection and Tracking Experiment

## Protocol

- Input: `C:/Users/vlctus/OneDrive/Desktop/Pattern/project development/dataset/PIE_clips/set01/video_0001.mp4`
- Evaluation interval: frames 0 through 1799 (60.00 seconds at 30.000 FPS).
- Detector: pretrained `C:/Users/vlctus/OneDrive/Desktop/Pattern/project development/yolo11s.pt`; no model training.
- Tracker: BoT-SORT using the separate `C:/Users/vlctus/OneDrive/Desktop/Pattern/project development/smartped_vehicle_botsort.yaml`.
- Detections and tracker IDs are model-generated predictions. PIE XML boxes and IDs are annotations.
- Trajectory displacement and speed are image-space estimates only.

## Predicted detections and tracks

- Vehicle detections: 13226
- Unique predicted vehicle tracks: 429
- Detections without tracker IDs: 0

| Predicted class | Detections |
|---|---:|
| bicycle | 204 |
| bus | 21 |
| car | 12527 |
| motorcycle | 3 |
| truck | 471 |

## PIE annotations in the same interval

- Vehicle annotation boxes (including outside states): 1312
- Visible boxes with valid geometry: 1300
- Unique annotated vehicle tracks: 10
- Visible boxes marked occluded: 48
- Boxes whose PIE types map to requested COCO classes: 1300
- Unmapped/out-of-scope boxes: 0
- Tracker observation gaps: 881; missing frame slots: 8734; gaps exceeding 15 frames: 92

| PIE type | Annotation boxes | Unique tracks | Mapped COCO class |
|---|---:|---:|---|
| car | 1312 | 10 | car |

## Frame-level annotation matching

- Method: greedy one-to-one maximum-IoU assignment per frame and mapped class.
- Eligible mapped annotation boxes: 1300
- Matched annotation boxes at IoU >= 0.50: 1233
- Annotation match rate (matched / eligible mapped annotation boxes): 0.9485
- Mean IoU over matched pairs: 0.8726
- This is annotation recall/coverage, not detection precision: PIE labels are not exhaustive, so unmatched predictions are not treated as false positives.

## Limitations

- PIE `bike` is left unmapped; it is not automatically equated with COCO bicycle or motorcycle. PIE `train` is counted but outside the requested COCO class set.
- Occluded boxes are included in matching when valid geometry is present; outside boxes are excluded from matching.
- Moving-camera motion, parallax, occlusion, detector misses, and tracker ID switches affect frame matching and trajectories. Frame-level IoU does not validate track identity.
- Tracker IDs are estimates, not guaranteed persistent ground truth. Gaps are retained explicitly; no missing observations are interpolated.
- Pixel displacement/image speed is not real-world distance or velocity. No collision, near-miss, TTC, or risk labels are created.
- YOLO detections may include visible objects absent from PIE annotations; these are not scored as false positives.
