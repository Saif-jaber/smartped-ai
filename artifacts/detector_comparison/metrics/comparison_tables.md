### AP and max-F1 operating point (test set04, n=10,593 images, 38,047 GT boxes)

| Metric | Pretrained YOLO11s | Fine-tuned YOLO11s | Delta (fine - pre) | Relative |
|---|---:|---:|---:|---:|
| Precision (max-F1) | 0.3115 | 0.3861 | +0.0746 | +23.9% |
| Recall (max-F1) | 0.4037 | 0.3690 | -0.0347 | -8.6% |
| mAP@0.50 | 0.2236 | 0.2884 | +0.0648 | +29.0% |
| mAP@0.50:0.95 | 0.1200 | 0.1629 | +0.0429 | +35.8% |

### Fixed operating point: conf=0.25, IoU=0.50 greedy matching (matches pipeline default)

| Metric | Pretrained YOLO11s | Fine-tuned YOLO11s | Delta (fine - pre) | Relative |
|---|---:|---:|---:|---:|
| True positives | 19,179 | 10,567 | -8,612 | -44.9% |
| False positives | 63,308 | 11,867 | -51,441 | -81.3% |
| False negatives | 18,868 | 27,480 | +8,612 | +45.6% |
| Precision | 0.2325 | 0.4710 | +0.2385 | +102.6% |
| Recall | 0.5041 | 0.2777 | -0.2264 | -44.9% |
| F1 | 0.3182 | 0.3494 | +0.0312 | +9.8% |

### Prediction volume and per-image behaviour

| Metric | Pretrained YOLO11s | Fine-tuned YOLO11s | Delta (fine - pre) | Relative |
|---|---:|---:|---:|---:|
| Boxes @ conf 0.001 | 1,328,794 | 966,295 | -362,499 | -27.3% |
| Boxes @ conf 0.250 | 82,487 | 22,434 | -60,053 | -72.8% |
| Boxes @ conf 0.500 | 44,473 | 9,243 | -35,230 | -79.2% |
| Max boxes in one image | 303 | 226 | -77 | -25.4% |
| Per-image @0.25 median | 8.0 | 2.0 | -6.0000 | -75.0% |
| Per-image @0.25 mean | 7.79 | 2.12 | -5.6691 | -72.8% |
| Per-image @0.25 p95 | 16.0 | 6.0 | -10.0000 | -62.5% |
| Per-image @0.25 max | 24 | 17 | -7 | -29.2% |

### Speed (as measured on this machine, RTX 4070 Laptop GPU)

| Metric | Pretrained YOLO11s | Fine-tuned YOLO11s | Delta (fine - pre) | Relative |
|---|---:|---:|---:|---:|
| Preprocess (ms/image) | 2.33 | 2.77 | +0.4381 | +18.8% |
| Inference (ms/image) | 115.79 | 114.35 | -1.4374 | -1.2% |
| Postprocess (ms/image) | 10.80 | 10.26 | -0.5435 | -5.0% |
| FPS inference-only | 8.64 | 8.74 | +0.1086 | +1.3% |
| FPS end-to-end | 7.76 | 7.85 | +0.0939 | +1.2% |

### Checkpoint / model complexity

| Property | Pretrained YOLO11s | Fine-tuned YOLO11s |
|---|---:|---:|
| File size (bytes) | 19,313,732 | 19,149,914 |
| SHA-256 | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` | `9e1d876ff9ccc49d03f43da9dbe5bb4df71c2a7514dbaad24585e4a1f855ecfb` |
| Parameters | 9,458,752 | 9,428,179 |
| GFLOPs @ 640 | 21.84 | 21.67 |
| Classes | 80 (class 0 = person) | 1 (class 0 = pedestrian) |

### max_det sensitivity (300 vs 1000)

| Model | max_det | Boxes @0.001 | Max boxes/image | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Pretrained | 300 | 1,328,791 | 300 | 0.223617 | 0.12001062 |
| Pretrained | 1000 | 1,328,794 | 303 | 0.223617 | 0.12001064 |
| Fine-tuned | 300 | 966,295 | 226 | 0.288428 | 0.16294832 |
| Fine-tuned | 1000 | 966,295 | 226 | 0.288428 | 0.16294832 |
