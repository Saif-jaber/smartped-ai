# Pedestrian Detector Comparison: Pretrained vs PIE Fine-Tuned YOLO11s

**Date:** 2026-11-05  
**Scope:** Read-only evaluation on the held-out PIE test split (`set04`)  
**Status:** Complete. No training, fine-tuning, weight modification, or test-set tuning was performed.

---

## Executive summary

Two YOLO11s checkpoints were evaluated under an identical protocol on the PIE pedestrian test split:

- **Pretrained:** `yolo11s.pt` (COCO, 80 classes; class 0 = `person`)
- **Fine-tuned:** `runs/detect/train-5/weights/best.pt` (PIE, 1 class; class 0 = `pedestrian`)

Both models were restricted to class 0 predictions and evaluated on the same 10,593 images and 38,047 annotated pedestrian boxes from `set04`.

**Main finding:** Fine-tuning improves overall ranking/localization quality and precision, but at the pipeline's current confidence threshold (`0.25`) it trades a large amount of recall for a large reduction in false positives.

| Result | Pretrained | Fine-tuned | Fine-tuned change |
|---|---:|---:|---:|
| mAP@0.50 | 0.2236 | **0.2884** | +29.0% |
| mAP@0.50:0.95 | 0.1200 | **0.1629** | +35.8% |
| Precision (max-F1 point) | 0.3115 | **0.3861** | +23.9% |
| Recall (max-F1 point) | **0.4037** | 0.3690 | -8.6% |
| Precision @ conf 0.25 | 0.2325 | **0.4710** | +102.6% |
| Recall @ conf 0.25 | **0.5041** | 0.2777 | -44.9% |
| F1 @ conf 0.25 | 0.3182 | **0.3494** | +9.8% |
| False positives @ conf 0.25 | 63,308 | **11,867** | -81.3% |
| False negatives @ conf 0.25 | **18,868** | 27,480 | +45.6% |
| Inference time (ms/image) | 115.79 | **114.35** | -1.2% |
| End-to-end FPS | 7.76 | **7.85** | +1.2% |

**Bottom line:** The fine-tuned checkpoint is the better detector by AP and by precision at a common threshold, and it emits 72.8% fewer boxes at `conf=0.25`. However, it also misses substantially more annotated pedestrians at that threshold. The decision is therefore a **precision-recall policy choice**, not a simple accuracy win. Before deployment, calibrate the fine-tuned model's confidence threshold on the validation split (`set06`) to the pipeline's recall requirement, then re-check precision and downstream crossing-intention performance. Do not tune on the test split (`set04`).

---

## 1. Objective

Compare the pretrained YOLO11s checkpoint and the PIE fine-tuned YOLO11s checkpoint for pedestrian detection in the SmartPed AI pipeline, using the same data, class mapping, and evaluation procedure, and produce a reproducible recommendation for detector selection.

The evaluation was designed to answer:

1. Which checkpoint has better overall detection quality (AP)?
2. Which checkpoint is better at a fixed, operational confidence threshold?
3. How does the precision-recall trade-off affect the pipeline's default `conf=0.25`?
4. Is there a meaningful inference-speed difference?
5. What should the pipeline use next, given the PIE annotation limitations?

---

## 2. Checkpoints and environment

### 2.1 Checkpoints

| Property | Pretrained YOLO11s | Fine-tuned YOLO11s |
|---|---|---|
| Path | `yolo11s.pt` | `runs/detect/train-5/weights/best.pt` |
| Size | 19,313,732 bytes | 19,149,914 bytes |
| SHA-256 | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` | `9e1d876ff9ccc49d03f43da9dbe5bb4df71c2a7514dbaad24585e4a1f855ecfb` |
| Classes | 80 | 1 |
| Class 0 name | `person` | `pedestrian` |
| Parameters | 9,458,752 | 9,428,179 |
| GFLOPs @ 640 | 21.84 | 21.67 |

The fine-tuned checkpoint is the `best.pt` from the prior `train-5` run. It was selected on the validation split (`set06`) during training; its validation metrics were P=0.35896, R=0.32509, mAP50=0.25915, mAP50-95=0.13807 at epoch 11. This report evaluates it once on the held-out test split (`set04`). The validation metrics are provenance only and are not directly comparable to the test metrics below.

### 2.2 Environment

| Item | Value |
|---|---|
| Python | 3.12.3 |
| Ultralytics | 8.4.198 |
| PyTorch | 2.14.1+cu126 |
| CUDA | 12.6 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| OS | Windows 11 |
| OpenCV | 5.0.0 |

---

## 3. Data and evaluation protocol

### 3.1 Dataset and split

PIE detection export under `artifacts/pie_detection_yolo/`:

| Split | Recording sets | Images | Annotated boxes |
|---|---|---:|---:|
| Train | `set01`, `set02`, `set03`, `set05` | - | - |
| Validation | `set06` | - | - |
| Test | `set04` | 10,593 | 38,047 |

Test split details:

- 16 videos (`video_0001` through `video_0016`).
- 10,593 images and 10,593 label files; every image has a label file.
- All 38,047 boxes are class 0 (`pedestrian`).
- Boxes per image: min 1, median 3, mean 3.59, p95 9, max 11.
- Image size: 1920 x 1080 (RGB).
- Normalized box area: min 2.89e-05, median 0.00212, mean 0.00604, max 0.09827.

No test images or labels were used for training or threshold selection. `set04` is the held-out test set.

### 3.2 Class mapping

The two checkpoints use different class spaces:

- Pretrained: 80 COCO classes; index 0 is `person`.
- Fine-tuned: 1 class; index 0 is `pedestrian`.

For a valid comparison, both models were restricted to **class index 0** during validation and NMS. This means the pretrained model could only emit `person` detections, and the fine-tuned model could only emit `pedestrian` detections. Other COCO classes were excluded.

### 3.3 Evaluation procedure

A custom `PedestrianOnlyValidator(DetectionValidator)` was used. It restricts NMS to class 0 and collapses the metric names to the single pedestrian class. AP, precision, and recall are computed by the unmodified Ultralytics `DetMetrics` machinery.

Primary evaluation settings:

| Setting | Value |
|---|---|
| Split | `test` (`set04` only) |
| Image size | 640 |
| Batch | 16 |
| Device | GPU 0 |
| Workers | 4 |
| Confidence for AP | 0.001 |
| NMS IoU | 0.70 |
| `max_det` | 1000 (uncapped primary run) |
| `rect` | True |
| Augmentation | False |
| `agnostic_nms` | False |
| `single_cls` | False |
| Seed | 0 |
| Deterministic | True |
| Save JSON / TXT | False |

The standard Ultralytics `max_det` default is 300. The pretrained model hit that cap on at least one image (303 candidates at `conf=0.001`), while the fine-tuned model did not (max 226). To keep the comparison symmetric, the primary run used `max_det=1000` for both models. A sensitivity run at `max_det=300` is included in Section 4.5; the difference is negligible (three low-confidence boxes, mAP50-95 change of 1.7e-08).

### 3.4 Two precision/recall views

Two different operating-point views are reported because they answer different questions:

1. **Ultralytics max-F1 point:** precision and recall at the confidence that maximizes the mean F1 score on the PR curve. Each model gets its own optimal threshold. This is useful for comparing the shape of the PR curve, but it is not a common deployment threshold.
2. **Fixed `conf=0.25`, IoU=0.50 greedy matching:** both models are evaluated at the same confidence, which matches the pipeline's default (`smartped_ai/video_runner/pipeline.py:1052`, `smartped_ai/trajectory/readers.py:535-536`). This is the operational comparison.

The fixed-threshold matching uses greedy IoU matching at IoU 0.50. It is not the same as AP, which integrates over IoU thresholds 0.50:0.05:0.95.

### 3.5 Dataset mirror and integrity

Ultralytics writes `*.cache` files next to label files. To avoid modifying the original dataset, evaluation used a temporary mirror:

- Images: directory junction to the original images (read-only use).
- Labels: copied to `%TEMP%\opencode\smartped_detector_eval\pie_detection_yolo`.
- A mirror `data.yaml` points to the mirror.

The original dataset, annotations, model weights, and existing pipeline modules were not modified. A test-split cache file created during an early smoke test was removed. Pre-existing train/validation cache files (dated 2026-11-02) were left untouched. Model and dataset hashes are recorded in Section 9.

---

## 4. Results

All results below are from the primary `max_det=1000` run on 10,593 test images and 38,047 annotated pedestrian boxes.

### 4.1 AP and max-F1 operating point

| Metric | Pretrained YOLO11s | Fine-tuned YOLO11s | Delta (fine - pre) | Relative |
|---|---:|---:|---:|---:|
| Precision (max-F1) | 0.3115 | 0.3861 | +0.0746 | +23.9% |
| Recall (max-F1) | 0.4037 | 0.3690 | -0.0347 | -8.6% |
| mAP@0.50 | 0.2236 | 0.2884 | +0.0648 | +29.0% |
| mAP@0.50:0.95 | 0.1200 | 0.1629 | +0.0429 | +35.8% |

The fine-tuned model has higher AP at both IoU ranges and higher precision at its own max-F1 point. Its max-F1 recall is slightly lower than the pretrained model's max-F1 recall.

### 4.2 Fixed operating point: `conf=0.25`, IoU=0.50 (pipeline default)

| Metric | Pretrained YOLO11s | Fine-tuned YOLO11s | Delta (fine - pre) | Relative |
|---|---:|---:|---:|---:|
| True positives | 19,179 | 10,567 | -8,612 | -44.9% |
| False positives | 63,308 | 11,867 | -51,441 | -81.3% |
| False negatives | 18,868 | 27,480 | +8,612 | +45.6% |
| Precision | 0.2325 | 0.4710 | +0.2385 | +102.6% |
| Recall | 0.5041 | 0.2777 | -0.2264 | -44.9% |
| F1 | 0.3182 | 0.3494 | +0.0312 | +9.8% |

At the pipeline's default threshold, the fine-tuned model is much more precise and emits far fewer false positives, but it detects only 10,567 of the 38,047 annotated boxes versus 19,179 for the pretrained model. The F1 gain is modest (+9.8%) because the precision gain is largely offset by the recall loss.

### 4.3 Prediction volume and per-image behaviour

| Metric | Pretrained YOLO11s | Fine-tuned YOLO11s | Delta (fine - pre) | Relative |
|---|---:|---:|---:|---:|
| Boxes @ conf 0.001 | 1,328,794 | 966,295 | -362,499 | -27.3% |
| Boxes @ conf 0.250 | 82,487 | 22,434 | -60,053 | -72.8% |
| Boxes @ conf 0.500 | 44,473 | 9,243 | -35,230 | -79.2% |
| Max boxes in one image | 303 | 226 | -77 | -25.4% |
| Per-image boxes @0.25, median | 8.0 | 2.0 | -6.0 | -75.0% |
| Per-image boxes @0.25, mean | 7.79 | 2.12 | -5.67 | -72.8% |
| Per-image boxes @0.25, p95 | 16.0 | 6.0 | -10.0 | -62.5% |
| Per-image boxes @0.25, max | 24 | 17 | -7 | -29.2% |

The pretrained model is much more aggressive: at `conf=0.25` it produces a median of 8 boxes per image, versus 2 for the fine-tuned model. This is the direct source of both its higher recall and its much higher false-positive count.

### 4.4 Speed

| Metric | Pretrained YOLO11s | Fine-tuned YOLO11s | Delta (fine - pre) | Relative |
|---|---:|---:|---:|---:|
| Preprocess (ms/image) | 2.33 | 2.77 | +0.44 | +18.8% |
| Inference (ms/image) | 115.79 | 114.35 | -1.44 | -1.2% |
| Postprocess (ms/image) | 10.80 | 10.26 | -0.54 | -5.0% |
| FPS inference-only | 8.64 | 8.74 | +0.11 | +1.3% |
| FPS end-to-end | 7.76 | 7.85 | +0.09 | +1.2% |

The speed difference is negligible. The fine-tuned model is about 1.2% faster in inference and end-to-end throughput, which is within run-to-run noise and consistent with its slightly smaller 1-class head.

The measured throughput is low for YOLO11s at 640 (about 8.6-8.7 FPS inference-only, 7.8 FPS end-to-end). A direct benchmark on this machine confirmed the validator timing (about 90-120 ms/image). The absolute value is hardware/environment-specific and should not be generalized to other machines. It is reported as measured, not adjusted.

### 4.5 `max_det` sensitivity

| Model | `max_det` | Boxes @0.001 | Max boxes/image | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Pretrained | 300 | 1,328,791 | 300 | 0.223617 | 0.12001062 |
| Pretrained | 1000 | 1,328,794 | 303 | 0.223617 | 0.12001064 |
| Fine-tuned | 300 | 966,295 | 226 | 0.288428 | 0.16294832 |
| Fine-tuned | 1000 | 966,295 | 226 | 0.288428 | 0.16294832 |

The cap affected only three low-confidence pretrained boxes and changed mAP50-95 by 1.7e-08. The uncapped run is used as the primary result for symmetry; the capped run supports the same conclusions.

### 4.6 Derived deltas

- mAP@0.50: **+0.0648 absolute, +29.0% relative** for fine-tuned.
- mAP@0.50:0.95: **+0.0429 absolute, +35.8% relative** for fine-tuned.
- Max-F1 precision: **+0.0746 absolute, +23.9% relative** for fine-tuned.
- Max-F1 recall: **-0.0347 absolute, -8.6% relative** for fine-tuned.
- Fixed `conf=0.25` precision: **+0.2385 absolute, +102.6% relative** for fine-tuned.
- Fixed `conf=0.25` recall: **-0.2264 absolute, -44.9% relative** for fine-tuned.
- Fixed `conf=0.25` F1: **+0.0312 absolute, +9.8% relative** for fine-tuned.
- False positives at `conf=0.25`: **51,441 fewer** for fine-tuned (-81.3%).
- False negatives at `conf=0.25`: **8,612 more** for fine-tuned (+45.6%).
- Boxes at `conf=0.25`: **60,053 fewer** for fine-tuned (-72.8%).
- Boxes at `conf=0.50`: **35,230 fewer** for fine-tuned (-79.2%).
- Inference time: **1.44 ms/image lower** for fine-tuned (-1.2%).
- End-to-end FPS: **+0.09** for fine-tuned (+1.2%).

---

## 5. Visual comparison

Twelve test frames were rendered with four panels each: original + ground truth, pretrained detections, fine-tuned detections, and an overlay of both models. Predictions use `conf=0.25`, `iou=0.70`, `imgsz=640`, class 0 only. This is a **display threshold**, not the AP threshold (`conf=0.001`).

### 5.1 Selection method

The ten required examples were selected using **ground-truth geometry and PIE occlusion metadata only**, without looking at either model's predictions:

- **Small or distant:** at least two GT boxes, median GT box area in the smallest quartile.
- **Partially occluded:** at least one PIE `part` occlusion box and at least two GT boxes.
- **Multiple close together:** at least three GT boxes with the strongest pairwise GT box overlap.
- **Crowded:** high GT box count (at least 9 pedestrians).
- **Clear, unobstructed:** one to three GT boxes, all with `occlusion=none`, larger than the 75th percentile area, and no GT box overlap.

Two additional examples are labelled **model disagreement supplement**. They were chosen by the largest absolute difference in detection count at `conf=0.25` among a deterministic 20-frame-per-video sample. The criterion is symmetric and is not a quality judgment; these examples illustrate the operating-point gap, not a winner.

### 5.2 Visual index

| Category | Frame | GT | Pretrained @0.25 | Fine-tuned @0.25 | Montage |
|---|---|---:|---:|---:|---|
| small_or_distant | video_0009 / frame_008862 | 3 | 1 | 0 | [PNG](visuals/small_or_distant/video_0009_frame_008862_montage.png) |
| small_or_distant | video_0004 / frame_012882 | 2 | 1 | 0 | [PNG](visuals/small_or_distant/video_0004_frame_012882_montage.png) |
| partially_occluded | video_0012 / frame_012592 | 10 | 10 | 1 | [PNG](visuals/partially_occluded/video_0012_frame_012592_montage.png) |
| partially_occluded | video_0007 / frame_006021 | 8 | 14 | 7 | [PNG](visuals/partially_occluded/video_0007_frame_006021_montage.png) |
| multiple_close | video_0006 / frame_006365 | 10 | 6 | 7 | [PNG](visuals/multiple_close/video_0006_frame_006365_montage.png) |
| multiple_close | video_0008 / frame_015531 | 6 | 1 | 0 | [PNG](visuals/multiple_close/video_0008_frame_015531_montage.png) |
| crowded | video_0010 / frame_005645 | 11 | 11 | 3 | [PNG](visuals/crowded/video_0010_frame_005645_montage.png) |
| crowded | video_0001 / frame_014857 | 11 | 15 | 4 | [PNG](visuals/crowded/video_0001_frame_014857_montage.png) |
| clear_unobstructed | video_0011 / frame_003540 | 1 | 15 | 1 | [PNG](visuals/clear_unobstructed/video_0011_frame_003540_montage.png) |
| clear_unobstructed | video_0003 / frame_001590 | 1 | 10 | 1 | [PNG](visuals/clear_unobstructed/video_0003_frame_001590_montage.png) |
| model_disagreement_supplement | video_0004 / frame_004530 | 5 | 20 | 4 | [PNG](visuals/model_disagreement_supplement/video_0004_frame_004530_montage.png) |
| model_disagreement_supplement | video_0013 / frame_012540 | 5 | 16 | 0 | [PNG](visuals/model_disagreement_supplement/video_0013_frame_012540_montage.png) |

Each frame directory also contains the four individual panels (`original_gt`, `pretrained`, `finetuned`, `overlay`). The machine-readable manifest is `visuals/visual_manifest.json`.

### 5.3 Observations

- **Small or distant pedestrians:** both models struggle. The fine-tuned model produces no detections at `conf=0.25` in both selected frames; the pretrained model finds one. Small boxes are a weak spot for both.
- **Partially occluded:** the pretrained model is more sensitive (10 and 14 detections versus 1 and 7). The fine-tuned model is conservative and misses many occluded boxes at the display threshold.
- **Multiple pedestrians close together:** mixed. The fine-tuned model finds slightly more in one frame (7 vs 6) and fewer in the other (0 vs 1). There is no consistent winner in this small sample.
- **Crowded scenes:** the pretrained model is much more aggressive (11 and 15 detections versus 3 and 4). The fine-tuned model misses many annotated pedestrians at `conf=0.25`, which is consistent with its lower recall.
- **Clear, unobstructed:** with one annotated pedestrian, the pretrained model produces 15 and 10 detections, while the fine-tuned model produces one. The extra pretrained detections are likely unannotated people or objects. This is direct visual evidence of the PIE annotation caveat below.
- **Model disagreement supplement:** the pretrained model produces 20 and 16 detections versus 4 and 0 for the fine-tuned model. The frames were selected by a symmetric count difference, so they illustrate the operating-point gap rather than a quality ranking.

### 5.4 PIE annotation caveat

PIE annotations are selective and non-exhaustive. They were created for crossing-intention analysis and do not necessarily label every pedestrian in every frame. Therefore:

- An unmatched prediction is not necessarily a real false positive; it may be an unannotated pedestrian.
- Absolute precision is a lower bound, especially for the more aggressive pretrained model.
- AP is also computed against the provided annotations, so it inherits the same limitation.
- The fine-tuned model may have learned the annotation policy (which pedestrians are included), which improves measured precision but could reduce recall of real, unannotated pedestrians.

The clear-scene examples support this: the pretrained model detects many people that the GT does not include. The fine-tuned model's precision advantage is therefore partly, but not entirely, an annotation artifact. Its higher AP and its recall gap on annotated boxes are still measured consistently against the same labels.

---

## 6. Interpretation

### 6.1 AP vs operating point

AP evaluates the full confidence range and rewards ranking and localization quality. The fine-tuned model is clearly better by AP: +29.0% mAP@0.50 and +35.8% mAP@0.50:0.95. This is the strongest evidence that fine-tuning improved the detector's overall quality on PIE.

The fixed-threshold results answer a different question. At the pipeline's default `conf=0.25`, the fine-tuned model shifts the operating point toward precision: it cuts false positives by 81.3% and raises precision from 0.2325 to 0.4710, but it detects 44.9% fewer annotated pedestrians. F1 improves only modestly (+9.8%) because the recall loss nearly offsets the precision gain.

### 6.2 Why the models behave differently

- The pretrained COCO model has seen a very large, diverse person dataset. It is aggressive and generalizes to varied poses and occlusions, but it also produces many false positives on this domain. Its broad `person` class includes seated, standing, and partially visible people, not only crossing pedestrians.
- The fine-tuned model is specialized to PIE pedestrians and has a single-class head. It is more conservative and precise, but it misses more annotated boxes at a fixed threshold. This may reflect domain-specific confidence calibration and the PIE annotation policy.
- The fine-tuned model is slightly smaller (1-class head) and essentially the same speed.

### 6.3 What this means for the pipeline

The pipeline uses `conf=0.25` and `classes=[0]` for pedestrian detection and tracking. At that exact operating point:

- If false tracks and downstream false crossing events are the main pain point, the fine-tuned model is attractive: far fewer boxes, much higher precision, slightly better F1, same speed.
- If missing a pedestrian is safety-critical, the pretrained model's higher recall (0.504 vs 0.278) is a major advantage. The fine-tuned model misses 27,480 annotated boxes versus 18,868 for the pretrained model.
- The fine-tuned model is not a drop-in replacement at `conf=0.25` if the pipeline depends on high recall.

### 6.4 Checkpoint provenance

The fine-tuned checkpoint was selected on `set06` during the prior training run. The test split (`set04`) was not used for training or threshold selection. The test metrics are higher than the validation metrics (mAP50 0.288 vs 0.259; mAP50-95 0.163 vs 0.138), which is plausible given split differences and the class-restricted evaluation, but the two should not be compared directly.

---

## 7. Recommendation

**Conditional recommendation: do not replace the detector at `conf=0.25` without threshold calibration.**

1. **If precision and false-alarm reduction are the priority**, and the pipeline can tolerate lower recall or recover it through tracking and interaction logic, adopt the fine-tuned YOLO11s. At `conf=0.25` it cuts false positives by 81.3% with slightly better F1 and the same speed.

2. **If recall is safety-critical**, keep the pretrained YOLO11s at `conf=0.25` for now, or use it as a high-recall proposal generator. The fine-tuned model's 44.9% recall drop at the same threshold is too large to ignore for crossing-intention safety.

3. **Recommended next step (best of both):** calibrate the fine-tuned model's confidence threshold on the **validation split (`set06`)** to a target recall comparable to the pretrained model (for example, recall at or above 0.50), then measure the precision/FP cost and the downstream tracking and crossing-intention metrics. Do not tune on `set04`.

   - If the fine-tuned model retains a meaningful precision or FP advantage at matched recall, adopt it.
   - If it does not, keep the pretrained model or use a hybrid: pretrained high-recall proposals plus fine-tuned verification/tracking.

4. **Evaluate downstream, not just detection AP.** Compare tracking ID switches, trajectory completeness, interaction/crossing precision and recall, and false alarms per minute. A detector with better AP but worse recall at the pipeline threshold may still produce worse crossing-intention performance.

5. **Do not change the crossing-intention model** as part of this detector comparison. The recommendation is limited to the detector choice and threshold calibration.

---

## 8. Limitations

- **Selective annotations:** PIE labels are not exhaustive. Unmatched detections may be real unannotated pedestrians; absolute precision is a lower bound, especially for the pretrained model.
- **Single dataset and split:** results are specific to PIE `set04` and this export. No cross-dataset, cross-camera, or cross-weather generalization was tested.
- **One checkpoint per model:** the fine-tuned checkpoint is a single `best.pt` from `train-5` (epoch 11) and may not be fully converged or optimally selected. The comparison is between these specific checkpoints, not the best possible fine-tuned model.
- **Semantic mismatch:** COCO `person` and PIE `pedestrian` are related but not identical categories. The pretrained model may detect seated or non-crossing people; the fine-tuned model may be tuned to the PIE inclusion policy.
- **Speed environment:** timing was measured on one Windows laptop with an RTX 4070 Laptop GPU and OneDrive-backed storage. Absolute FPS is not a hardware benchmark; the relative comparison is the useful part.
- **Visual sample:** the 12 frames are illustrative, not a statistically representative sample. The required categories were selected without model outputs, but the sample is small.
- **Fixed-threshold matching:** the `conf=0.25`, IoU=0.50 greedy matching is a simple protocol, not AP, and may differ from the tracker's association behavior.
- **`max_det`:** the primary run uses 1000 instead of the Ultralytics default 300 to avoid truncation asymmetry. The sensitivity run shows the effect is negligible.
- **No test-time augmentation or ensembling:** both models were evaluated in single-scale, non-augmented mode.

---

## 9. Reproducibility and artifacts

### 9.1 Commands

```powershell
# Phase 1: readiness and hashes
& ".venv\Scripts\python.exe" "artifacts\detector_comparison\scripts\run_readiness.py"

# Primary evaluation (uncapped, max_det=1000)
$env:SMARTPED_EVAL_MAX_DET="1000"
$env:SMARTPED_EVAL_TAG="_maxdet1000"
& ".venv\Scripts\python.exe" "artifacts\detector_comparison\scripts\run_evaluation.py"
Remove-Item Env:SMARTPED_EVAL_MAX_DET
Remove-Item Env:SMARTPED_EVAL_TAG

# Sensitivity run (Ultralytics default max_det=300)
& ".venv\Scripts\python.exe" "artifacts\detector_comparison\scripts\run_evaluation.py"

# Visual selection and rendering
& ".venv\Scripts\python.exe" "artifacts\detector_comparison\scripts\select_visual_frames.py"
& ".venv\Scripts\python.exe" "artifacts\detector_comparison\scripts\render_visual_comparison.py"

# Tables and visual index
& ".venv\Scripts\python.exe" "artifacts\detector_comparison\scripts\build_comparison_table.py"
& ".venv\Scripts\python.exe" "artifacts\detector_comparison\scripts\build_visual_index.py"
```

### 9.2 Key artifacts

| Artifact | Path |
|---|---|
| Readiness evidence | `artifacts/detector_comparison/configs/readiness.json` |
| Primary config | `artifacts/detector_comparison/configs/evaluation_config_maxdet1000.json` |
| Sensitivity config | `artifacts/detector_comparison/configs/evaluation_config.json` |
| Visual selection | `artifacts/detector_comparison/configs/visual_selection.json` |
| Primary summary | `artifacts/detector_comparison/metrics/evaluation_summary_maxdet1000.json` |
| Sensitivity summary | `artifacts/detector_comparison/metrics/evaluation_summary.json` |
| Pretrained metrics | `artifacts/detector_comparison/metrics/eval_pretrained_yolo11s_maxdet1000.json` |
| Fine-tuned metrics | `artifacts/detector_comparison/metrics/eval_finetuned_train5_best_maxdet1000.json` |
| Comparison tables | `artifacts/detector_comparison/metrics/comparison_tables.md` |
| Derived deltas | `artifacts/detector_comparison/metrics/derived_deltas.json` |
| Visual index | `artifacts/detector_comparison/metrics/visual_index.md` |
| Evaluation logs | `artifacts/detector_comparison/logs/eval_*_maxdet1000.log` |
| Ultralytics runs | `artifacts/detector_comparison/runs/*_maxdet1000/` |
| Visuals | `artifacts/detector_comparison/visuals/` |

### 9.3 Integrity checks

- Pretrained SHA-256: `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` (unchanged).
- Fine-tuned SHA-256: `9e1d876ff9ccc49d03f43da9dbe5bb4df71c2a7514dbaad24585e4a1f855ecfb` (unchanged).
- Original dataset files: no modifications. The test-split cache created during the early smoke test was removed; the pre-existing train/validation caches (dated 2026-11-02) were left untouched.
- Existing pipeline modules: no modifications.
- The evaluation wrote only under `artifacts/detector_comparison/` and the temporary mirror under `%TEMP%\opencode\smartped_detector_eval\pie_detection_yolo`. The mirror can be deleted; it is not part of the project outputs.
- The working tree also shows unrelated changes to `README.md` and `.gitignore`; this evaluation did not write to those files.

---

## Appendix A: Raw metric values

Primary run (`max_det=1000`):

```json
{
  "pretrained_yolo11s": {
    "results_dict": {
      "metrics/precision(B)": 0.3115251968603616,
      "metrics/recall(B)": 0.4037374825873262,
      "metrics/mAP50(B)": 0.2236172177128881,
      "metrics/mAP50-95(B)": 0.12001063812111275,
      "fitness": 0.12001063812111275
    },
    "fixed_conf_0.25_iou_0.50": {
      "tp": 19179,
      "fp": 63308,
      "fn": 18868,
      "precision": 0.23250936511207826,
      "recall": 0.5040870502273503,
      "f1": 0.31823385932599924
    }
  },
  "finetuned_train5_best": {
    "results_dict": {
      "metrics/precision(B)": 0.3861309476961059,
      "metrics/recall(B)": 0.36902186617131644,
      "metrics/mAP50(B)": 0.2884279002249578,
      "metrics/mAP50-95(B)": 0.16294831890526534,
      "fitness": 0.16294831890526534
    },
    "fixed_conf_0.25_iou_0.50": {
      "tp": 10567,
      "fp": 11867,
      "fn": 27480,
      "precision": 0.47102612106623876,
      "recall": 0.2777354324913922,
      "f1": 0.34943205304145103
    }
  }
}
```

## Appendix B: Interpretation checklist for deployment

- [ ] Calibrate the fine-tuned model's confidence threshold on `set06`, not `set04`.
- [ ] Target a recall level consistent with the crossing-intention safety requirement.
- [ ] Re-measure precision, FP volume, and F1 at the calibrated threshold.
- [ ] Run the full SmartPed pipeline and compare tracking and crossing-intention metrics, not only detection AP.
- [ ] Visually inspect a sample of unmatched detections to estimate how many are unannotated real pedestrians.
- [ ] Keep the pretrained checkpoint available as a high-recall fallback or proposal generator.
- [ ] Do not change the crossing-intention model as part of detector selection.
