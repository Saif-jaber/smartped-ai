# SmartPed AI System Quality Audit

**Audit scope:** Existing source, configuration, saved experiment outputs, tests, and the original and updated 60-second annotated videos.  
**Audit mode:** Read-only. No source, configuration, model, dataset, or prior experiment output was changed. No training or inference experiment was run for this audit.  
**Test verification:** `python -m unittest discover -s tests -q` — **98 tests passed**.

## Executive assessment

SmartPed AI has a functioning, modular offline pipeline and has generated a complete 60-second video-run output. Its strongest defensible results are limited to specific measurements: the existing crossing-intention model has held-out PIE annotation-feature metrics; pedestrian/vehicle boxes are temporally aligned in the demonstrated run; and the updated annotated MP4 opens, decodes fully, and retains the source frame rate and resolution.

The project is **not yet an accuracy-validated pedestrian/vehicle perception system**. The project artifacts do not contain usable detector training results or precision/recall/mAP evaluation. The tracker has no measured identity metrics, and sampled frames show overlapping detections assigned different IDs. Pedestrian movement-state agreement is poor. The crossing-intention model cannot produce valid predictions from the current tracker interactions; the integration correctly reports all such predictions unavailable. Interaction distances and motion remain image-space descriptors, not calibrated safety or collision estimates.

| Major feature | Classification | Evidence-based conclusion |
|---|---|---|
| PIE data ingestion and split handling | Working but needs stronger validation | Deterministic parsing and set-level splits exist, but the saved dataset validation report is **FAIL** with three frame-set mismatches and two incomplete ego-telemetry warnings. |
| PIE pedestrian detection preparation | Working but needs stronger validation | A reproducible sparse, set-split YOLO dataset exists. PIE is selective, not exhaustive, so it is not a general negative/background or complete-object benchmark. |
| PIE-trained pedestrian detector | Not integrated | A one-epoch training-arguments file exists, but no usable fine-tuned detector checkpoint or saved mAP/precision/recall results were found. The demonstrated pipelines use pretrained `yolo11s.pt`. |
| Pedestrian detection and tracking inference | Working but needs stronger validation | The saved 60-second run has tracked pedestrian observations; detector and identity accuracy have not been established. |
| Pedestrian movement classification | Experimental | On 448 matched observations, the best tested method reached 0.326 agreement; the original baseline reached 0.219. |
| Camera-motion compensation | Experimental | Synthetic tests and flow diagnostics exist, but the compensated classifier did not outperform the calibrated raw-motion baseline on the evaluation clip. No world-motion calibration exists. |
| Vehicle detection and tracking | Working but needs stronger validation | A 60-second run and annotation-box coverage comparison exist. Only annotated cars were evaluable; no tracking-identity metrics exist. |
| Crossing intention on PIE annotation features | Working but needs stronger validation | The saved Random Forest has held-out set04 metrics on 328 eligible annotation-derived samples; the test set is one recording set. |
| Crossing intention on YOLO/BoT-SORT tracks | Not supported by available evidence | Input features have different sources/semantics and no verified PIE-to-tracker identity map. The integration appropriately produces no probability. |
| Pedestrian–vehicle interaction descriptors | Working but needs stronger validation | Exact frame/timestamp alignment and pixel-space separation/movement are implemented; thresholds and interpretation are not validated against safety ground truth. |
| Collision risk, physical TTC, or real-world speed | Not supported by available evidence | No calibrated geometry or suitable collision/near-miss labels support these claims. |
| End-to-end video processing and annotation | Working but needs stronger validation | The 60-second output fully decodes at source resolution/FPS; throughput is below real time and only one video has been demonstrated. |
| Automated tests | Working and validated | All 98 tests pass. Most are unit/synthetic or mocked tests and do not establish learned-model accuracy or track identity quality. |

## A. Pedestrian detection

**Classification:** Working but needs stronger validation  
**Priority:** High  
**Estimated effort:** 3–7 days for a measured evaluation baseline; additional 1–3 weeks if fine-tuning is justified by that baseline.

### Current implementation

The detection-preparation pipeline creates a one-class YOLO dataset from visible PIE pedestrian boxes. It samples every 30th frame per video, retaining track boundaries and action/crossing/occlusion transitions, and assigns whole recording sets to splits. The generated dataset has 32,989 images and 131,015 labels from 740,901 source pedestrian boxes. The detector pipelines use YOLO11s and a confidence threshold of 0.25; the video runner and saved vehicle metrics identify the active checkpoint as pretrained `yolo11s.pt`.

The project contains `runs/detect/train/args.yaml`, which records a run configured with `epochs: 1`, `imgsz: 640`, and the prepared PIE data. No usable `best.pt`/`last.pt` fine-tuned checkpoint or `results.csv`/saved detection metric report was found in the inspected training output. The current pipeline therefore cannot be credited with using or validating a PIE-fine-tuned detector.

### Verified functionality and quantitative evidence

- The 60-second run contains 5,007 pedestrian tracker observations. These are tracked outputs, not a complete count of raw detector proposals; the trajectory video adapter skips boxes without tracker IDs.
- Retained pedestrian observation confidence: min 0.250, median 0.598, 90th percentile 0.766, max 0.891.
- Vehicle pipeline confidence threshold is also 0.25; retained vehicle-observation confidence is min 0.250, median 0.645, 90th percentile 0.900, max 0.964.
- Existing visual spot-checking rendered 60 selected annotation frames. Twelve were manually reviewed: five marked correct, none incorrect, seven uncertain. Forty-seven frames in the reported mismatch interval remain pending review. This is a limited annotation-visual check, not a detector evaluation.

No valid pedestrian detector precision, recall, mAP50, or mAP50-95 was found. The confidence distributions are descriptive only; they do not indicate correctness.

### Weaknesses and missing validation

- PIE pedestrian labels are selective and not exhaustive. A frame without a pedestrian box is not necessarily a confirmed pedestrian-free negative. Unmatched predictions cannot be called false positives without exhaustive review.
- The sparse sample is not every annotated frame. It includes valid visible boxes and preserves excluded annotation metadata, but frame sampling limits temporal diversity.
- There is no saved threshold sweep, held-out detector benchmark, object-size breakdown, or measured false-positive/false-negative review. Small, distant, occluded, and partially visible pedestrian performance is unknown.
- Dataset preparation used recording-set splits, which is appropriate, but the detector's actual training/evaluation outcome and the checkpoint used by inference are not reproducibly documented together.

### Recommended improvement

First evaluate the exact deployed checkpoint on a manually reviewed, representative, set-separated image sample. Report precision/recall and AP only where the labels are sufficiently exhaustive; separately report annotation match coverage where they are not. Stratify by box size, occlusion, and distance/image region, and perform a confidence sweep. Only fine-tune if the measured baseline justifies it; retain a checkpoint hash, training configuration, validation outputs, and exact inference checkpoint path.

## B. Pedestrian tracking

**Classification:** Working but needs stronger validation  
**Priority:** High  
**Estimated effort:** 3–7 days for a curated identity-labelled evaluation and metrics; longer if re-identification needs redesign.

### Current implementation

Pedestrians are tracked through Ultralytics YOLO tracking with the project's BoT-SORT configuration. The tracker uses high/low/new thresholds of 0.25/0.10/0.25, a 360-frame buffer, match threshold 0.8, sparse optical-flow global-motion compensation, and ReID enabled with `model: auto`. The same principal tracker settings are used in the vehicle BoT-SORT configuration.

### Verified functionality and quantitative evidence

- In the set01/video_0001 first 60 seconds, 5,007 pedestrian observations were assigned to 173 tracker IDs.
- The trajectory report counts 3,886 unobserved frame slots inside observed track spans, 415 gap events, and 108 tracks with at least one gap; 32 tracks have gaps above 15 frames.
- These values describe gaps in tracker output, not ground-truth missed detections or identity switches.

### Weaknesses and missing validation

- No IDF1, HOTA, MOTA, ID-switch count, fragmentation score, or track-precision/recall evaluation was found.
- Annotation matching is frame/box based and does not establish that a tracker ID corresponds to the same annotated person over time.
- PIE's selectively annotated objects and the absence of an audited tracker-to-annotation identity crosswalk make identity evaluation incomplete.
- The configured long track buffer and ReID may help recovery but add complexity and have not been shown to improve identity quality on these videos.

### Recommended improvement

Create a small but representative manual identity-ground-truth subset containing crossings, occlusions, distant pedestrians, and dense scenes. Match predicted and annotated boxes per frame, then review identity continuity and report standard tracking metrics plus fragmentation/ID-switch examples. Compare tracker settings only after this baseline; do not infer identity quality from observation count or gap statistics.

## C. Vehicle detection and tracking

**Classification:** Working but needs stronger validation  
**Priority:** High  
**Estimated effort:** 3–7 days for annotation review and track evaluation; 1–2 days for a focused duplicate/overlap adjudication sample.

### Current implementation

Vehicle inference uses pretrained YOLO11s with BoT-SORT and retains COCO bicycle, car, motorcycle, bus, and truck classes. PIE-to-COCO matching maps `car`, `bicycle`, `bus`, and `truck`; PIE `bike` is deliberately left unmapped because its intended class is ambiguous, and `train` is out of the selected class scope.

### Verified functionality and quantitative evidence

For the first 60 seconds of set01/video_0001:

- 13,226 tracked detections formed 429 class-aware tracks: 12,527 car, 471 truck, 204 bicycle, 21 bus, and 3 motorcycle detections.
- The matched PIE annotation interval contains 1,300 visible, valid boxes, all annotated as cars across 10 annotated vehicle tracks.
- A greedy one-to-one maximum-IoU, same-frame, same-class match found 1,233/1,300 boxes at IoU ≥ 0.5 (0.9485 annotation-box match coverage); mean matched IoU was 0.8726.
- The experiment explicitly does not count unmatched predictions as false positives because PIE vehicle labels may not be exhaustive. The match rate is not detector precision, and it does not measure tracker identity accuracy.
- Vehicle track output has 881 gap events, 8,734 missing frame slots within observed track spans, and 92 gaps above 15 frames.

Read-only overlap analysis of the saved vehicle detections found 612 same-frame, same-class, different-track box pairs with IoU ≥ 0.5; 34 had IoU ≥ 0.75 and one had IoU ≥ 0.90. Visual review of selected high-overlap frames showed examples where different IDs appear to cover the same physical car. These are candidate duplicate/fragmented track observations, not a formal count of duplicate vehicles; some overlap can also arise from occlusion and annotation ambiguity.

### Weaknesses and missing validation

- Only cars are present in the corresponding interval's PIE vehicle labels. Bicycle, bus, motorcycle, and truck detection quality is not evaluated there.
- There is no exhaustive negative set, detector precision/recall, class-wise AP, confidence calibration, or track identity metric.
- PIE `bike` and `train` mapping policies appropriately avoid unsupported equivalence, but leave portions of the scene outside evaluation.
- Vehicle box overlap examples require identity-level adjudication before assigning cause to NMS, tracker fragmentation, or separate adjacent vehicles.

### Recommended improvement

Review a stratified sample of both matched and unmatched car detections, and adjudicate the high-IoU different-ID candidate pairs over short frame windows. Extend the labelled benchmark to the vehicle classes the project intends to claim. Report frame-level box metrics separately from track-level identity metrics; do not call the 0.9485 match rate precision or recall.

## D. Trajectory extraction and movement classification

**Classification:** Movement classification is Experimental; trajectory extraction is Working but needs stronger validation  
**Priority:** High  
**Estimated effort:** 2–4 days for a stronger image-space calibration/evaluation protocol; more if new movement labels or calibrated geometry are required.

### Current implementation

Trajectory features are calculated from tracked bounding boxes and timestamps. Missing observations are counted and are not interpolated. The movement classifier uses smoothed image-space displacement/velocity, a configurable maximum gap, and a standing-speed threshold. The camera-motion module estimates background flow from Lucas–Kanade features, masks tracked pedestrian boxes, and uses local flow with a global affine fallback. Ego telemetry is summarized for context but is not used to convert image measurements to world coordinates.

### Verified functionality and quantitative evidence

On the 60-second set01/video_0001 trajectory run:

- 5,007 observations, 173 tracks; 4,640 labelled walking, 150 standing, 217 unknown.
- Expected slots within observed track spans: 8,893; observations: 5,007; reported observation coverage ratio: 0.563. This is not recall against all people or all frames.
- 3,886 frame slots are missing within observed track spans; 71.2% of those missing slots fall in gaps over 15 frames.

The movement experiment compares the same 448 matched prediction/annotation observations:

| Method | Threshold | Agreement / accuracy | Balanced accuracy | Macro F1 | Confusion matrix (true rows, standing then walking) |
|---|---:|---:|---:|---:|---|
| Original baseline | 10.00 px/s | 0.219 | 0.500 | 0.179 | `[[0, 350], [0, 98]]` |
| Calibrated, no compensation | 90.74 px/s | 0.326 | 0.558 | 0.319 | `[[51, 299], [3, 95]]` |
| Camera-motion compensated | 35.36 px/s | 0.286 | 0.543 | 0.269 | `[[30, 320], [0, 98]]` |

The original threshold classifies all 448 matched samples as walking. Camera compensation recovers valid flow on 4,999/4,999 observations, mostly local flow, but this availability metric is not a correctness measure. The calibrated uncorrected threshold performs better than the compensated method on this clip.

### Weaknesses and missing validation

- Walking/standing agreement remains poor. Even the best tested method misclassifies 299/350 true-standing observations and has low standing recall.
- The calibration clips (set03/video_0001 and two set05 clips) and evaluation clip (set01/video_0001) are recording-disjoint but all lie in the project's train partition. This is not a held-out test-set evaluation of the classifier/calibration protocol.
- PIE frame-level action labels are not direct measurements of pixel velocity; box jitter, camera motion, depth, perspective, and changes in apparent scale affect the comparison.
- Compensation masks pedestrian boxes but cannot establish true background motion in the presence of moving objects, parallax, or independently moving scene regions. An affine transform is only an image-motion approximation.
- Ego telemetry has no camera intrinsics/extrinsics, mounting calibration, or road-plane homography. No output is a real-world speed.
- Gaps and 0.563 coverage are within observed identity spans; they do not explain whether a missing record is due to detector miss, tracker filtering, occlusion, or ID change.

### Recommended improvement

Before using movement labels in downstream claims, evaluate on recording sets excluded from threshold selection, stratify by pedestrian size/visibility and camera motion, and manually review uncertain transitions. Compare raw and compensated image-space features using the same annotations and include per-class confusion metrics. If physical velocity is ever needed, obtain camera/road calibration and an independently validated projection; do not rescale pixels using ego speed alone.

## E. Crossing-intention prediction

**Classification:** Working but needs stronger validation on PIE annotation features; tracker-based prediction is Not supported by available evidence  
**Priority:** High  
**Estimated effort:** 1–2 days to preserve and present the existing limitation clearly; approximately 2–4 weeks for tracker-derived feature/label alignment, model development, and a held-out evaluation.

### Current implementation

The saved Random Forest and Logistic Regression use 21 features derived from PIE spatial annotations and frame-level attributes. The sample is taken at `critical_point`. The target is a visible crossing action during the following 30 frames (one second); already-crossing, irrelevant, and insufficiently labelled future cases are excluded under the saved rules. Preprocessing is fitted on training data, and thresholds are selected on validation only.

The model split is by recording set: train sets 01/02/03/05 (492 samples), validation set06 (103), test set04 (328). The Random Forest validation threshold is frozen at 0.401957; set04 is not used for fitting or threshold tuning.

### Verified functionality and quantitative evidence

For the held-out set04 annotation-derived samples (328 eligible tracks; 238 negative and 90 positive), the Random Forest reports:

- Precision 0.566, recall 0.856, F1 0.681, average precision 0.724.
- Confusion matrix `[[179, 59], [13, 77]]` with rows actual negative/positive and columns predicted negative/positive.
- Validation set06 metrics were precision 0.650, recall 0.897, F1 0.754, average precision 0.767.
- The held-out F1 is 0.072 lower than validation F1. Since each split is a single recording set, the difference does not establish the cause or estimate across-route variability.

Feature metadata explicitly excludes identifiers and future/target fields and documents that the numeric motion features are normalized image-space estimates. The saved metrics include the frozen thresholds and the test evaluation.

### Compatibility with the live tracker

The model requires annotation-derived `action_at_prediction`, `look_at_prediction`, and `gesture_at_prediction`, and it was trained on annotation boxes/histories at the annotation-defined critical point. The current interaction input contains YOLO/BoT-SORT boxes and tracker-local IDs, does not supply these features with equivalent semantics, and has no verified PIE-identity-to-tracker-ID mapping. The integration metrics mark tracker inference incompatible and report **0 of 29,564 interactions with valid crossing predictions**; all 29,564 are explicitly unavailable. Set01 is also part of the model's training split.

The integration's explicit unavailable value is the correct behavior. It does not imply the model failed on compatible annotation features, nor does the set04 test metric validate tracker-derived predictions.

### Weaknesses and missing validation

- Validation and test each contain only one recording set; the estimates may be sensitive to route, scene, and sample composition.
- The test has 90 positive samples and 238 negative samples. Precision is moderate, so false positives remain material even on the annotation-defined task.
- No feature-importance/interpretability report or probability calibration analysis was found in the saved metrics/report.
- There is no live tracker feature contract, crosswalk, or tracker-derived held-out test.

### Recommended improvement

Keep the existing Random Forest unchanged and describe its scope as annotation-feature PIE evaluation. To generate live predictions, first build and audit track-to-annotation correspondences on training recordings, then derive the same features only if their temporal and semantic definitions can be preserved. Otherwise train a new tracker-feature model from appropriately aligned training data, retain set-level separation, freeze the threshold on validation, and evaluate on a held-out recording set. This requires new model training and a defensible identity/label alignment process; do not fill missing features with guessed values.

## F. Pedestrian–vehicle interaction analysis

**Classification:** Working but needs stronger validation  
**Priority:** Medium for descriptive image-space output; High before any safety/risk interpretation  
**Estimated effort:** 3–5 days for geometry and scene-stratified review; additional annotation effort for near-miss or collision evaluation.

### Current implementation

The interaction pipeline joins observations on exact `(set_id, video_id, frame_index)` keys, validates timestamps, and preserves pedestrian and class-aware vehicle IDs separately. Pedestrian location is the bounding-box bottom center. Separation is Euclidean image-space distance from that footpoint to the nearest point on the vehicle rectangle. Relative movement is derived from changes in footpoint and vehicle-box center. Gaps over the configured limit invalidate movement estimates. Closest approach is the minimum observed distance within the longest valid consecutive segment, not a projected future event. The `nearby` threshold is 150 pixels.

### Verified functionality and quantitative evidence

- 5,007 pedestrian observations / 173 tracks; 13,226 vehicle observations / 429 tracks.
- 29,564 pair observations and 2,543 unique pairs; 1,851 persistent and 692 transient pairs.
- Of 1,525 common frames, all 1,525 timestamps agree exactly within the configured tolerance; no timestamp mismatches were reported.
- 26,698 per-observation movement estimates are valid and 2,866 invalid; 1,487 persistent-pair summaries have valid closest observed separation and 364 are invalid.
- The integration/runner reports preserve the explicit unavailable intention prediction rather than manufacturing a value.

### Weaknesses and missing validation

- The 150-pixel threshold is experimental and depends on resolution, perspective, and object scale. It is not a physical clearance or collision-risk boundary.
- A zero pixel separation means the footpoint lies within the vehicle rectangle in the image; it does not establish physical contact.
- Camera motion, parallax, scale change, box jitter, occlusion, and track-ID changes affect the apparent separation and relative movement.
- No near-miss/collision ground truth, calibrated TTC, or real-world distance is available. The interaction pipeline does not validate safety prediction.
- Some interaction observations can be geometrically valid while an object's physical identity or exact footpoint is uncertain.

### Recommended improvement

Keep outputs explicitly labelled as image-space observations. Validate footpoint/rectangle geometry on reviewed frames and test sensitivity to the 150-pixel threshold across resolution and scene regions. If a risk model is later proposed, first create a reviewed event set with an agreed definition and appropriate camera/road geometry; report it as a new validated task rather than extending the current descriptors into risk labels.

## G. End-to-end video runner and annotated output

**Classification:** Working but needs stronger validation  
**Priority:** High for demonstrating limitations correctly; Medium for batch-processing reliability  
**Estimated effort:** 1–3 days for reproducibility/provenance checks and an additional representative video run; optimization effort depends on demonstration requirements.

### Current implementation

The runner probes the source, then calls the existing pedestrian, vehicle, trajectory, interaction, and integration implementations in sequence. It preserves set/video IDs, frame indices, and timestamps, writes separate CSVs and metrics, and renders an annotated MP4. It does not create missing detections or force unavailable crossing predictions. The updated render uses green pedestrian boxes, blue vehicle boxes, compact class/ID/confidence labels, movement labels when available, and a small panel for nearby image-space pairs.

### Verified functionality and quantitative evidence

The successful run processes the first 1,800 frames (60 seconds) of the 30-FPS, 1920×1080 set01/video_0001 source. The original annotated output and the updated render-v2 both report 1,800 frames and have been fully decoded. The updated rendering metrics confirm matching source resolution and FPS. Selected frames from both outputs were visually inspected.

- Total processing time: 145.72 seconds; 12.35 frames/second; 0.412 real-time factor. This is a batch pipeline, not real-time processing.
- GPU device recorded as CUDA.
- Pedestrian and vehicle model pipelines instantiate the shared checkpoint separately and run sequentially.
- The interaction metrics report 1,525 common frames and zero timestamp spread/mismatch.
- Render-v2 improves label placement and moves pair details into a fixed panel. Visual inspection still shows dense boxes in crowded scenes, a panel that may say `+N more`, and zero-pixel pair separations; these are presentation/geometry limitations, not evidence of collision.

### Weaknesses and missing validation

- Only one 60-second source interval is evidenced. Runtime, memory, empty-scene behavior, and identity consistency over all 53 clips are not established by that run.
- The runner is approximately 2.43 times slower than source playback on the recorded environment.
- Inference loads the shared checkpoint twice sequentially rather than reusing one detector instance. No performance comparison or memory profile was found.
- Metrics record the source and checkpoint paths but do not provide cryptographic hashes for the source, model, configuration, or code revision. A path is useful provenance but does not independently prove artifact identity.
- The test suite tests input validation, duration/frame limits, alignment, empty outputs, rendering, and cleanup paths, but does not test one full real-model pipeline run.
- The original annotated video has visibly crowded labels around close pairs. Render-v2 improves this but the frame remains busy in dense scenes; visual usability has only been sampled, not systematically scored.

### Recommended improvement

For the final demonstration, report the exact source, checkpoint, confidence, tracker settings, duration, device, throughput, and unavailable-prediction count. Keep the latest render and CSVs from the same run together. Add a lightweight run manifest with source/model/config hashes and software versions, then smoke-test a second representative recording. Optimize model reuse only if runtime matters, and preserve output equivalence if changed.

## Testing and validation audit

**Classification:** Working and validated as a test suite; model quality remains not established by tests  
**Priority:** Medium  
**Estimated effort:** 2–5 days to add curated model/identity evaluation fixtures; ongoing effort for annotation and benchmark maintenance.

### Existing tests

The full test command passed **98 tests across 11 test files**. Test areas include:

- Dataset XML/CSV parsing, frame-set mismatch reporting, track attributes, and split assignment.
- Detection data preparation, box normalization, sample selection, and leakage-resistant set splits.
- Crossing target horizon/exclusion rules, feature cutoffs, and no interpolation.
- Camera-motion compensation behavior on synthetic translation, unavailable-flow fallback, calibration/evaluation separation, movement confusion metrics, and gap diagnostics.
- Interaction footpoint geometry, exact frame alignment, duplicate-row prevention, movement validity, gap handling, and closest-observed-approach requirements.
- Integration recording/identity isolation, temporal causality, prediction expiry, missing and null predictions, schema mismatch, and empty input.
- Vehicle class mapping, annotation matching, and gap handling.
- Video input validation, frame/duration limits, timestamp alignment, missing predictions, empty detections, render-label bounds, output decoding, and resource cleanup.
- Visual spot-check parsing and review-manifest updates.

### What tests do not establish

- Passing unit tests do not establish YOLO precision/recall/mAP, tracker identity quality, movement-state validity, or crossing-intention generalization.
- Synthetic optical flow does not validate compensation quality on moving-camera PIE footage.
- Mocked pipeline tests do not establish that the real model outputs are correct or robust across routes.
- The visual spot-check sample is incomplete: only 12/60 rendered samples were manually reviewed and seven reviewed frames remain uncertain; 47 mismatch frames are pending.
- No complete CI test re-runs inference, verifies trained-checkpoint provenance, evaluates detector/tracker metrics, or checks regression against a curated ground-truth sample.

### Recommended improvement

Keep unit tests for deterministic logic and add separate, small, versioned evaluation fixtures with reviewed labels. Report model-quality metrics independently from software test results. A full expensive inference run should remain an explicitly scheduled validation step, not a prerequisite for routine unit tests.

## Remediation roadmap

### Five highest-priority issues

| Rank | Issue | Why it matters | Effort / prerequisites |
|---:|---|---|---|
| 1 | No measured detector precision/recall/AP or verified fine-tuned checkpoint | The project cannot substantiate pedestrian/vehicle detector accuracy or explain confidence-threshold behavior. | 3–7 days to construct and evaluate a reviewed benchmark. If needed, 1–3 weeks for fine-tuning and repeat evaluation. Requires labelled examples and confirmed checkpoint provenance. |
| 2 | Tracker identity quality is unmeasured; overlapping IDs appear in sampled vehicle frames | Interactions and trajectories depend on stable IDs, but there are no IDF1/HOTA/ID-switch results. | 3–7 days for a curated, identity-labelled subset and metrics; more for tracker changes. Requires manual identity review or a reliable annotation crosswalk. |
| 3 | Movement classification is weak and camera-motion compensation did not improve on the calibrated raw baseline | The best measured agreement is 0.326; walking/standing should not be presented as reliable behavioral ground truth. | 2–4 days for better evaluation/calibration and review, without retraining a detector. More annotation may be needed for a larger reliable movement set. |
| 4 | Existing crossing model cannot accept live tracker interactions | All 29,564 set01 interaction rows correctly have unavailable crossing predictions. The annotation-trained model is not validated on tracker features. | 2–4 weeks for identity/feature alignment, a tracker-compatible training dataset/model, validation threshold selection, and held-out-set evaluation. Requires additional training and likely annotation mapping. |
| 5 | PIE dataset validation has unresolved frame-set mismatches and incomplete ego telemetry | The errors can affect dataset scope assumptions and validation; 47 visual mismatch samples remain pending. | 1–3 days to confirm CSV contract and review the 50-frame set06 interval plus remaining samples. No model retraining required unless data corrections change the training set. |

### Three improvements with greatest expected benefit

1. **Build a small, curated evaluation benchmark first.** Review representative frames and short track sequences for pedestrians and all claimed vehicle classes; report box and identity metrics separately. This establishes whether retraining is needed and supplies defensible evidence.
2. **Make movement outputs conservative and measurable.** Calibrate on more recording groups, reserve held-out groups, report per-class scores, and mark low-confidence/short/gappy histories unknown. Current data suggest this could improve demonstration reliability without retraining YOLO.
3. **Create a tracker-compatible crossing-intention dataset and evaluation.** Derive tracker histories on training sets, align them to annotations without assuming IDs correspond, preserve the one-second future target and leakage safeguards, train only on train groups, select the threshold on validation, and report a held-out recording-set result. This is necessary before any live crossing probability claim.

### Work by prerequisite

- **Can be addressed without model retraining:** Resolve/interpret CSV frame-set discrepancies; finish the visual review; add source/model/config hashes and software versions; evaluate confidence thresholds; measure tracker identity quality; improve movement-state calibration/unknown handling; test geometry sensitivity; improve runtime/model reuse if needed.
- **Requires additional model training if pursuing the capability:** A PIE-fine-tuned detector requires training only if the pretrained baseline underperforms the chosen benchmark. A live tracker-feature crossing model requires training on tracker-derived features and labels; the saved annotation-feature Random Forest should not be repurposed by substituting guessed inputs.
- **Requires new annotations or ground truth:** Exhaustive detector precision/recall; track identity metrics; reliable pedestrian movement labels across held-out routes; a tracker-to-PIE identity crosswalk; and any collision/near-miss or safety-risk evaluation.
- **PIE dataset limitations:** Only six recording sets limit independent split groups; object annotations are selective rather than exhaustive; the demonstrated vehicle interval has only car annotations; `bike` class semantics are ambiguous; two ego telemetry files are incomplete; and no camera calibration, collision labels, or validated risk ground truth are present.

### Essential before the final demonstration

- Clearly state that the pedestrian/vehicle detector quality has not been measured with a valid exhaustive benchmark and that the demonstrated checkpoint is pretrained YOLO11s.
- Present movement labels as experimental image-space estimates; do not claim that camera-motion compensation yields real-world velocity.
- State that crossing probability is unavailable for live tracker interactions (0/29,564 in the saved run); do not show annotation-model test metrics as live-video performance.
- Describe the 150-pixel interaction threshold and separation as image-space descriptors only. Do not claim collision prediction, physical TTC, calibrated distance, or safety-risk levels.
- Identify the 60-second run's source interval, 0.412 real-time factor, and that only one recording has been demonstrated end-to-end.
- Prefer the updated render-v2, while acknowledging dense-scene label overlap and the limited visual review.

### Reasonable to defer

- Detector fine-tuning, unless the measured baseline justifies it.
- Broad tracker reconfiguration before identity metrics show a problem and isolate its cause.
- Real-world speed, TTC, collision, or risk classification until calibration and ground truth exist.
- Full-dataset inference and performance optimization if the project demonstration is explicitly an offline prototype.
- Dashboard integration and probability display from live tracker records until feature compatibility is established.

## Evidence index

- [Dataset validation report](../dataset_validation/validation_report.md)
- [Pedestrian detection preparation report](../pie_detection_yolo/preparation_report.md)
- [YOLO training arguments](../../runs/detect/train/args.yaml)
- [Camera-motion experiment report](../trajectory_extraction/camera_motion_calibrated_set01_video_0001_60s/camera_motion_experiment_report.md)
- [Vehicle metrics](../vehicle_tracking/vehicle_metrics.json) and [vehicle report](../vehicle_tracking/VEHICLE_TRACKING_REPORT.md)
- [Crossing-intention test metrics](../crossing_intention/test_metrics.json) and [test report](../crossing_intention/test_report.md)
- [Interaction metrics](../interaction_analysis/interaction_metrics.json) and [interaction report](../interaction_analysis/INTERACTION_ANALYSIS_REPORT.md)
- [Integration metrics](../integrated_analysis/integration_metrics.json)
- [Original video-run metrics](../video_runs/set01_video_0001_runner_60s/processing_metrics.json)
- [Updated render-v2 metrics](../video_runs/set01_video_0001_render_v2/render_metrics.json)
- [Visual spot-check report](../visual_spot_check/visual_spot_check_report.md)
- [Test suite](../../tests/)
