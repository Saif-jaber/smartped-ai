# Pedestrian–Vehicle Risk Assessment Audit

**Scope:** audit only. No code, configuration, model, or existing artifact was modified. No risk score, TTC, collision label, or risk tier was generated.

## Executive findings

The existing pedestrian and vehicle prediction outputs for `set01/video_0001` cover the same first 60 seconds (frames 0–1799 at 30 FPS). Their integer frame indices and frame-derived timestamps align exactly. A same-frame join is therefore technically straightforward for this clip. An exploratory join found:

- Pedestrian tracker output: **5,007 observations across 173 tracker-local tracks**.
- Vehicle tracker output: **13,226 observations across 429 `(class, tracker ID)` tracks**.
- **1,525 frames** have both types of predicted observation.
- **29,564** same-frame pedestrian/vehicle detection pairs, representing **2,543 observed tracker-pair combinations**.
- For each pedestrian observation, distance from its bounding-box bottom-center to the nearest vehicle box had median **143.2 pixels** (10th percentile **19.0**, 90th percentile **458.1**); zero indicates the footpoint lies in/on a predicted vehicle box. These are exploratory pixel-space diagnostics, not physical distances or risk thresholds.
- Of the 2,543 observed predicted pair combinations, **692 occur in one frame only**, while **251 persist for at least 30 frames**. These counts show why transient pairs and track identity uncertainty must be represented explicitly.

PIE ground-truth spatial boxes also allow frame-aligned pedestrian/vehicle geometry. In the first 60 seconds of this clip, there are **922 visible pedestrian boxes across 8 annotated pedestrian tracks**, **1,300 visible valid vehicle boxes across 10 annotated vehicle tracks**, and **445 frames with both classes present**, yielding **2,693 same-frame annotated object-pair observations** before any spatial filtering. This supports studying annotation alignment and geometric candidate generation, but PIE does **not** annotate collision, near-miss, dangerous interaction, or pair-specific risk outcomes.

**Conclusion:** a system can produce traceable, image-space pedestrian–vehicle interaction candidates and quality/uncertainty indicators from current outputs. Available evidence does not justify validated real-world TTC, collision prediction, or Low/Medium/High risk thresholds. Those claims require calibration and separately reviewed interaction labels.

## Evidence inspected

- Pedestrian YOLO + BoT-SORT trajectory output and report: [`trajectory_features.csv`](../trajectory_extraction/yolo_botsort_set01_video_0001_60s/trajectory_features.csv), [`trajectory_summary.json`](../trajectory_extraction/yolo_botsort_set01_video_0001_60s/trajectory_summary.json), [`trajectory_annotation_comparison.json`](../trajectory_extraction/yolo_botsort_set01_video_0001_60s/trajectory_annotation_comparison.json), and [`trajectory_report.md`](../trajectory_extraction/yolo_botsort_set01_video_0001_60s/trajectory_report.md).
- Camera-motion experiment: [`camera_motion_experiment_report.md`](../trajectory_extraction/camera_motion_calibrated_set01_video_0001_60s/camera_motion_experiment_report.md).
- Vehicle YOLO11s + separate BoT-SORT output and report: [`vehicle_detections.csv`](../vehicle_tracking/vehicle_detections.csv), [`vehicle_trajectories.csv`](../vehicle_tracking/vehicle_trajectories.csv), [`vehicle_metrics.json`](../vehicle_tracking/vehicle_metrics.json), and [`VEHICLE_TRACKING_REPORT.md`](../vehicle_tracking/VEHICLE_TRACKING_REPORT.md).
- Crossing-intention model outputs: [`validation_predictions.csv`](../crossing_intention/validation_predictions.csv), [`test_predictions.csv`](../crossing_intention/test_predictions.csv), [`feature_metadata.json`](../crossing_intention/feature_metadata.json), and [`test_report.md`](../crossing_intention/test_report.md).
- Annotation semantics and moving-camera constraints: [`PIE_ANNOTATION_SEMANTICS.md`](../dataset_validation/PIE_ANNOTATION_SEMANTICS.md).
- Relevant implementations: [`trajectory/models.py`](../../smartped_ai/trajectory/models.py), [`trajectory/features.py`](../../smartped_ai/trajectory/features.py), [`trajectory/camera_motion.py`](../../smartped_ai/trajectory/camera_motion.py), [`intention/pipeline.py`](../../smartped_ai/intention/pipeline.py), and [`vehicle_tracking/pipeline.py`](../../smartped_ai/vehicle_tracking/pipeline.py).

## 1. Same-frame and timestamp association

For observations from model outputs, associate candidates only when the complete recording identity matches:

```text
(set_id, video_id, frame_index)
```

The tracker IDs are separate namespaces. A pedestrian tracker ID must never be compared directly to a vehicle tracker ID or assumed equal to a PIE annotation ID. Keep keys class-aware, e.g. `(set_id, video_id, "pedestrian", ped_track_id)` and `(set_id, video_id, "vehicle", vehicle_class, vehicle_track_id)`.

In the inspected 60-second outputs, both pipelines use frame index `0..1799` and derive timestamp as `frame_index / 30`. Across the 29,564 same-frame candidate object pairs, the recorded timestamps differed by **0 seconds**. Frame index should remain the primary key; timestamps are a consistency check. If future sources use timestamps or different frame rates, establish and test a synchronization tolerance rather than rounding arbitrary times together.

The output is asynchronous in observation coverage: the pedestrian file has **5,007 rows**, the vehicle file **13,226**, and only 1,525 frames contain both. A non-detection is not evidence that an object disappeared, moved away, or was safe. Pair generation should operate on present observations and explicitly mark when either side is stale/missing.

## 2. Image-space geometry and relative motion

For each visible pedestrian box `B_p = (x1,y1,x2,y2)`, a practical image-plane anchor is its bottom-center:

```text
p = ((x1 + x2) / 2, y2)
```

For a vehicle box `B_v`, the closest point on its rectangle to `p` is:

```text
q_x = clamp(p_x, v_x1, v_x2)
q_y = clamp(p_y, v_y1, v_y2)
d_px = sqrt((p_x - q_x)^2 + (p_y - q_y)^2)
```

Alternative features include center-to-center displacement, box intersection/IoU, horizontal/vertical offsets, normalized coordinates `(x/W, y/H)`, or footpoint-to-vehicle-box distance. Bounding-box distance is a reproducible screen-space proxy, but boxes are not physical object footprints: bottom-center can be inaccurate under occlusion, truncation, slopes, and detector jitter. A zero distance means geometric overlap/contact of the 2D footpoint with the rectangle, not a physical collision.

At consecutive observations with valid timestamps, derive per-track image-plane velocity from an explicitly selected anchor `z`:

```text
v_px = (z(t2) - z(t1)) / (t2 - t1)
v_rel_px = v_vehicle_px - v_pedestrian_px
r_px = z_vehicle - z_pedestrian
```

The pedestrian trajectory output stores center displacement/velocity in pixels and validity/gap flags. The vehicle trajectory output stores center, component displacements, frame gap, an optional image-space speed, and gap flags. The vehicle trajectory currently does not store component velocity fields; these can be derived from its component displacements and elapsed time, preserving the source intervals and image-space units. The existing outputs do not use a common world coordinate system and their anchor choices should be made consistent before computing relative motion.

An optional **image-space closest-approach-time proxy** can be calculated under a constant-relative-image-velocity assumption:

```text
t_star = - dot(r_px, v_rel_px) / dot(v_rel_px, v_rel_px)
d_star_px = norm(r_px + t_star * v_rel_px)
```

Only a positive finite `t_star` and sufficiently nonzero, valid `v_rel_px` describe a future closest approach in this simplified 2D model. This quantity is in seconds only because image velocity is per second; its geometry is still pixels. It is **not validated TTC**, not time-to-collision, and not a physical collision forecast. It assumes linear motion in the image plane, ignores scale/depth changes, road topology, object extent and articulation, and does not compensate for all ego-camera motion.

## 3. Crossing-intention predictions

The crossing model predicts a binary target: whether any visible `crossing` label occurs in the next 30 frames after a pedestrian’s PIE `critical_point`. It is not a collision or conflict model and is not trained to estimate the chance of colliding with a particular vehicle.

The saved validation predictions cover **set06** and test predictions cover **set04**; neither contains set01. Each sample is tied to a PIE pedestrian track ID and a single prediction frame, with three model rows per sample (majority, Logistic Regression, Random Forest). Thus the stored predictions cannot directly contribute to the current set01/video_0001 60-second experiment. They also are not continuous frame-by-frame intention probabilities.

For a future compatible recording, intention may be included only as a **separate, track-linked contextual feature** if:

1. the pedestrian identity is reliably mapped between the model track and the PIE track or the existing model is validly applied to that pedestrian’s past observations;
2. the output corresponds to a known cutoff and is not carried beyond the task’s prediction horizon as if it were current;
3. its learned probability is treated as probability of the specified crossing-action target, not probability of a collision; and
4. it is not multiplied into a risk probability without interaction-risk labels and calibration.

Track-level `intention_prob`, track-level crossing outcome, future `cross` labels, and `crossing_point` are not real-time model inputs. The existing leakage-safe intention feature metadata excludes these fields and future observations.

## 4. Missing observations, occlusion, gaps, and identity uncertainty

The pedestrian output reports **3,886 missing observations within track spans**, coverage **0.563**, 108 tracks with missing observations, and 44 long-gap observations. The vehicle output reports **881 gaps between observations**, **8,734 missing frame slots**, and **92 gaps exceeding 15 frames**. Neither module interpolates detections into observed rows. Vehicle output uses tracker-estimated IDs; the vehicle report explicitly says they are not guaranteed persistent. The pedestrian comparison likewise establishes tracker IDs are not PIE IDs and its metrics are frame-local, not identity metrics.

Recommended risk-input rules:

- Preserve original observations and attach `last_seen_frame/time`, frame/time gap, detector confidence, visibility/occlusion flags, and track source.
- Do not extrapolate or silently bridge a gap. Mark pair features invalid/stale when either track exceeds a justified maximum gap; test several gap policies rather than declaring one universal value.
- Do not stitch IDs across gaps or assume BoT-SORT IDs are persistent ground truth. A suspected ID switch should break pair history or lower an explicit identity-quality indicator.
- Use only visible, geometrically valid boxes for direct distance. Treat occluded/partial boxes as higher-uncertainty inputs; keep `outside=1` states as absence/status, not zero-size geometry.
- Avoid declaring “low risk” when either object is not detected, annotation is absent, or the pair history is short. Use “insufficient observation”/unknown separately from a low estimate.
- Report uncertainty and coverage with each candidate. A single-frame overlap may be a box error, tracking switch, or occlusion artifact rather than an event.

## 5. Risk tiers: no defensible thresholds yet

No evidence inspected supports numeric low/medium/high boundaries in pixels, pixel speed, proxy time, or model probability. The vehicle and pedestrian output reports are detection/trajectory evaluations, not validation of a risk score. There are no PIE pair-level collision/near-miss labels from which to calibrate thresholds.

The first implementation should therefore emit traceable continuous features and a **non-categorical, unvalidated interaction-candidate status**, plus “insufficient data” when coverage/identity/gaps fail quality checks. It should not label these outputs “Low”, “Medium”, or “High”.

To establish ordinal levels later:

1. Define an operational event/risk rubric with explicit evidence and time interval (for example, reviewer-rated interaction severity, not just bounding-box proximity).
2. Have multiple reviewers label synchronized clips; measure agreement and preserve uncertain/ambiguous labels.
3. Fit/calibrate a simple score or ordinal model only on labeled training recording groups; choose any tier boundaries on separate validation groups.
4. Evaluate once on untouched recording groups, report per-tier confusion/recall, calibration, coverage, and uncertainty.
5. Keep an “unknown/insufficient evidence” state. With only six PIE recording sets and prior use of several groups for other experiments, use a newly labeled split or external data and document grouping; do not treat thousands of correlated frames as independent samples.

Percentile thresholds can describe relative ordering within a defined dataset, but are batch- and route-dependent and do not have safety semantics. They are not a substitute for risk-tier validation.

## 6. What PIE can and cannot evaluate

**Can support:**

- Pedestrian and vehicle frame-level box detection matching by same set/video/frame, class, and one-to-one IoU assignment.
- Track annotation matching / ID continuity evaluation when predicted observations are associated to PIE IDs at matched frames, using explicit matching policies and per-class identity metrics.
- The presence and temporal overlap of PIE-annotated pedestrian and vehicle boxes. In the first minute of the example clip, the 922 pedestrian and 1,300 vehicle visible boxes create 2,693 same-frame annotation-pair observations across 445 frames before spatial filtering.
- Descriptive image-space geometry between annotated pairs, without interpreting pixel values physically.
- Pedestrian action/crossing-label context at annotated frames, with explicit policy for `crossing`, `not-crossing`, and `crossing-irrelevant`.

**Cannot support as ground truth:**

- Collision, near-miss, injury, “unsafe interaction”, or severity classification.
- A pair-specific crossing conflict or confirmed physical contact.
- Metric clearance, relative physical speed, or calibrated TTC.
- Exhaustive false-positive counts for detections not covered by PIE’s selected interaction-focused annotations.
- Reliable identity matching by comparing numeric tracker IDs directly with PIE IDs.

The existing pedestrian annotation comparison shows why identity and completeness matter: on the first-minute overlap it evaluates 922 annotated boxes and reports a 0.497 ground-truth box match rate; frame-local matching is not a persistent tracker-ID mapping. The vehicle first-minute report matches 1,233 of 1,300 eligible annotated boxes (0.9485 annotation coverage, mean matched IoU 0.8726), but does not evaluate vehicle IDF1 or pair-risk validity. Neither metric validates interaction risk.

Spatial `cross` is an observed pedestrian crossing-action label relative to the ego vehicle/path. Track-level `crossing` is an observed outcome. `intention_prob` is human-aggregated intention. These are distinct signals, not collision/near-miss labels.

## 7. Moving ego camera and metric limitations

The moving camera introduces ego translation/rotation, perspective, parallax, and changing scale. A stationary object can move across the image; equal pixel offsets at different depths do not represent equal physical distances. Vehicle and pedestrian image-space trajectories therefore cannot be combined into real-world relative velocity by a simple subtraction.

PIE ego/OBD telemetry records the ego vehicle, not every road user. The inspected camera-motion report states telemetry lacks camera intrinsics, extrinsics, mounting geometry, and a road-plane homography and is not used to convert speed into pixel displacement. The existing optical-flow compensation estimates apparent background motion and can reduce some image drift, but it does not recover depth, road-plane scale, or metric object motion. The existing vehicle trajectories are not camera-compensated.

Accordingly:

- Name units `px`, `normalized image coordinate`, or `px/s`; use terms such as *image-space distance*, *image-plane relative motion*, and *closest-approach-time proxy*.
- Never label `d_px` as meters, `px/s` as physical speed, or `t_star` as validated TTC.
- If metric TTC is a product requirement, obtain and validate suitable camera calibration/road-plane mapping and object-motion estimates against independent measurements, and report a validated uncertainty range.

## 8. Proposed implementation design (future work only)

### Inputs

- Pedestrian and vehicle detection/trajectory rows with source type, class, set/video/frame/timestamp, track ID, box, confidence, visibility/occlusion, gap and observation-quality metadata.
- Image dimensions and frame rate for normalization/time checks.
- Optional matched PIE track IDs and spatial labels for offline analysis only.
- Optional crossing-intention output with its target definition, cutoff, model version, and identity-link confidence.
- A separately curated, synchronized pedestrian–vehicle interaction annotation set for eventual risk-model evaluation.

### Processing stages

1. Validate schemas, coordinate dimensions, split/video identity, frame indices, timestamp consistency, and class/identity namespaces.
2. Align present observations at exact set/video/frame; join timestamps as a check. Retain all source IDs.
3. Generate candidate pairs using same-frame geometry and a documented spatial gate. Treat this gate as candidate retrieval, not as a risk threshold. Consider ground-plane footpoint/vehicle lower box only as image-space features.
4. Maintain pair histories only while both source identities remain credible. Compute image-space offsets, box overlap, valid relative displacement/velocity, pair duration, gaps, occlusion, and confidence. Do not invent missing observations.
5. Attach crossing-intention output only when identity and temporal cutoff are valid. Keep the crossing probability distinct from any future risk score.
6. Emit an evidence bundle per frame/pair: features and units, source predictions, IDs, timestamps, quality flags, optional PIE links, and explicit status such as `candidate_unvalidated` or `insufficient_observation`.
7. Defer ordinal risk levels until independent annotated interactions are available and a grouped validation protocol is defined.

### Evaluation strategy

- Unit tests: exact frame/time join; cross-video isolation; class/ID namespaces; box-distance formulas; relative-motion sign and units; zero relative velocity; past/negative proxy times; missing timestamps; gap invalidation; occlusion; ID switch; confidence/coverage statuses; intention horizon/cutoff and leakage.
- Integration tests: synchronize both existing model outputs and PIE annotations on a small synthetic clip; confirm outputs retain provenance and never emit metric units or a confirmed collision label.
- Perception evaluation: report pedestrian/vehicle detection and track quality separately before any risk evaluation.
- Interaction-candidate review: manually label candidate/non-candidate clips and report reviewer agreement, candidate coverage, and false candidate rate; these do not constitute risk validation.
- Risk evaluation: only after operationally defined human-reviewed outcomes exist. Keep recording sets grouped, tune on validation groups, preserve a held-out test group, and report calibration, discrimination, event recall, false alerts per time, lead time, and coverage with uncertainty.

## Audit conclusion

Existing pedestrian and vehicle predictions share a frame/timestamp basis on the same 60-second clip and permit same-frame image-space candidate pairing. Their current gap statistics and frame-local PIE matching show substantial identity/coverage uncertainty. Crossing-intention predictions exist only at track-specific critical points for set06/set04 and cannot directly supply per-frame set01 risk features. PIE supports object-level detection/tracking and observed crossing-action analysis, but not collision/near-miss validation. Until camera geometry and independent interaction labels are available, report only image-space candidate features and uncertainty; do not report real-world TTC, collision prediction, or Low/Medium/High risk tiers.
