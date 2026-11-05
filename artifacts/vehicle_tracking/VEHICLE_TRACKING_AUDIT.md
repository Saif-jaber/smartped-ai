# PIE Vehicle Detection, Tracking, and Interaction Audit

**Audit scope:** read-only review of the local SmartPed AI project and PIE dataset. No existing code, configuration, weights, or generated artifacts were modified. This report is the only new file. No risk assessment was started.

## Executive summary

The local PIE spatial XML contains **1,682 vehicle track elements and 528,349 vehicle boxes across 53 clips**. Each vehicle box carries a frame index and coordinates/state flags, with a per-box `id` and `type` attribute. The current dataset-ingestion pipeline already parses all spatial object categories and writes normalized records that include vehicles. The YOLO training-data preparation and the video tracking adapter, however, are intentionally pedestrian-only: the parser selects only pedestrian boxes, the generated YOLO dataset has one pedestrian class, and video tracking filters to COCO class 0 (`person`).

The existing `yolo11s.pt` checkpoint exposes COCO vehicle-related classes, so it can produce vehicle detections at inference; the current adapter does not request them, and no vehicle-specific detection accuracy has been established on PIE. PIE’s separate `bicycle` and `bike` annotation types need a reviewed mapping before supervised training or evaluation against COCO classes.

The safest next implementation is an isolated vehicle path that reuses generic parsing, geometry validation, frame synchronization, and evaluation utilities while retaining a separate class-aware tracking state and outputs. Do not mix vehicle observations into the current pedestrian-only stream unless object category is carried through the complete identity and output schema. Spatial pedestrian–vehicle relations can be derived at aligned frames, but without camera calibration they remain image-space relations—not metric distance, physical speed, time-to-collision, collision, or near-miss ground truth.

## Evidence reviewed

- PIE spatial annotation ingestion and normalized record creation: [`parsing.py`](../../smartped_ai/data/parsing.py), [`pipeline.py`](../../smartped_ai/data/pipeline.py), and [`dataset_validation/PIE_ANNOTATION_SEMANTICS.md`](../dataset_validation/PIE_ANNOTATION_SEMANTICS.md).
- Pedestrian-only YOLO preparation: [`prepare_detection.py`](../../smartped_ai/data/prepare_detection.py), [`pie_detection_preparation.json`](../../configs/pie_detection_preparation.json), and [`pie_detection_yolo/data.yaml`](../pie_detection_yolo/data.yaml).
- Tracking and trajectories: [`readers.py`](../../smartped_ai/trajectory/readers.py), [`models.py`](../../smartped_ai/trajectory/models.py), [`pipeline.py`](../../smartped_ai/trajectory/pipeline.py), and [`smartped_botsort.yaml`](../../smartped_botsort.yaml).
- Existing architecture and data limitations: [`SMARTPED_TECHNICAL_ANALYSIS.md`](../../SMARTPED_TECHNICAL_ANALYSIS.md), [`trajectory_extraction.md`](../../docs/trajectory_extraction.md), and [`pie_dataset.json`](../../configs/pie_dataset.json).
- Relevant tests: [`test_dataset_ingestion.py`](../../tests/test_dataset_ingestion.py), [`test_detection_preparation.py`](../../tests/test_detection_preparation.py), [`test_trajectory_extraction.py`](../../tests/test_trajectory_extraction.py), [`test_trajectory_integration.py`](../../tests/test_trajectory_integration.py), and [`test_visual_spot_check.py`](../../tests/test_visual_spot_check.py).

The counts below were independently checked against all 53 local spatial XML files matching `dataset/annotations/set*/*_annt.xml`. Each vehicle track element had one consistent ID across its boxes, and no vehicle box was missing its `id` or `type` attribute in this scan.

## 1. Vehicle annotations in PIE spatial XML

Vehicle annotations are in the same `annotations/setXX/video_XXXX_annt.xml` files as pedestrian and road-context annotations. A vehicle is represented as a track with `label="vehicle"`. Its child `<box>` elements encode observations in individual video frames; each vehicle box has `type` and `id` attributes nested as `<attribute>` children.

The local scan found these PIE `type` values:

| PIE `type` | Unique vehicle tracks | Annotated boxes |
|---|---:|---:|
| `car` | 1,471 | 444,285 |
| `bicycle` | 105 | 31,767 |
| `bike` | 15 | 4,959 |
| `truck` | 50 | 22,969 |
| `train` | 18 | 11,689 |
| `bus` | 23 | 12,680 |
| **Total** | **1,682** | **528,349** |

These are annotation counts, not detector predictions. PIE annotations are interaction-focused and should not be assumed to exhaustively label every visible vehicle in every frame. In particular, an unannotated vehicle cannot safely be treated as a confirmed negative/background example.

Other spatial track labels in the XML include pedestrian, traffic light, sign, crosswalk, and transit station. They are separate object categories and should not be merged into a vehicle detector’s class map.

## 2. Vehicle box, identity, visibility, and frame representation

Observed XML fields are:

- **Frame:** the `frame` attribute on `<box>`, aligned to the corresponding clip’s frame index. Use the exact `setXX` and `video_XXXX` as well as `frame`; frame numbers or track IDs alone are not global identifiers.
- **Box coordinates:** `xtl`, `ytl`, `xbr`, and `ybr` (top-left and bottom-right in the source image coordinate system).
- **Track identity:** child `<attribute name="id">…</attribute>` on each box. The local scan found 1,682 unique `(set, video, id)` vehicle identities and no ID changes across a track’s boxes.
- **Vehicle type:** child `<attribute name="type">…</attribute>` on each box. Each scanned identity retained one type.
- **Outside/visibility state:** `outside="1"` marks an outside/no-current-box state; `outside="0"` boxes have a spatial box. There were **1,910 `outside=1` boxes** and **526,439 `outside=0` boxes**. Outside states are not usable detector targets and should not be converted into zero-sized boxes.
- **Occlusion:** `occluded="1"` is independent of `outside` and occurs with a box; **100,333** vehicle boxes were marked occluded in this scan. Occlusion does not mean “outside” and should have an explicit, documented inclusion policy.
- **Keyframe:** `keyframe="1"` indicates a keyframe box and `keyframe="0"` a non-keyframe/interpolated box. The preparation pipeline currently preserves the flag in metadata; whether to train from all interpolated boxes or a keyframe-only subset is an experiment/design choice, not something to infer from the flag.

The all-class ingestion parser already preserves category, ID, frame, coordinates, `outside`, `occluded`, keyframe status, and attributes in `object_annotations.jsonl.gz`. A sample normalized vehicle record contains `object_category: "vehicle"`, `track_id`, `frame_index`, `bbox_xyxy`, `outside`, `occluded`, `keyframe`, and `attributes: {"type": ..., "id": ...}`.

The existing `read_pedestrian_xml` helper is narrower: it enters a track only when `label == "pedestrian"` and consequently does not return vehicle boxes. Its name and the unit tests reflect this scope.

## 3. Reuse of data preparation

**Reusable components:** PIE dataset configuration and recording-set split validation; all-class normalized ingestion; bounding-box validation and XYXY-to-YOLO normalization; frame extraction; output manifests; metadata and visual review patterns.

**Not reusable unchanged as a vehicle-preparation pipeline:**

- `read_pedestrian_xml` filters to pedestrian tracks.
- `_prepare_video` uses this reader and its frame-selection logic is organized around pedestrian action, crossing, and occlusion transitions.
- YOLO output labels hard-code `class_id = 0`, and generated `data.yaml` declares `nc: 1` and `names: {0: pedestrian}`.
- Current output metadata/reporting and dataset counts are named and summarized as pedestrian data.
- The annotated-frame CSV is a visible-pedestrian frame index, not an all-object frame list. It must not be used to exclude vehicle-only annotated frames.

Recommendation: keep PIE’s normalized all-category annotation representation as the source, then add a separate configurable vehicle-data preparation path that validates geometry, maps reviewed types to stable class IDs, extracts frames containing eligible vehicle boxes, and records every source annotation and exclusion. Retain the existing recording-set groups (train: sets 01/02/03/05; validation: 06; test: 04); do not randomly split frames or windows.

## 4. Vehicle class compatibility with the current YOLO model

The local `yolo11s.pt` checkpoint reports 80 COCO class names. Relevant class IDs are:

| COCO class ID | COCO name | PIE annotation type |
|---:|---|---|
| 1 | `bicycle` | `bicycle` is a possible direct match |
| 2 | `car` | `car` |
| 3 | `motorcycle` | no unambiguous PIE `type` match established by this audit |
| 5 | `bus` | `bus` |
| 6 | `train` | `train` |
| 7 | `truck` | `truck` |

Thus pretrained inference can request several useful vehicle categories without retraining: bicycle, car, bus, train, and truck, and it can also emit motorcycle detections. This confirms model-label availability only; it does **not** establish detection performance on PIE or that every PIE type maps directly to a COCO category.

PIE distinguishes `bicycle` and `bike` in its annotations. The `bike` category must not be silently mapped to COCO `bicycle` or `motorcycle`; review the official PIE type semantics and representative annotated images first, then document a canonical mapping or preserve a separate class. Also decide whether trains are in scope for the intended road-user task.

The current video adapter in `trajectory/readers.py` passes `classes=[0]` to Ultralytics, and its test asserts that setting. Therefore **the existing deployed code path requests only `person`**, despite the broader classes exposed by the weights. Any vehicle inference must be a separate explicit configuration/path; this audit did not run vehicle inference or score the checkpoint.

## 5. Temporal and spatial pedestrian–vehicle matching

Recommended matching keys and levels:

1. **Annotation/prediction-to-ground-truth box matching:** compare boxes only within the same `(set_id, video_id, frame_index, object_category)` and type/class mapping. Use IoU-based one-to-one assignment per frame for detection metrics; keep ignored/outside/ambiguous targets explicit. Do not match detections from a different frame or clip.
2. **Track evaluation:** compare predicted object tracks to PIE IDs independently per class using frame-aligned matches and identity metrics (for example IDF1, HOTA, MOTA, and ID switches), while documenting the target-limited nature of the annotations.
3. **Pedestrian–vehicle association:** join pedestrian and vehicle records by `(set_id, video_id, frame_index)`, then produce candidate pairs using transparent image-space geometry—such as box overlap, normalized separation between pedestrian footpoint and vehicle box, or persistent closeness over consecutive frames. Preserve both track IDs and the frames supporting each pair. A temporal window may be applied only after frame alignment is established.

PIE provides annotated pedestrian and vehicle IDs in the same spatial XML, so annotated tracks can be paired at shared frames. For detector/tracker outputs, add an explicit object category and keep identity scoped by category, set, and video. Pixel overlap or separation can rank image-space candidate interactions; it must not be reported as meters, physical clearance, or real-world time-to-contact without suitable calibration and validation.

## 6. Moving ego-camera limitations

PIE video is captured from an ego vehicle. Consequently, image motion combines object motion, ego-camera translation/rotation, perspective, and parallax. A vehicle box can move in the image even when the object’s world motion is small; equal pixel distances at different image locations/depths do not imply equal real distances.

The per-frame OBD/ego XML includes ego speed and inertial/heading fields. It describes the recording vehicle, not every pedestrian or other vehicle. The local dataset has no camera intrinsics/extrinsics or road-plane homography identified in the inspected files. Ego speed or heading alone cannot convert a pixel offset into road-plane meters or recover another object’s metric velocity.

The existing BoT-SORT YAML selects `gmc_method: sparseOptFlow`; this is tracking association compensation, not metric camera calibration. The trajectory package also contains optical-flow camera-motion estimation for image-space displacement. These methods can reduce some apparent motion or identity problems but are not a world-coordinate reconstruction and cannot validate collision/near-miss labels.

Finally, the local analysis documents no direct collision, near-miss, or risk ground-truth labels in PIE. Any later interaction/conflict output must be introduced as a derived candidate/proxy with an explicit definition and independent human review—not presented as annotated collision truth.

## 7. Tracking and trajectory support without disturbing pedestrians

The existing tracking/trajectory code is partly generic and partly pedestrian-specific:

- `TrackObservation` stores track ID, frame, bounding box, optional confidence/occlusion, set, and video. Its identity is `(set_id, video_id, track_id)` and does **not** contain an object category.
- The PIE XML trajectory reader calls `read_pedestrian_xml`, skips outside boxes, and creates pedestrian observations only.
- The video adapter filters detector outputs to class 0. Its produced observations do not retain the detector class.
- The trajectory feature engine calculates generic image-space displacement for each input track, but the extraction configuration and walking/standing state classifier are pedestrian-oriented. Do not apply pedestrian walking/standing labels to vehicles.
- The current BoT-SORT configuration has one tracker state/configuration and ReID options. The existing code does not establish a tested category-aware shared tracker stream.

To preserve pedestrian behavior, leave the current class-0 model/tracker path and artifacts unchanged. Add an isolated vehicle detection/tracking path with its own class allowlist, output namespace, and tracker state/configuration, or implement a multi-class tracker only after explicitly carrying category through detections, association, track identity, serialization, and evaluation. A joint stream must prevent cross-class association and ID collisions. Generic motion features may be reused on vehicle tracks after class is retained, but pedestrian state labels and interaction labels remain separate concerns.

## 8. Existing test coverage

Existing tests provide useful scaffolding:

- Dataset ingestion tests exercise spatial XML parsing, attributes, and ego telemetry.
- Detection-preparation tests verify the pedestrian reader’s filtering, outside/occlusion handling, geometry normalization, sampling, and one-class preparation behavior.
- Trajectory tests cover PIE XML pedestrian observations, track IDs, gaps, duplicate observations, movement features, and video adapter behavior.
- Visual spot-check tests can parse and render a vehicle annotation and test frame/category alignment.

The current tests do **not** demonstrate vehicle YOLO dataset generation, vehicle-class inference, class-aware tracking, vehicle trajectory identity isolation, or pedestrian–vehicle interaction matching. The video-adapter test specifically asserts the `[0]` class filter. New vehicle functionality should have separate tests rather than weakening that regression contract.

## 9. Recommended implementation approach and required components

1. **Annotation adapter and type audit:** reuse normalized PIE records; preserve `set/video/frame`, ID, type, box, outside, occlusion, and keyframe; reject malformed IDs/geometry and report counts. Resolve `bike` mapping using annotation documentation plus reviewed images before training/evaluation.
2. **Vehicle detection preparation:** define an explicit class map, dataset policy for occluded/interpolated boxes, class-aware YOLO labels, frame selection independent of the pedestrian CSV, and metadata that links every generated target back to its PIE ID and source frame. Keep grouped recording-set splits.
3. **Inference adapter:** allow configured COCO classes while keeping pedestrian-only inference as the default. Preserve predicted class ID/name, confidence, box, frame, and source identity. Clearly mark detections as model outputs, distinct from PIE ground truth.
4. **Isolated tracking:** keep vehicle tracker state/output separate from the current pedestrian path, or implement a category-aware shared tracker with tested association and namespaced identity. Support gap/occlusion accounting without inventing observations.
5. **Evaluation:** report per-class box and track metrics, class distributions, eligible annotated targets, coverage, ID switches, and limitations from incomplete annotation. Do not count unannotated visible vehicles as certain false positives/background.
6. **Interaction candidates:** synchronize tracked pedestrian and vehicle observations by set/video/frame, calculate only explicitly named image-space geometric/persistence features, and retain both IDs. Do not call candidates collision or near-miss ground truth.

## 10. Expected outputs

For a future implementation, recommended reproducible outputs under a dedicated `artifacts/vehicle_tracking/` subdirectory:

- vehicle dataset-generation configuration and reviewed PIE-to-detector class map;
- split manifest grouped by recording set;
- vehicle annotation metadata with source track/frame, type, box, visibility, occlusion, keyframe, and inclusion/exclusion reason;
- class-aware YOLO dataset configuration, sampled images/labels, and visual QA manifest;
- per-frame vehicle detections (class, confidence, box) and separately serialized vehicle tracks (category, track ID, frame, box, confidence);
- per-class detection/tracking metrics and coverage/exclusion diagnostics;
- synchronized pedestrian–vehicle candidate-pair records with both source IDs and image-space feature definitions;
- readable reports that label model predictions, annotations, and any derived interaction heuristic distinctly.

These are recommendations only; no such files were created by this audit.

## 11. Proposed tests

- Parse synthetic XML containing pedestrian, multiple vehicle types, outside and occluded boxes; assert category, type, ID, coordinates, frame, keyframe, and flags are preserved.
- Verify vehicle parsing includes only `label="vehicle"` while existing pedestrian parsing and outputs remain unchanged.
- Test missing/mixed IDs, mixed types within a track, malformed frames, invalid geometry, and outside boxes; require explicit diagnostics and no invented observations.
- Test a documented class map, including a deliberate unresolved/unsupported PIE type policy; ensure no silent `bike` remapping.
- Test vehicle YOLO conversion, frame selection including vehicle-only frames absent from the pedestrian CSV, grouped split assignments, and traceability from output labels to source annotations.
- Test inference class filters and class-name/confidence preservation; verify the person-only default still requests class 0.
- Test independent vehicle/pedestrian tracker state or class-aware association, identity namespacing, and regression invariance of pedestrian-only results.
- Test per-class detection/tracking matching on same-frame boxes, one-to-one matching, occlusion/gaps, class mismatches, and track ID switches.
- Test pedestrian–vehicle pairing at exact frame alignment, persistence across missing observations, and deterministic image-space features; assert no metric-distance or collision label is synthesized.
- Test output schemas and reports distinguish PIE annotations, YOLO detections, tracker-generated IDs, and derived interaction candidates.

## Audit conclusion

PIE contains a substantial, directly usable set of classed vehicle tracks. The current ingestion layer already understands the all-class XML format, while the existing detector-preparation and tracking adapters intentionally do not process vehicles. The pretrained COCO checkpoint has vehicle categories, but local inference is person-only and PIE’s `bike` taxonomy needs clarification. An isolated vehicle path can be built without disturbing pedestrian functionality by preserving class identity, using aligned per-frame evidence, and keeping all motion/interactions explicitly image-space until independently calibrated. PIE alone supplies no collision or near-miss ground truth.

No code or tests were changed or run as part of this audit.
