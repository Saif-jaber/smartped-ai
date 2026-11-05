# PIE Annotation Semantics Investigation

**Investigation date:** 2026-11-02  
**Scope:** Annotation meaning, CSV/XML frame coverage, three reported discrepancies, and suitability for SmartPed AI dataset preparation.  
**Boundary:** No original dataset files were changed, annotations corrected, or models trained.

## Executive findings

**The PIE annotated-frame CSV is a list of frames containing at least one visible pedestrian annotation. It is not a list of every frame containing any spatial annotation, and it is not limited to frames in an intention-estimation experiment.**

This interpretation is stated by the official annotation interface implementation: when it generates a CSV, `get_annotated_frame_numbers()` iterates only `pedestrian` tracks, excludes boxes with `outside=1`, and takes the unique frame indices from the remaining boxes. The official annotation README separately says that “annotated” image extraction uses those CSV frame lists. The local CSV contains a declared count followed by frame indices; the first value after the video name is the count, not a frame index.

The local CSV frame sets for all three mismatch clips exactly equal their XML **visible-pedestrian** frame sets. The 52 reported XML-only indices are explained by:

- one visible **crosswalk-only** frame in set02;
- one visible **crosswalk** frame while the pedestrian track is marked **outside** in set03; and
- a 50-frame **vehicle-only** interval in set06 while the pedestrian is outside the image.

Therefore, these 52 indices are **expected differences between two annotation scopes**, not evidence of a video/XML temporal offset or a need to edit the data. The existing validation report flags them because the local validator compares the CSV against frames with any visible spatial class.

**Recommendation:** Proceed with preparing a *pedestrian-target* detection dataset from visible pedestrian XML boxes, subject to the partial-annotation safeguards below. Do not use the three CSV/XML differences as reasons to correct or discard source annotations. Do not treat unannotated people, non-CSV frames, or other objects as reliable detector negatives.

## Evidence and confidence

### Confirmed by the official PIE documentation and interface

The official [annotation README](https://github.com/aras62/PIE/blob/master/annotations/README.md) defines the spatial object classes, label meanings, per-track intention attributes, and the relationship between action and intention. It directs users to the dataset interface for processing details.

The official [`utilities/pie_data.py`](https://github.com/aras62/PIE/blob/master/utilities/pie_data.py) implements the CSV generation rule in `get_annotated_frame_numbers()`:

1. Iterate the spatial XML tracks.
2. Keep tracks whose label is `pedestrian`.
3. Skip boxes whose `outside` flag is `1`.
4. Collect unique frame indices from the remaining boxes.
5. Write the number of collected frames followed by those frame indices.

The same interface's `extract_and_save_images(extract_frame_type='annotated')` uses that pedestrian-frame list. The [official PIE dataset page](https://data.nvision2.eecs.yorku.ca/PIE_dataset/) describes spatial boxes for pedestrians and vehicles **that interact with the driver**, plus infrastructure. This is not a guarantee of exhaustive labeling of every visible road user.

The official interface is the strongest evidence for the CSV's intended semantics. The local data provenance/version was not independently authenticated with a checksum; however, the CSVs in each of the three reported clips exactly follow the documented pedestrian-visible-frame rule.

### Confirmed in the local repository and files

- The local validator's `parse_spatial_xml()` collects visible frame indices across **all** spatial classes; `run_pipeline()` compares that set to the CSV. This is why non-pedestrian-only frames become `annotation_csv_frame_mismatch` errors. See [parsing.py](../../smartped_ai/data/parsing.py), [pipeline.py](../../smartped_ai/data/pipeline.py), and the [validation report](./validation_report.md).
- For each of the three affected clips, a direct comparison found the CSV frame set exactly equal to the XML's visible pedestrian frame set.
- The affected source videos decode at the corresponding zero-based frame indices. Spot-check images are available under the earlier [visual spot-check output](../visual_spot_check/visual_spot_check_report.md).
- The existing [technical analysis](../../SMARTPED_TECHNICAL_ANALYSIS.md) correctly notes that PIE annotations are interaction-focused and that action labels and intention targets are distinct. Its earlier sample-only XML/CSV comparison should not be generalized to all object classes.

## Annotation sources and intended scope

| Source | Contents | Appropriate interpretation |
|---|---|---|
| `annotations/setXX/video_XXXX_annt.xml` | Per-frame tracks and bounding boxes for pedestrians, vehicles, traffic lights, signs, crosswalks, and transit stations; box-level attributes; `outside`, `occluded`, and keyframe flags. | Primary source for spatial detection/tracking targets. Use the object class and visibility state to select the task's boxes. |
| `annotations/setXX/setXX_annotated_frames.csv` | Per video, a declared count and the frame indices generated from **visible pedestrian boxes** (`outside != 1`). | Convenient index of frames containing at least one visible annotated pedestrian. It is not an all-class frame index, a negative-frame list, or an intention-window list. |
| `annotations_attributes/setXX/video_XXXX_attributes.xml` | One record per pedestrian track, including observed crossing outcome, human-aggregated intention probability, event/experiment frame references, and context/demographic attributes. | Track-level supervision and event metadata. Join to spatial XML by pedestrian track ID; do not treat these as independent per-frame labels. |
| `annotations_vehicle/setXX/video_XXXX_obd.xml` | Per-video-frame ego-vehicle/sensor measurements (including speed, position, heading, and inertial fields). | Ego-motion context only. It is not the ground-truth speed or trajectory of every pedestrian or other vehicle. |
| `PIE_clips/setXX/video_XXXX.mp4` | Moving-vehicle camera footage, divided into clips and grouped by set. | Image source. XML frame indices refer to frames within the matching clip; use the corresponding clip and frame index. |

The official README describes six sets and 53 clips; clips within one set are continuous parts of a recording. Keep a whole set in one split to reduce route/recording leakage.

## CSV and spatial-XML relationship

### What the CSV does and does not mean

- **Does it list every frame with any spatial annotation?** No. It is generated from pedestrian tracks only. A frame with only a vehicle, crosswalk, sign, traffic light, or transit station is not included by that generator.
- **Does it list only frames relevant to pedestrian intention estimation?** No. The generator does not filter by `intention_prob`, `critical_point`, `exp_start_point`, crossing outcome, or an intention experiment window. It lists frames with a visible pedestrian annotation.
- **Does it list selected pedestrian frames?** It lists the unique frames containing a pedestrian XML box not marked `outside=1`. A fully or partially occluded pedestrian can still be included if the spatial box is not marked outside.
- **What is its relationship with spatial XML?** It is a derived pedestrian-frame index for the spatial annotations, not a redundant index of every class's visible boxes. For all three mismatching clips, the local CSV equals the XML visible-pedestrian frame set exactly.

### The three reported cases

| Case | Local XML evidence | CSV evidence | Interpretation |
|---|---|---|---|
| `set02/video_0002`, frame **2105** | One visible `crosswalk` box (`2_2_31c`); no pedestrian box at this frame. Frame 2106 has a visible pedestrian box and is in the CSV. | Frame 2105 is absent; the clip has 8,325 CSV indices and 8,325 visible-pedestrian frame indices. | Expected CSV exclusion of a crosswalk-only frame. This is not a CSV omission under the official generation rule. |
| `set03/video_0005`, frame **3339** | A crosswalk box remains visible. Pedestrian `3_5_397` and four traffic-light tracks have `outside=1`; there is no visible pedestrian box. Frame 3338 has a visible pedestrian box; frame 3340 marks the crosswalk outside. | Frame 3339 is absent; the clip has 7,635 CSV indices and 7,635 visible-pedestrian frame indices. | Expected exclusion because no pedestrian box is visible at frame 3339. The XML still carries outside-state markers and a visible crosswalk. |
| `set06/video_0002`, frames **5260–5309** | Each of the 50 frames has a visible vehicle box for `6_2_1415v` (type `car`); there are no visible pedestrian boxes in this interval. Pedestrian `6_2_1769` is marked outside at frame 5230 and next has a visible box at frame 5310. | None of frames 5260–5309 is listed; frame 5310 is listed. The clip has 7,444 CSV indices and 7,444 visible-pedestrian frame indices. | Expected omission of a vehicle-only interval from a pedestrian-frame index. It does not show a 50-frame offset. |

The set06 pedestrian metadata records `exp_start_point=5163`, `critical_point=5211`, and `crossing_point=5310`; its `crossing=1` and `intention_prob=0.9166666667`. At frame 5310 the pedestrian box reappears with `cross=crossing`. This illustrates why the intention/event fields are not interchangeable with the CSV: the CSV follows visible pedestrian boxes, while the metadata describes a track-level experiment and later crossing event.

The existing validation error is therefore a **validator scope mismatch**: its all-class `visible_frames` set is compared with a pedestrian-only CSV. No original annotation should be changed to make those sets identical. For future validation, compare the CSV against visible pedestrian frames; if useful, separately report per-class spatial coverage.

## Pedestrian behavior and intention fields

### Frame-level `action`

The spatial XML `action` field describes observed pedestrian behavior per annotated box. The documented classes are `walking` and `standing`. This is a physical/action label, not a prediction of a future road crossing and not a statement of mental intent.

### Frame-level `cross`

The spatial XML `cross` field describes observed crossing action in relation to the ego vehicle:

- `crossing`: crossing the path of the ego vehicle;
- `not-crossing`: not performing that relevant crossing action;
- `crossing-irrelevant`: crossing the road, but not in the ego vehicle's path.

Do not silently merge `crossing-irrelevant` with `not-crossing`: those labels answer different questions. For a binary action classifier, either exclude the irrelevant cases under a documented task definition or retain a separate class.

### Track-level `crossing` outcome

The pedestrian-attributes XML `crossing` field is a track-level observed outcome: `1` crossing, `0` not crossing, and `-1` irrelevant. The official README says “in front of the ego-vehicle”; irrelevant cases include people near the road who are not intending to cross, such as someone waiting for a bus or hailing a taxi.

This track outcome is not the same field/granularity as each spatial box's `cross` label. Keep the two sources and definitions distinct; preserve `-1` rather than treating it as a negative crossing example.

### Track-level `intention_prob`

`intention_prob` is a continuous value in `[0,1]`, described by the official README as the average human response about the pedestrian's intention to cross **prior to the critical point**. There is one estimate per pedestrian track. It is a soft, human-aggregated estimate, not a per-frame action label or proof that the pedestrian later crossed.

The official documentation explicitly separates intention from action: a person may intend to cross but be prevented from crossing, or may not intend to cross and remain by the road. Do not infer intention from `walking`/`standing` or substitute `cross`/`crossing` outcomes for the intention target.

### Event/experiment frame metadata

- `exp_start_point`: start frame of the clip used in the human experiment.
- `critical_point`: last frame of that human-experiment clip.
- `crossing_point`: observed start of crossing; for a person who does not cross, the official README says the last frame minus three is used.

These are per-track frame references, not extra per-frame target labels. For intention experiments, use the official task protocol and only information available by the prediction cutoff. In particular, do not feed a future `crossing_point`, post-cutoff boxes, or future actions into an intention predictor. The `crossing_point` fallback for non-crossing tracks is a protocol convention, not a literal observed crossing event.

## Suitability for SmartPed AI tasks

| Planned task | Recommended source/target | Policy and caveats |
|---|---|---|
| Pedestrian object detection | Spatial XML `pedestrian` boxes on frames where `outside != 1`; use CSV as an index of frames with visible pedestrian annotations, not as the box source. | Suitable for a PIE interaction-target detector. The annotations are not exhaustive for every visible person. Do not assign unannotated visible people as background during training; on evaluation, ignore or manually review detections on them rather than scoring them as false positives. Conventional exhaustive detection metrics need a manually reviewed exhaustive subset or an explicit ignore/unlabeled policy. Exclude outside markers. Treat fully occluded boxes as a separate visibility subset until their geometry policy is established. |
| Pedestrian tracking | Spatial XML pedestrian track IDs, frame indices, boxes, and visibility/occlusion flags. | Suitable for target-track tracking experiments. IDs are local to the PIE set/video namespace, not global identities. Report evaluation on annotated target tracks; gaps, outside markers, occlusion, and unannotated bystanders limit full-scene tracking claims. Keep sets grouped in splits. |
| Crossing-intention prediction | Per-track `intention_prob` as the soft target; the attributes XML supplies `exp_start_point` and `critical_point` for the experiment window. | Keep the target continuous where possible and report calibration/soft-target metrics. If binarizing, predeclare and justify the threshold. Restrict model inputs to pre-cutoff observations; use a documented PIE sequence-generation protocol. Do not relabel observed `crossing` action as intent. Preserve/exclude `crossing=-1` by a documented policy. |
| Crossing-action classification | Per-box spatial XML `cross` labels, aligned with the same pedestrian box/frame. | This is observed action, not intent. Preserve `crossing-irrelevant` as a distinct class or explicitly exclude it for a path-relevant binary task. Evaluate by track and recording set, not as independent random frames. |
| Walking/standing action recognition | Per-box spatial XML `action` labels. | Suitable as a separate frame-level behavior task. Do not interpret these two labels as crossing or intention labels. |

## Annotation limitations and protocol risks

1. **Partial object coverage.** The official dataset page describes pedestrians and vehicles that interact with the driver. The annotations are not a complete census of every visible person/vehicle. CSV-absent frames are not necessarily empty or safe negative samples.
2. **Outside tracks.** `outside=1` means the object has no visible box in that frame. Do not train a visible-object detector on such a box. A track can leave and later reappear; the set06 pedestrian demonstrates an outside marker followed by a later annotated crossing.
3. **Occlusion.** The official README describes pedestrian occlusion levels `0` (none or under 25%), `1` (25–75%), and `2` (over 75%), while other classes use binary occlusion levels. Local XML also carries an `occluded` attribute/flag and pedestrian text attributes such as `occlusion=none/part/full`. Preserve the source fields and do not assume the binary flag and categorical severity are interchangeable. Report performance by visibility policy.
4. **Moving camera.** Video is recorded from an onboard/moving camera. Boxes are image-coordinate annotations at the corresponding frame, not world coordinates. Ego motion, changing perspective, parallax, and object motion all affect pixel trajectories; pixel displacement is not metric pedestrian speed. OBD records describe the ego vehicle, not every road user's motion.
5. **Temporal dependence and leakage.** Consecutive frames and tracks are correlated; clips within a set are continuous. Use set-level splits, not random-frame splits, and avoid using future frames/event metadata as intention-model inputs.
6. **Frame-list semantics.** The CSV enumerates pedestrian-positive annotated frames under the official generator. It is neither an exhaustive frame-level label of “no pedestrians” nor the index of all spatial XML activity.
7. **Visual verification limits.** The generated spot-check report records selected reviewed frames; most individual mismatch frames remain pending. A frame-list scope explanation is confirmed, but no claim is made that every box or track is visually perfect.

## Recommendation: pedestrian detection data preparation

**Proceed, with a target-only policy.** Build pedestrian detection records from visible `pedestrian` boxes in spatial XML and use the CSV only to locate pedestrian-annotated frames. Preserve source files, track IDs, occlusion/outside flags, and frame indices in derived manifests. Keep the three reported differences as an informational all-class-vs-pedestrian scope report, not data corrections or exclusion rules.

Before creating training/evaluation labels, record these safeguards:

- State explicitly that PIE labels selected ego-interacting targets, not exhaustive pedestrians in every image.
- Do not mark unannotated visible people, vehicle-only frames, or CSV-absent frames as reliable pedestrian negatives.
- Exclude `outside=1` boxes from visible detection targets; retain their track state in metadata.
- Define handling for partial/fully occluded annotations and evaluate visibility strata separately.
- For conventional detection precision/recall/mAP, create a small, manually reviewed exhaustive evaluation subset or an ignore policy for unlabeled people. Do not claim standard exhaustive-scene detection quality from PIE's partial annotations alone.
- Keep all clips from a recording set together in train/validation/test.

**Proceeding does not imply training a model now.** The dataset is suitable for a carefully scoped first pedestrian-detection data-preparation step, but not for treating every unannotated image region as confirmed background.

## Remaining uncertainties

- The local CSVs match the official generation rule on all three reported mismatch clips, but the exact downloaded annotation release/provenance was not verified by checksum.
- The documentation establishes the intention score is an average of human responses before `critical_point`; the exact response aggregation, confidence/variance, and a universal probability-to-class threshold are not established by the local files inspected.
- Fully occluded box geometry and the exact policy for any distinction between the CVAT `occluded` flag and PIE's categorical pedestrian `occlusion` attribute should be confirmed before deciding whether such boxes are detection targets.
- The three issue cases establish the scope mismatch; they do not prove every CSV row in all 53 clips is byte-for-byte identical to regenerated output. A later semantic validator can cheaply compare all CSV rows to visible pedestrian-frame sets without extracting video frames.
- No claim is made that every box is geometrically exact, every track ID is globally unique, or all visible pedestrians are annotated.

## Exact files and sources inspected

### Local project files

- `README.md`
- `SMARTPED_TECHNICAL_ANALYSIS.md`
- `configs/pie_dataset.json`
- `smartped_ai/data/parsing.py`
- `smartped_ai/data/pipeline.py`
- `smartped_ai/data/report.py`
- `artifacts/dataset_validation/validation_report.md`
- `artifacts/dataset_validation/validation_issues.json`
- `artifacts/visual_spot_check/sample_manifest.json`
- `artifacts/visual_spot_check/visual_spot_check_report.md`
- `dataset/annotations/set02/set02_annotated_frames.csv`
- `dataset/annotations/set02/video_0002_annt.xml`
- `dataset/annotations/set03/set03_annotated_frames.csv`
- `dataset/annotations/set03/video_0005_annt.xml`
- `dataset/annotations_attributes/set03/video_0005_attributes.xml`
- `dataset/annotations/set06/set06_annotated_frames.csv`
- `dataset/annotations/set06/video_0002_annt.xml`
- `dataset/annotations_attributes/set06/video_0002_attributes.xml`
- `dataset/PIE_clips/set02/video_0002.mp4`
- `dataset/PIE_clips/set03/video_0005.mp4`
- `dataset/PIE_clips/set06/video_0002.mp4`

### Official PIE resources

- [PIE dataset overview](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)
- [PIE annotation documentation](https://github.com/aras62/PIE/blob/master/annotations/README.md)
- [PIE annotation/data interface (`pie_data.py`)](https://github.com/aras62/PIE/blob/master/utilities/pie_data.py)
- [PIE repository README](https://github.com/aras62/PIE/blob/master/README.md)
- [PIE paper (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.pdf)
