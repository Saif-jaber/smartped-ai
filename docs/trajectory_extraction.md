# Pedestrian Trajectory Extraction

This milestone adds a modular, read-only trajectory extraction layer for SmartPed AI. It converts tracked pedestrian detections into per-observation movement features and per-track summaries. It does not run detection or tracking by default, modify the PIE dataset, alter trained weights, change tracker YAML files, or claim real-world speed from pixel measurements.

## Inputs

The extractor accepts:

- Canonical JSONL tracking records, one object per line.
- CSV tracking records, with `bbox_xyxy`/`bbox_x1..y2` or top-left `x/y/w/h` columns.
- PIE spatial XML files (`*_annt.xml`) or a directory containing them.
- Optional video inference through Ultralytics, using existing detector weights and optionally an existing tracker YAML.

Every input uses an exact identity of `(set_id, video_id, track_id)`. Missing track IDs are skipped and reported; they are never created. Duplicate observations for the same identity and frame are rejected rather than merged. Inputs are sorted deterministically, but track identities are not stitched across videos, gaps, or ID switches.

Canonical JSONL record:

```json
{"set_id":"set01","video_id":"video_0001","track_id":"p1","frame_index":0,"timestamp_seconds":0.0,"bbox_xyxy":[10,20,40,90],"confidence":0.91,"occluded":false}
```

`timestamp_seconds` is optional. When absent, time is derived from `frame_index / fps`.

## Usage

Run from the project root:

```powershell
python -m smartped_ai.trajectory.cli --input dataset/annotations/set01/video_0001_annt.xml
```

Use canonical tracking output:

```powershell
python -m smartped_ai.trajectory.cli --input tracks.jsonl --set-id set06 --video-id video_0002
```

Optional video inference with existing assets:

```powershell
python -m smartped_ai.trajectory.cli --input dataset/PIE_clips/set01/video_0001.mp4 --set-id set01 --video-id video_0001 --model yolo11s.pt --tracker-config smartped_botsort.yaml
```

Use `--max-seconds 60` to stop inference after the corresponding video frames without trimming or modifying the source video. Use `--annotations` to compare predictions with PIE spatial XML:

```powershell
python -m smartped_ai.trajectory.cli --input dataset/PIE_clips/set01/video_0001.mp4 --input-format video --set-id set01 --video-id video_0001 --max-seconds 60 --model yolo11s.pt --tracker-config smartped_botsort.yaml --annotations dataset/annotations/set01/video_0001_annt.xml --output-dir artifacts/trajectory_extraction/yolo_botsort_set01_video_0001_60s
```

Video inference requires Ultralytics in the active environment. If it is unavailable, use an exported CSV/JSONL tracking file or PIE XML. The detector and tracker files are read only. `_read_video` streams `YOLO.track()` results and uses the tracker's own IDs; it does not infer tracks from annotated videos or rendered predictions.

Configuration is in `configs/trajectory_extraction.json`:

| Field | Default | Meaning |
|---|---:|---|
| `fps` | 30.0 | Fallback frame rate when timestamps are absent |
| `max_gap_frames` | 15 | Maximum consecutive-observation gap with valid velocity |
| `standing_speed_threshold_px_s` | 10.0 | Image-space walking/standing threshold |
| `smoothing_window_observations` | 9 | Recent valid observations used for state smoothing |
| `smoothing_min_observations` | 3 | Minimum recent observations before majority smoothing |

Override the fallback frame rate for one run with `--fps`.

## Calculations

For a bounding box `(x1, y1, x2, y2)`:

```text
center = ((x1 + x2) / 2, (y1 + y2) / 2)
displacement = center[t] - center[t-1]
velocity = displacement / time_gap
speed = sqrt(velocity_x^2 + velocity_y^2)
direction = atan2(displacement_y, displacement_x) mod 360
```

Image coordinates use y pointing down. Direction degrees are `[0, 360)`: `0` right, `90` down, `180` left, `270` up. Zero displacement has no direction.

Velocity remains valid only when the frame gap is positive, at most `max_gap_frames`, and the time gap is positive. A missing timestamp, a first observation, or a long gap produces `velocity_valid=false` and `smoothed_movement_state=unknown`. Displacement is still recorded when available, but long-gap observations are explicitly flagged and smoothing history is reset. Disappearance is therefore not interpreted as standing.

The raw state is `walking` when speed meets the threshold, otherwise `standing`. The smoothed state is a recent majority vote over valid observations. First observations and invalid intervals remain `unknown`.

## Outputs

The default output directory is `artifacts/trajectory_extraction/`:

- `trajectory_features.csv`: flat table with one row per observation.
- `trajectory_features.jsonl`: nested JSON per observation.
- `trajectory_summary.json`: config, inference metadata when applicable, aggregate counts, exact per-track summaries, missing-observation counts, track durations, and validation issues.
- `trajectory_report.md`: concise generated methodology and result report.
- `trajectory_annotation_comparison.json`: optional prediction-versus-PIE comparison.

CSV columns include identity, frame/timestamp, bounding box, center, width/height, frame/time gaps, displacement, velocity, speed, direction, raw/smoothed state, validity flags, confidence, occlusion, and source path.

Track summaries report observed frames, the expected frame span, missing observations inside that span, and duration in seconds. Missing observations are gaps inside an observed track span; they are not filled or treated as stationary.

Example JSONL reasoning fields:

```json
{"track_id":"p1","frame_index":1,"frame_gap":1,"time_gap_seconds":0.0333,"displacement_px":{"x":10.0,"y":0.0,"magnitude":10.0},"velocity_px_s":{"x":300.0,"y":0.0,"magnitude":300.0},"direction_deg":0.0,"direction_label":"right","raw_movement_state":"walking","smoothed_movement_state":"walking","velocity_valid":true,"gap_exceeded":false,"velocity_invalid_reason":null}
```

## Prediction versus PIE comparison

When `--annotations` is supplied, the module compares predictions against visible PIE pedestrian boxes for overlapping frames:

- **Predictions:** YOLO + BoT-SORT tracker outputs with tracker-local IDs.
- **Ground truth:** PIE spatial XML boxes and action attributes. PIE annotations cover selected tracked pedestrians and are not exhaustive scene ground truth.
- Boxes are matched per frame one-to-one by descending IoU at `--iou-threshold` (default `0.5`).
- The comparison reports ground-truth box match rate, prediction match rate on annotated frames (not precision), frame match rate, mean IoU, matched/unmatched counts, track coverage, and walking/standing agreement for matched boxes.
- Tracker IDs and PIE IDs use different identity spaces and are not compared by value. This is not MOTA/IDF1.
- Predictions on frames without visible PIE pedestrian annotations are excluded from box matching; they may depict unannotated people.
- PIE action labels and velocity-derived walking/standing states are different measurement sources, so agreement is diagnostic rather than a benchmark score.

## Camera-motion compensation and threshold calibration

The separate experiment in `smartped_ai.trajectory.experiment` estimates background optical flow from video frames, masking pedestrian boxes. It uses forward/backward-checked Lucas-Kanade features, takes a local median flow near each pedestrian, and falls back to a robust global affine estimate when there are too few local samples. Estimated camera displacement is subtracted from pedestrian image-space displacement before velocity and movement-state classification. If motion cannot be estimated, the original displacement is retained and the feature records why compensation was unavailable.

Run the configured experiment with the existing YOLO + BoT-SORT CSV; this does not rerun or retrain the detector or alter tracker settings:

```powershell
python -m smartped_ai.trajectory.experiment --config configs/camera_motion_experiment.json
```

The experiment calibrates the walking/standing threshold and local-flow radius from PIE `walking`/`standing` annotations on `set03/video_0001`, `set05/video_0001`, and `set05/video_0002`. It evaluates the original threshold, a separately calibrated uncorrected baseline, and the compensated method on the same saved detections from `set01/video_0001`, using the same frame-local IoU matches. These are distinct recordings, but all four recordings are in the PIE train partition; this is recording-disjoint calibration/evaluation, not a held-out official test-split result.

The PIE ego-vehicle XML includes GPS/OBD speed, heading, and inertial fields, but does not provide camera intrinsics, extrinsics, mounting geometry, or a road-plane homography. It is therefore useful as context, not as a direct source of pixel compensation or metric pedestrian velocity.

### Reproduced 60-second result

The configured 60-second evaluation selected a local-flow radius of `220 px`, a compensated threshold of `35.36 px/s`, and a threshold of `90.74 px/s` for the uncorrected calibrated baseline. The original configured threshold is `10 px/s`. Metrics below use the 448 matched observations with both a PIE action label and a valid walking/standing prediction. Confusion-matrix rows are ground truth `[standing, walking]`; columns are prediction `[standing, walking]`.

| Method | Agreement / accuracy | Balanced accuracy | Macro F1 | Confusion matrix |
|---|---:|---:|---:|---|
| Original baseline | 0.219 | 0.500 | 0.179 | `[[0, 350], [0, 98]]` |
| Calibrated baseline | 0.326 | 0.558 | 0.319 | `[[51, 299], [3, 95]]` |
| Camera-motion compensated | 0.286 | 0.543 | 0.269 | `[[30, 320], [0, 98]]` |

| Method | Standing precision / recall / F1 | Walking precision / recall / F1 |
|---|---|---|
| Original baseline | undefined / 0.000 / 0.000 | 0.219 / 1.000 / 0.359 |
| Calibrated baseline | 0.944 / 0.146 / 0.252 | 0.241 / 0.969 / 0.386 |
| Camera-motion compensated | 1.000 / 0.086 / 0.158 | 0.234 / 1.000 / 0.380 |

Compensation improves agreement over the original threshold on this evaluation, but the calibrated uncorrected baseline performs better than the compensated method. The compensation estimate was available for `4,999/4,999` evaluated observation intervals (4,992 local-flow and 7 global-fallback estimates); this availability is not evidence that the estimate represents true camera or pedestrian world motion.

### Observation-coverage diagnostics

The saved detector/tracker output contains `5,007` observations across 173 tracks. The expected count of `8,893` is only the number of frame positions inside those observed tracks' first-to-last spans: `3,886` positions are absent, for a within-span coverage ratio of `0.563`. The gaps comprise 299 events of 2-5 frames (518 missing positions), 72 events of 6-15 frames (603), and 44 events longer than 15 frames (2,765); the longest-gap bucket accounts for `71.2%` of missing in-span positions and occurs across 32 tracks.

These are diagnostics on supplied tracker IDs, not detector recall or full-video coverage. The artifacts cannot determine whether each absent box is due to a detector miss, tracker failure, occlusion, or an identity break. PIE annotations are selective, so unmatched predictions are not automatically false detections. No missing observations are synthesized, interpolated, or classified as standing. The same run's box comparison reports 458 matched of 922 visible PIE boxes (0.497 ground-truth box match rate), but this is not exhaustive-scene detection recall.

The generated comparison and experiment report are in `artifacts/trajectory_extraction/camera_motion_calibrated_set01_video_0001_60s/`, including `camera_motion_experiment_report.md`, `camera_motion_experiment.json`, the method-specific comparison JSON files, and compensated features.

## Limitations

- Pixel velocity is not real-world speed.
- Perspective and camera motion can create pixel movement for stationary people.
- The walking/standing threshold is a heuristic that needs camera-specific calibration.
- Track quality depends on the upstream tracker; ID switches and missed detections are not corrected.
- No interpolation, gap bridging, or identity re-association is performed.
- PIE spatial annotations cover selected tracked pedestrians, not every person in the scene.
- Downstream crossing-intention and risk-assessment components must treat unknown states explicitly.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests cover constant motion, displacement/velocity/direction, image-axis direction labels, short versus long gaps, no false standing after a gap, stationary smoothing, exact identity preservation across videos, optical-flow translation recovery and compensation fallback, threshold metrics and calibration, disjoint calibration/evaluation recordings, gap diagnostics without synthesizing observations, CSV/JSONL/PIE XML inputs, missing-ID rejection, duplicate rejection, CSV/JSONL/summary/report export, missing-observation/duration summaries, mocked YOLO tracker outputs and frame limiting, and prediction-versus-PIE comparison with explicit ground-truth/prediction labels.
