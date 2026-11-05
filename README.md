### Intelligent Pedestrian Detection, Crossing Intention Prediction and Safety Risk Assessment

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

`Computer Vision` · `Object Detection` · `Object Tracking` · `Pedestrian Intention Prediction` · `Collision Risk Assessment` · `Intelligent Transportation Systems` · `Deep Learning`

SmartPed AI is an AI-powered pedestrian safety system designed to analyze traffic video footage, detect and track pedestrians and vehicles, predict pedestrian crossing intentions, and assess potential pedestrian-vehicle collision risks.

The project combines computer vision, object tracking, and machine learning to support safer urban environments through intelligent traffic monitoring and early risk identification.

## Project Overview

Pedestrian safety is a major challenge in modern urban environments. Unexpected pedestrian crossings and interactions between pedestrians and moving vehicles can create dangerous situations.

SmartPed AI aims to address this challenge by analyzing video footage and identifying potentially risky pedestrian-vehicle interactions. By combining object detection, tracking, behaviour analysis, and risk assessment, the system aims to provide useful safety insights and visual alerts.

## Key Features

- **Pedestrian Detection:** Identify pedestrians in traffic video footage.
- **Vehicle Detection:** Detect vehicles and monitor their movement.
- **Object Tracking:** Maintain consistent identities for pedestrians and vehicles across video frames.
- **Behaviour Recognition:** Analyze pedestrian behaviour, including walking, standing, and crossing-related actions.
- **Crossing Intention Prediction:** Predict whether a pedestrian is likely to cross the road.
- **Collision Risk Assessment:** Estimate potential pedestrian-vehicle collision risks using movement patterns, trajectories, and time-to-collision calculations.
- **Visual Monitoring Dashboard:** Display detection results, tracking IDs, predictions, risk alerts, and traffic statistics.
- **Risk Visualization:** Highlight potentially dangerous interactions to support human review.

## System Architecture

The proposed system consists of several interconnected components:

1. **Video Input:** Process recorded traffic footage.
2. **Object Detection:** Detect pedestrians and vehicles using a deep learning object detection model.
3. **Object Tracking:** Track detected objects across consecutive frames.
4. **Behaviour Analysis:** Extract movement and behavioural information from tracked objects.
5. **Intention Prediction:** Analyze pedestrian movement sequences to estimate crossing intentions.
6. **Risk Assessment:** Evaluate pedestrian-vehicle interactions and estimate potential collision risks.
7. **Visualization:** Present the results through an interactive monitoring dashboard.

## Proposed Technologies

| Component | Candidate Technologies |
|---|---|
| Programming Language | Python |
| Object Detection | YOLO / RT-DETR |
| Object Tracking | ByteTrack / BoT-SORT |
| Behaviour Recognition | LSTM / GRU / Transformer |
| Computer Vision | OpenCV |
| Machine Learning | PyTorch |
| Backend | FastAPI |
| Frontend | React |

*Final model selection will depend on dataset suitability, annotation quality, and experimental evaluation.*

## Dataset

SmartPed AI uses the **PIE (Pedestrian Intention Estimation) Dataset** as its primary research dataset.

The dataset provides traffic video footage and annotations that support pedestrian behaviour analysis and crossing intention prediction.

Dataset resources:

- [Official PIE Dataset Website](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)
- [PIE Dataset GitHub Repository](https://github.com/aras62/PIE)

The dataset is intended for research and development. Its camera viewpoint and recording conditions will be considered when evaluating the system's applicability to real-world traffic monitoring.

**Note:** The dataset is not included in this repository because of its size and separate access and licensing conditions.

### Dataset ingestion and validation

The first data milestone parses PIE spatial annotations, pedestrian attributes and intention labels, ego-vehicle telemetry, and annotated-frame CSVs. It probes video metadata when `ffprobe` is available and does not extract video frames.

Run from the project root:

```powershell
python -m smartped_ai.data.cli
```

Paths and recording-set split assignments are configured in `configs/pie_dataset.json`. Generated normalized records, manifests, statistics, and the validation report are written under `artifacts/dataset_validation/`, outside the original dataset directory. The default split keeps each `setXX` together; with only six recording groups, validation and test estimates have limited independent samples.

### Visual annotation spot-checking

The read-only PIE frame inspector overlays spatial annotations, IDs, and available pedestrian action/crossing labels on selected video frames. It does not modify or bulk-extract the original dataset. Generate the mismatch and behavior-category examples after running dataset ingestion:

```powershell
python -m smartped_ai.inspection.visual_spot_check --samples
```

Outputs are written to `artifacts/visual_spot_check/`, including rendered frames, a sample manifest, and `visual_spot_check_report.md`. To inspect one frame interactively and save its annotated screenshot:

```powershell
python -m smartped_ai.inspection.visual_spot_check --set set06 --video video_0002 --frame 5260
```

Press `s` to save the screenshot and `q` or Escape to close the window. Use `--no-display` to render a frame without opening a window. The inspector reports XML/CSV membership but leaves visual findings pending until a reviewer records them; it never silently corrects an alignment mismatch.

Record a manual finding in the report and manifest after inspecting a generated sample:

```powershell
python -m smartped_ai.inspection.visual_spot_check --record-review --set set06 --video video_0002 --frame 5260 --review-status uncertain --finding "Vehicle box is plausible; extent needs a closer review." --likely-cause "CSV may list pedestrian frames only, while XML includes vehicle boxes."
```

### PIE pedestrian detection dataset preparation

Prepare a sparse, one-class YOLO dataset from visible pedestrian boxes in the original spatial XML:

```powershell
python -m smartped_ai.data.prepare_detection
```

The configuration is `configs/pie_detection_preparation.json`; it reuses the dataset paths and
recording-set splits in `configs/pie_dataset.json`. The default stride samples frames at a
30-frame cadence within each video and always includes track starts/ends and action, crossing,
or occlusion-state transitions. XML keyframes and interpolated boxes are distinguished in
metadata; set `preserve_keyframes` to `true` to additionally extract every valid keyframe frame.
Use `--frame-stride 60` to override the configured cadence for one run.

Outputs are written to `artifacts/pie_detection_yolo/`. The pipeline retains valid occluded
pedestrian boxes, excludes `outside=1` and invalid geometry from YOLO labels, and records all
pedestrian boxes and their sampling/exclusion status in `annotation_metadata.jsonl.gz`. The
annotated-frame CSV is used only for a visible-pedestrian frame cross-check, not as the box
source. Unannotated people are not treated as background examples. No model is downloaded or
trained, and original videos and annotations are left unchanged.

### Pedestrian trajectory and movement extraction

Extract per-observation image-space movement features and per-track summaries from PIE spatial XML, canonical tracking CSV/JSONL, or an existing video setup:

```powershell
python -m smartped_ai.trajectory.cli --input dataset/annotations/set01/video_0001_annt.xml
```

For an existing video, provide the detector weights and the existing tracker YAML. Video inference requires Ultralytics and is optional. This runs real YOLO + BoT-SORT tracking before trajectory extraction; it does not infer tracks from annotated/rendered video:

```powershell
python -m smartped_ai.trajectory.cli --input dataset/PIE_clips/set01/video_0001.mp4 --input-format video --set-id set01 --video-id video_0001 --max-seconds 60 --model yolo11s.pt --tracker-config smartped_botsort.yaml --annotations dataset/annotations/set01/video_0001_annt.xml --output-dir artifacts/trajectory_extraction/yolo_botsort_set01_video_0001_60s
```

`--max-seconds` stops streaming inference at the requested segment length without trimming or modifying the source video. `--annotations` adds an optional PIE comparison while keeping predictions and ground truth explicitly separated. Configuration is in `configs/trajectory_extraction.json`. Outputs are written as CSV, JSONL, a JSON summary, an optional comparison JSON, and a Markdown report. Track summaries include missing-observation counts and durations. Track identities are preserved exactly as `(set_id, video_id, track_id)`; observations are not interpolated, merged, or re-associated. Long gaps and missing timestamps produce an explicit `unknown` state rather than a false standing result. Distances, velocity, and direction are image-space quantities, not real-world speed. See `docs/trajectory_extraction.md` for the schema, calculations, comparison method, and limitations.

### Vehicle detection and tracking

Run the separate pretrained YOLO11s + BoT-SORT vehicle experiment:

```powershell
python -m smartped_ai.vehicle_tracking --config configs/vehicle_tracking.json
```

The default experiment processes the first 60 seconds of `set01/video_0001`, filters
COCO `bicycle`, `car`, `motorcycle`, `bus`, and `truck`, and uses the independent
`smartped_vehicle_botsort.yaml` tracker config. It does not retrain YOLO or change
pedestrian tracking. PIE `bike` is explicitly unmapped pending a reviewed taxonomy;
PIE `train` is counted but outside the selected COCO classes. Detections, trajectories,
frame-level annotation matches, visual samples, metrics, and a report are saved under
`artifacts/vehicle_tracking/`. PIE annotations and model-generated detections are
reported separately. Motion remains image-space only, and PIE annotations are not
assumed exhaustive. No collision, TTC, or risk scoring is performed.

### End-to-end video processing

Run the existing pedestrian, vehicle, trajectory, interaction, and intention-
integration pipelines on one MP4 and write the annotated video and CSV/JSON outputs
to a new run directory:

```powershell
python -m smartped_ai.video_runner --source dataset/PIE_clips/set01/video_0001.mp4 --duration 60 --output-dir artifacts/video_runs/set01_video_0001_runner_60s
```

To revise only the visual overlays from a prior run without rerunning inference or
changing its CSV outputs, use a separate empty destination:

```powershell
python -m smartped_ai.video_runner --source dataset/PIE_clips/set01/video_0001.mp4 --render-from-existing-run artifacts/video_runs/set01_video_0001_runner_60s --output-dir artifacts/video_runs/set01_video_0001_render_v2
```

`--output-dir` must be new or empty. `--duration` processes the first N seconds;
`--confidence` sets the confidence threshold for both detectors. The runner reuses
the configured YOLO checkpoint and separate pedestrian/vehicle BoT-SORT configs;
it does not retrain models or overwrite prior experiment outputs. Required artifacts
include detection and trajectory CSVs, interaction and integrated CSVs, metrics,
a report, and `annotated_video.mp4`. The saved crossing-intention model is not run
on tracker-derived features when their semantics or pedestrian identity mapping
are incompatible, so probability fields can remain unavailable. Distances and
motion are image-space estimates only; this runner does not infer real-world
velocity, TTC, collision, or risk levels.

### PIE crossing prediction baseline

Train and validate the leakage-safe one-second crossing-action baseline:

```powershell
pip install -r requirements.txt
python -m smartped_ai.intention.pipeline --config configs/crossing_intention.json
```

The pipeline uses one sample per pedestrian track at `critical_point`. It predicts whether any of the next 30 fully annotated frames is labeled `crossing`; ambiguous windows and `crossing-irrelevant` tracks are excluded. Inputs use only current/past normalized boxes, short-history image-space motion, and current-frame action/look/gesture. Models fit preprocessing and classifier parameters on train sets 01/02/03/05; thresholds are selected on validation set06. The final set04 test split is counted and written to the feature dataset but is never passed to a model or evaluated. Expected split counts are enforced before training. Models, validation predictions/metrics, feature metadata, split manifest, and report are written under `artifacts/crossing_intention/`. This predicts observed crossing action in a fixed horizon, not human-aggregated intention probability. See `docs/crossing_intention.md`.

## Risk Assessment

The system aims to estimate potential collision risks by analyzing:

- Pedestrian and vehicle positions.
- Movement direction and speed.
- Object trajectories.
- Pedestrian crossing intentions.
- Time-to-collision (TTC).
- Relative movement between pedestrians and vehicles.

The risk assessment component is intended to identify potentially dangerous interactions and provide alerts for human review. It is not intended to replace human judgment or guarantee collision prevention.

## Project Objectives

- Develop a computer vision pipeline for pedestrian and vehicle detection.
- Track pedestrians and vehicles across video frames.
- Recognize pedestrian behaviours and estimate crossing intentions.
- Develop a trajectory-based pedestrian-vehicle risk assessment method.
- Visualize predictions and potential safety risks.
- Evaluate the system using appropriate detection, prediction, and risk-assessment metrics.

## Development Status

**Currently under development.**

| Component | Status |
|---|---|
| Project setup | In progress |
| Dataset preparation | In progress |
| Pedestrian trajectory extraction | Initial module |
| Object detection | Pretrained inference baseline |
| Object tracking | Initial YOLO + BoT-SORT integration |
| Behaviour recognition | Planned |
| Crossing intention prediction | Planned |
| Collision risk assessment | Planned |
| Monitoring dashboard | Planned |
| Model evaluation | Planned |

## Future Improvements

- Improve prediction accuracy through temporal modelling.
- Evaluate performance across different traffic scenarios.
- Improve trajectory estimation and collision-risk assessment.
- Optimize inference speed for real-time applications.
- Explore deployment in smart-city traffic monitoring environments.

## Research and Intended Use

SmartPed AI is an academic research and development project focused on pedestrian safety, computer vision, and intelligent transportation systems.

The system is designed as a decision-support tool that can help identify potentially risky situations and support human monitoring.

## License

License information will be added once the project's distribution and dataset licensing requirements have been reviewed.

---

**SmartPed AI | Towards Safer and Smarter Urban Mobility.**
