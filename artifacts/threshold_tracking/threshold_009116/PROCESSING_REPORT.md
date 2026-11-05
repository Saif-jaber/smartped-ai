# SmartPed AI Video Processing Report

## Run summary

- Input: `C:\Users\vlctus\OneDrive\Desktop\Pattern\project development\dataset\PIE_clips\set01\video_0001.mp4`
- Recording identity: `set01/video_0001`
- Source video: 30.000 FPS, 1920x1080, 18,000 frames (600.00s).
- Processed: 1,800 frames (60.00s).
- Annotated video verification: 1,800/1,800 frames fully decoded, frames, 30.000 FPS, 1920x1080, opens and decodes: True.
- Processing wall time: 253.06s; pipeline throughput: 7.11 source frames/s; real-time factor: 0.237x.
- Detector execution device: `cuda` (Ultralytics selects an available accelerator by default).
- Pedestrian detector: `C:\Users\vlctus\OneDrive\Desktop\Pattern\project development\runs\detect\train-5\weights\best.pt` at confidence 0.091162; vehicle detector: `C:\Users\vlctus\OneDrive\Desktop\Pattern\project development\yolo11s.pt` at confidence 0.250000.

## Detection and tracking

- Pedestrian tracked observations: 3,886; unique tracker IDs: 69.
- Vehicle detections: 13,226; unique class-aware tracks: 429.
- Vehicle classes: `{"bicycle": 204, "bus": 21, "car": 12527, "motorcycle": 3, "truck": 471}`.
- Detection/trajectory gaps were retained as observed; no positions were interpolated.

## Interaction and intention integration

- Interaction pair observations: 22,133; unique track pairs: 1,335.
- Interactions with valid crossing predictions: 0; unavailable: 22,133.
- Intention unavailable reason(s): `{"no_identity_mapping": 22133}`.
- The RF uses annotation-derived PIE features. No tracker-derived RF inference is claimed; unavailable probabilities remain null.
- Pixel distances and displacements are image-space only. No real-world speed, TTC, collision, or risk level is inferred.

## Warnings

- Crossing predictions are unavailable: the saved model expects annotation-derived features and has no verified tracker-ID crosswalk; probabilities remain null.

## Output files

- `annotated_video.mp4`: `C:\Users\vlctus\OneDrive\Desktop\Pattern\project development\artifacts\threshold_tracking\threshold_009116\annotated_video.mp4`
- `pedestrian_detections.csv`: `C:\Users\vlctus\OneDrive\Desktop\Pattern\project development\artifacts\threshold_tracking\threshold_009116\pedestrian_detections.csv`
- `vehicle_detections.csv`: `C:\Users\vlctus\OneDrive\Desktop\Pattern\project development\artifacts\threshold_tracking\threshold_009116\vehicle_detections.csv`
- `pedestrian_trajectories.csv`: `C:\Users\vlctus\OneDrive\Desktop\Pattern\project development\artifacts\threshold_tracking\threshold_009116\pedestrian_trajectories.csv`
- `vehicle_trajectories.csv`: `C:\Users\vlctus\OneDrive\Desktop\Pattern\project development\artifacts\threshold_tracking\threshold_009116\vehicle_trajectories.csv`
- `interaction_observations.csv`: `C:\Users\vlctus\OneDrive\Desktop\Pattern\project development\artifacts\threshold_tracking\threshold_009116\interaction_observations.csv`
- `interaction_summary.csv`: `C:\Users\vlctus\OneDrive\Desktop\Pattern\project development\artifacts\threshold_tracking\threshold_009116\interaction_summary.csv`
- `integrated_interactions.csv`: `C:\Users\vlctus\OneDrive\Desktop\Pattern\project development\artifacts\threshold_tracking\threshold_009116\integrated_interactions.csv`
