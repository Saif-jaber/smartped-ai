      # SmartPed AI
### Intelligent Pedestrian Detection, Crossing Intention Prediction and Safety Risk Assessment

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
| Object detection | Planned |
| Object tracking | Planned |
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
