# Crossing-Intention and Interaction Integration

## Result and scope

This integration associates only saved Random Forest prediction records with interaction observations when recording identity, an explicit pedestrian-ID crosswalk, timestamp, and causal prediction age all validate. It does not train, tune, or run the model on incompatible tracker features.

- Interactions: 29,564 across 2,543 pedestrian–vehicle pairs.
- Interactions with valid crossing predictions: 0.
- Interactions with unavailable crossing predictions: 29,564.
- Association coverage: 0.0.
- Model inference performed: false.

## Model and feature compatibility

The saved model loaded as `sklearn.pipeline.Pipeline` with `sklearn.ensemble._forest.RandomForestClassifier` (300 estimators). The saved preprocessing configuration covers all 21 metadata features. The frozen Random Forest threshold remains `0.4019572450275602`, read from the existing validation metrics; no threshold was tuned here.

Tracker feature compatibility: `False`. The interaction schema lacks 21 of the model's input feature columns, including PIE action/look/gesture attributes. Remaining model inputs were defined on annotated PIE boxes and annotation histories at `critical_point`; they are not semantically equivalent to YOLO/BoT-SORT boxes, pair-conditioned interaction rows, or tracker-local histories. The engine therefore does not call `predict` or `predict_proba` and does not fill missing model features.

The training configuration places set01 in the training split, while the persisted prediction records are validation set06 and held-out test set04. They do not provide set01 predictions. Set04 predictions are used only as saved records and are never used to change the model or threshold.

The model artifact was serialized with scikit-learn version(s) 1.9.1; the active unpickler emitted compatibility warnings if versions differ. Those warnings are recorded in `integration_metrics.json`.

## Association rules

A model prediction is matched by `(set_id, video_id, mapped prediction track ID)` to an interaction pedestrian only through an explicit track-mapping CSV. Numeric IDs from the PIE annotation/model output and YOLO/BoT-SORT tracker are never assumed equivalent. A prediction can be attached only at or after its `prediction_frame`, and only through the 30-frame target horizon (age 0–30). Probability, binary prediction, and threshold are taken from the saved RF output and checked against the frozen threshold.

No crosswalk was available for this run, and all stored RF outputs belong to set06 or set04 rather than the set01/video_0001 interaction clip. Each integrated observation therefore contains a null probability/prediction with an explicit unavailable status; no annotation track is guessed.

Per-observation closest separation is computed causally from interaction rows up to that frame, within the current uninterrupted valid segment. It does not expose a future pair minimum to earlier observations. Pair summaries report the interaction engine's observed min/median pixel separation.

## Results and limitations

- Unique interaction pedestrians: 173.
- Matched pedestrians with a valid intention prediction: 0.
- Unavailable reasons: `{"no_identity_mapping": 29564}`.
- Timestamp/frame inconsistencies in interaction input: 0.

The intention target remains the existing label definition: at least one visible PIE `crossing` action in the next 30 frames after the annotation `critical_point`; this is not collision ground truth. The model was trained on annotation-derived samples, not validated on tracker-derived features. Image-space separation and movement are not meters or physical velocity. No collision-risk score or Low/Medium/High classification is created.

The Random Forest was loaded from the existing artifact without retraining. No model, feature, threshold, interaction, or previous experiment artifact was modified.

## Integrated pair summary

| Recording | Pedestrian | Vehicle | Observations | Min/median separation (px) | Predictions available | Coverage | Warnings |
|---|---|---|---:|---|---:|---:|---|
| set01/video_0001 | 1 | car/1 | 29 | 636.6/666.6 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 1 | car/10 | 1 | 1417.5/1417.5 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 1 | car/12 | 2 | 588.1/593.3 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 1 | car/14 | 10 | 444.5/469.4 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 1 | car/16 | 1 | 636.4/636.4 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 1 | car/17 | 3 | 1053.3/1057.1 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 1 | car/2 | 29 | 0.0/172.0 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 1 | car/3 | 26 | 1070.2/1231.4 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 1 | car/4 | 22 | 508.0/515.4 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 1 | car/5 | 6 | 226.1/230.4 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 1 | truck/12 | 2 | 575.3/576.9 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 1 | truck/4 | 4 | 587.1/609.4 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 102 | car/1 | 6 | 641.9/649.5 | 0 | 0.0% | first_pair_observation;frame_gap_exceeded;no_identity_mapping |
| set01/video_0001 | 102 | car/107 | 1 | 993.5/993.5 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 102 | car/118 | 1 | 1139.1/1139.1 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 102 | car/120 | 3 | 1179.4/1181.1 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 102 | car/126 | 1 | 619.6/619.6 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 102 | car/130 | 6 | 1217.8/1222.3 | 0 | 0.0% | first_pair_observation;frame_gap_exceeded;no_identity_mapping |
| set01/video_0001 | 102 | car/134 | 1 | 992.7/992.7 | 0 | 0.0% | first_pair_observation;no_identity_mapping |
| set01/video_0001 | 102 | car/145 | 1 | 1243.6/1243.6 | 0 | 0.0% | first_pair_observation;no_identity_mapping |

Showing 20 of 2543 pairs.
