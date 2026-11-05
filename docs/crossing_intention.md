# PIE crossing prediction baseline

The baseline predicts **future observed crossing action**, not a pedestrian's mental state or PIE's human-aggregated `intention_prob`.

## Target and eligibility

There is one sample per pedestrian track, linked by `(set_id, video_id, track_id)`.

- Prediction time is the track's `critical_point`.
- The target horizon is frames `critical_point + 1` through `critical_point + 30`, inclusive (one second at 30 fps).
- Positive means at least one visible future frame is labeled `crossing`.
- Negative means all 30 future frames are visible and labeled `not-crossing`.
- `crossing-irrelevant`, absent/outside/invalid future frames, or other future labels make the horizon ineligible.
- Exclude track-level `crossing=-1`, and tracks with a visible `cross` label of `crossing` from `exp_start_point` through the prediction cutoff.
- For a positive track, exclude a `crossing_point` at or before the cutoff.

`crossing_point`, `critical_point`, `exp_start_point`, track-level crossing outcome, future `cross`, and `intention_prob` are excluded from the feature matrix. Event points and future labels are used only for sample timing/eligibility and target construction. They are never model inputs.

## Features

Only observations with frame indices at or before the prediction cutoff contribute features:

- Current-frame normalized box center, footpoint, width, height, aspect ratio, and occlusion flag.
- Observation count, coverage ratio, largest gap, valid short-gap motion-pair count, and normalized image-space displacement/speed statistics over the preceding 30 frames.
- Current-frame `action`, `look`, and `gesture` categories; undefined values use an explicit missing category.

There is no interpolation. Missing numeric motion statistics stay missing until a median imputer fit on the training data. Motion is image-space, not metric pedestrian velocity. IDs and split labels are retained as metadata for auditing and dashboard linking, but are not model features.

## Models and evaluation protocol

The models are a training-majority baseline, class-weighted Logistic Regression, and class-weighted Random Forest. Numeric imputation/scaling and categorical one-hot encoding are fit inside model pipelines using training groups only. Validation set06 selects each learned model's positive-class threshold by F1; the majority baseline uses a fixed 0.5 threshold and always predicts the training majority class. Validation is reported after this threshold tuning and should not be treated as an unbiased final estimate.

The fixed groups are train sets 01/02/03/05, validation set06, and test set04. The test samples are generated and included in the traceable feature CSV, but are excluded from model fitting, threshold selection, predictions, and metrics. Only six independent set groups exist; validation and test each contain one group.

The checked-in configuration freezes the audited expected class counts and stops training on any mismatch:

| Split | Samples | Negative | Positive |
|---|---:|---:|---:|
| Train | 492 | 328 | 164 |
| Validation | 103 | 74 | 29 |
| Test (not evaluated) | 328 | 238 | 90 |

Run with `python -m smartped_ai.intention.pipeline --config configs/crossing_intention.json`. Outputs go to `artifacts/crossing_intention/`: generated feature data, saved models, feature metadata, generation configuration, split manifest, validation predictions, model metrics, and a generated Markdown report. PR-AUC is reported as average precision. Confusion-matrix rows/columns are negative then positive.
