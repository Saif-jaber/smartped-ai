# PIE crossing prediction baseline

This report evaluates a one-second-ahead PIE crossing-action target. It is not a direct measure of mental intention.

## Protocol

- One example per pedestrian track, prediction moment at `critical_point`.
- Positive: any visible `crossing` frame in the next 30 frames.
- Negative: all next 30 frames visible and labeled `not-crossing`.
- Irrelevant tracks, incomplete/ambiguous horizons, and tracks already crossing by the cutoff are excluded.
- Features use only the prediction-frame box and visible track observations at or before the cutoff.
- Preprocessing and model fitting use training set groups only; thresholds are selected on validation set06.
- The final test group set04 has not been evaluated.
- Image-space movement features are not real-world velocity.

## Sample counts

| Split | Recording sets | Samples | Negative | Positive | Unique tracks |
|---|---|---:|---:|---:|---:|
| train | set01, set02, set03, set05 | 492 | 328 | 164 | 492 |
| validation | set06 | 103 | 74 | 29 | 103 |
| test | set04 | 328 | 238 | 90 | 328 |

Only six independent recording-set groups are available. Validation is one set and test is one set, so measured generalization is uncertain. Validation metrics and tuned thresholds are model-development results, not unbiased final estimates.

## Validation results

Validation set06 contains 103 samples (29 positive, 74 negative). Training uses 492 samples from four recording sets.

| Model | Threshold (validation F1) | Precision | Recall | F1 | PR-AUC (average precision) | Confusion matrix [TN,FP;FN,TP] |
|---|---:|---:|---:|---:|---:|---|
| majority_baseline | 0.5000 | 0.000 | 0.000 | 0.000 | 0.282 | `[[74, 0], [29, 0]]` |
| logistic_regression | 0.3054 | 0.622 | 0.966 | 0.757 | 0.686 | `[[57, 17], [1, 28]]` |
| random_forest | 0.4020 | 0.650 | 0.897 | 0.754 | 0.767 | `[[60, 14], [3, 26]]` |

PR-AUC is reported as average precision. The majority baseline always predicts the training majority class; its score is the training positive prevalence. Logistic Regression and Random Forest use class weighting and validation-only F1 threshold selection.

## Limitations

- The target is future observed crossing action in a fixed one-second horizon, not the soft human-aggregated `intention_prob`.
- Complete visible annotation is required throughout the future horizon, so the resulting subset may be selected and does not represent every pedestrian track.
- PIE labels selected ego-interacting targets and is not exhaustive.
- Current/past `action`, `look`, and `gesture` fields can be missing or weakly informative; gesture uses an explicit missing category.
- Only six recording sets are available. No set-level cross-validation or final test evaluation is included in this run.
- Predictions retain set/video/track/frame identifiers for later dashboard linking.
