# Final held-out crossing prediction evaluation

**Set04 was used only for this final evaluation.** The saved Logistic Regression and Random Forest models were loaded as-is; neither was retrained. The exact existing target/feature generator was used, and thresholds were read from the saved validation metrics without tuning on set04.

## Test set

- Eligible samples: 328 (negative 238, positive 90)
- Unique pedestrian tracks: 328
- Recording sets: set04 only
- Prediction coverage: 328/328 eligible samples per model (100%); this is not coverage of all PIE pedestrians/annotations.
- Prediction moment: critical_point; outcome is future visible crossing action during the next 30 frames.
- Image-space movement features are not real-world velocity.

## Validation and final test comparison

Confusion matrices use `[[TN, FP], [FN, TP]]`; PR-AUC is average precision. Thresholds for learned models were chosen on validation set06 and are unchanged here.

| Model | Threshold | Val P/R/F1/AP | Test P/R/F1/AP | Test confusion | Δ F1 | Δ AP |
|---|---:|---|---|---|---:|---:|
| majority_baseline | 0.5000 | 0.000/0.000/0.000/0.282 | 0.000/0.000/0.000/0.274 | `[[238, 0], [90, 0]]` | +0.000 | -0.007 |
| logistic_regression | 0.3054 | 0.622/0.966/0.757/0.686 | 0.509/0.900/0.651/0.661 | `[[160, 78], [9, 81]]` | -0.106 | -0.024 |
| random_forest | 0.4020 | 0.650/0.897/0.754/0.767 | 0.566/0.856/0.681/0.724 | `[[179, 59], [13, 77]]` | -0.072 | -0.043 |

## Interpretation

Validation is set06 alone (103 samples); final test is set04 alone (328 samples). The measured differences reflect performance on these two distinct recording sets and their sample/class composition; with one group in each split, they do not establish why performance changed or estimate variation across all PIE recordings. The majority baseline predicts the training majority class for every sample.

Set04 was not used for fitting, preprocessing, feature selection, model selection, or threshold selection. No YOLO, BoT-SORT, trajectory, camera-motion, feature-definition, or threshold changes were made.
