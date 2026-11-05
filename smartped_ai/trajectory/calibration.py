from __future__ import annotations

import math
from typing import Any, Iterable

from .models import MOVEMENT_STANDING, MOVEMENT_WALKING

LABELS = (MOVEMENT_STANDING, MOVEMENT_WALKING)
_THRESHOLD_OBJECTIVE = "balanced_accuracy"


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def classification_metrics(
    ground_truth: Iterable[str],
    predictions: Iterable[str],
) -> dict[str, Any]:
    ground_truth_values = list(ground_truth)
    prediction_values = list(predictions)
    if len(ground_truth_values) != len(prediction_values):
        raise ValueError(
            "Ground truth and predictions must have the same number of values."
        )

    matrix = [[0, 0], [0, 0]]
    for truth, prediction in zip(ground_truth_values, prediction_values):
        truth_label = (truth or "").strip().lower()
        prediction_label = (prediction or "").strip().lower()
        if truth_label not in LABELS or prediction_label not in LABELS:
            continue
        matrix[LABELS.index(truth_label)][LABELS.index(prediction_label)] += 1

    total = sum(sum(row) for row in matrix)
    correct = matrix[0][0] + matrix[1][1]
    per_class: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []
    precisions: list[float] = []
    f1_scores: list[float] = []
    supports: dict[str, int] = {}
    predicted_counts: dict[str, int] = {}
    for index, label in enumerate(LABELS):
        true_positive = matrix[index][index]
        false_negative = sum(matrix[index]) - true_positive
        false_positive = sum(row[index] for row in matrix) - true_positive
        support = true_positive + false_negative
        predicted = true_positive + false_positive
        precision = _safe_ratio(true_positive, predicted)
        recall = _safe_ratio(true_positive, support)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else 0.0
            if support > 0 or predicted > 0
            else None
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
        supports[label] = support
        predicted_counts[label] = predicted
        if support > 0:
            recalls.append(recall if recall is not None else 0.0)
            precisions.append(precision if precision is not None else 0.0)
            f1_scores.append(f1 if f1 is not None else 0.0)

    return {
        "labels": list(LABELS),
        "confusion_matrix": {
            "rows": "ground_truth",
            "columns": "prediction",
            "matrix": matrix,
        },
        "evaluated_observations": total,
        "correct_observations": correct,
        "accuracy": _safe_ratio(correct, total),
        "balanced_accuracy": (
            sum(recalls) / len(recalls) if recalls else None
        ),
        "macro_precision": (
            sum(precisions) / len(precisions) if precisions else None
        ),
        "macro_recall": sum(recalls) / len(recalls) if recalls else None,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else None,
        "ground_truth_support": supports,
        "predicted_counts": predicted_counts,
        "per_class": per_class,
    }


def _quality_speed_samples(
    samples: Iterable[tuple[str, float | None]],
) -> list[tuple[str, float]]:
    valid: list[tuple[str, float]] = []
    for label, speed in samples:
        normalized = (label or "").strip().lower()
        if normalized not in LABELS or speed is None:
            continue
        value = float(speed)
        if not math.isfinite(value) or value < 0:
            continue
        valid.append((normalized, value))
    return valid


def _threshold_predictions(
    samples: list[tuple[str, float]],
    threshold: float,
) -> list[str]:
    return [
        MOVEMENT_WALKING if speed >= threshold else MOVEMENT_STANDING
        for _label, speed in samples
    ]


def evaluate_speed_threshold(
    samples: Iterable[tuple[str, float | None]],
    threshold: float,
) -> dict[str, Any]:
    valid = _quality_speed_samples(samples)
    if not valid:
        raise ValueError("No valid labelled speed samples were supplied.")
    labels = [label for label, _speed in valid]
    metrics = classification_metrics(
        labels,
        _threshold_predictions(valid, threshold),
    )
    return {
        "threshold_px_s": threshold,
        "objective": _THRESHOLD_OBJECTIVE,
        "metrics": metrics,
    }


def choose_speed_threshold(
    samples: Iterable[tuple[str, float | None]],
) -> dict[str, Any]:
    valid = _quality_speed_samples(samples)
    if not valid:
        raise ValueError("No valid labelled speed samples were supplied.")
    labels_present = {label for label, _speed in valid}
    missing = [label for label in LABELS if label not in labels_present]
    if missing:
        raise ValueError(
            "Threshold calibration requires both walking and standing samples; "
            f"missing {', '.join(missing)}."
        )

    candidates = sorted({0.0, *(speed for _label, speed in valid)})
    labels = [label for label, _speed in valid]
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        metrics = classification_metrics(
            labels,
            _threshold_predictions(valid, threshold),
        )
        balanced_accuracy = metrics["balanced_accuracy"] or 0.0
        macro_f1 = metrics["macro_f1"] or 0.0
        accuracy = metrics["accuracy"] or 0.0
        key = (balanced_accuracy, macro_f1, accuracy, threshold)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold_px_s": threshold,
                "objective": _THRESHOLD_OBJECTIVE,
                "candidate_count": len(candidates),
                "candidate_range_px_s": [candidates[0], candidates[-1]],
                "metrics": metrics,
            }
    assert best is not None
    return best
