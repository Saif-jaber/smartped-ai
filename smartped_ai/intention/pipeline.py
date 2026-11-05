from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..data.prepare_detection import PedestrianBox, read_pedestrian_xml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "crossing_intention.json"
_SPLITS = ("train", "validation", "test")
_LABELS = (0, 1)
_MISSING = "__missing__"

IDENTIFIER_COLUMNS = (
    "set_id",
    "video_id",
    "track_id",
    "split",
    "prediction_frame",
    "target_start_frame",
    "target_end_frame",
)
NUMERIC_FEATURE_COLUMNS = (
    "center_x_norm",
    "center_y_norm",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_aspect_ratio",
    "foot_x_norm",
    "foot_y_norm",
    "history_observation_count",
    "history_coverage_ratio",
    "max_history_gap_frames",
    "history_motion_pair_count",
    "mean_dx_norm_per_frame",
    "mean_dy_norm_per_frame",
    "mean_speed_norm_per_frame",
    "last_dx_norm_per_frame",
    "last_dy_norm_per_frame",
    "last_speed_norm_per_frame",
    "occluded_at_prediction",
)
CATEGORICAL_FEATURE_COLUMNS = (
    "action_at_prediction",
    "look_at_prediction",
    "gesture_at_prediction",
)
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
NUMERIC_FEATURE_INDICES = tuple(
    FEATURE_COLUMNS.index(name) for name in NUMERIC_FEATURE_COLUMNS
)
CATEGORICAL_FEATURE_INDICES = tuple(
    FEATURE_COLUMNS.index(name) for name in CATEGORICAL_FEATURE_COLUMNS
)
DATASET_COLUMNS = IDENTIFIER_COLUMNS + FEATURE_COLUMNS + ("target_crossing",)
FORBIDDEN_FEATURE_TOKENS = (
    "cross",
    "critical_point",
    "exp_start_point",
    "intention_prob",
    "crossing_point",
    "target",
    "label",
)


@dataclass(frozen=True)
class CrossingIntentionSample:
    set_id: str
    video_id: str
    track_id: str
    split: str
    prediction_frame: int
    target_start_frame: int
    target_end_frame: int
    target_crossing: int
    features: dict[str, int | float | str | None]

    def to_record(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "video_id": self.video_id,
            "track_id": self.track_id,
            "split": self.split,
            "prediction_frame": self.prediction_frame,
            "target_start_frame": self.target_start_frame,
            "target_end_frame": self.target_end_frame,
            **self.features,
            "target_crossing": self.target_crossing,
        }


def load_config(path: Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read crossing-intention config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Crossing-intention config must be a JSON object.")

    fps = _positive_int(raw.get("fps"), "fps")
    history_frames = _positive_int(
        raw.get("history_window_frames"), "history_window_frames"
    )
    horizon_frames = _positive_int(
        raw.get("prediction_horizon_frames"), "prediction_horizon_frames"
    )
    max_gap = _positive_int(
        raw.get("max_motion_gap_frames"), "max_motion_gap_frames"
    )
    random_state = raw.get("random_state")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise ValueError("random_state must be an integer.")

    assignments = raw.get("split_assignments")
    if not isinstance(assignments, dict):
        raise ValueError("split_assignments must be an object.")
    normalized: dict[str, list[str]] = {}
    seen: set[str] = set()
    for split in _SPLITS:
        values = assignments.get(split)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError(f"Split {split!r} must contain set-name strings.")
        for set_id in values:
            if set_id in seen:
                raise ValueError(f"Recording set {set_id!r} is assigned more than once.")
            seen.add(set_id)
        normalized[split] = list(values)

    expected = raw.get("expected_sample_counts")
    if not isinstance(expected, dict) or set(expected) != set(_SPLITS):
        raise ValueError("expected_sample_counts must define train, validation, and test.")
    for split in _SPLITS:
        values = expected[split]
        if (
            not isinstance(values, dict)
            or any(not isinstance(values.get(key), int) for key in ("total", "positive", "negative"))
            or values["total"] != values["positive"] + values["negative"]
        ):
            raise ValueError(f"Invalid expected sample counts for {split!r}.")

    dataset_root = _resolve_path(path, raw.get("dataset_root"), "dataset_root")
    output_dir = _resolve_path(path, raw.get("output_dir"), "output_dir")
    models = raw.get("models")
    if not isinstance(models, dict):
        raise ValueError("models must be an object.")
    return {
        "config_path": path.resolve(),
        "dataset_root": dataset_root,
        "output_dir": output_dir,
        "fps": fps,
        "history_window_frames": history_frames,
        "prediction_horizon_frames": horizon_frames,
        "max_motion_gap_frames": max_gap,
        "random_state": random_state,
        "split_assignments": normalized,
        "expected_sample_counts": expected,
        "models": models,
    }


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _resolve_path(config_path: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string.")
    path = Path(value)
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    return path.resolve()


def split_for_set(set_id: str, assignments: dict[str, list[str]]) -> str:
    matches = [split for split, sets in assignments.items() if set_id in sets]
    if len(matches) != 1:
        raise ValueError(
            f"Recording set {set_id!r} must occur in exactly one configured split."
        )
    return matches[0]


def _read_track_attributes(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Pedestrian attribute XML does not exist: {path}")
    attributes: dict[str, dict[str, str]] = {}
    try:
        for _, element in ET.iterparse(path, events=("end",)):
            if element.tag.rsplit("}", 1)[-1] != "pedestrian":
                continue
            track_id = (element.get("id") or "").strip()
            if not track_id:
                raise ValueError(f"Pedestrian attribute record has no ID in {path}.")
            if track_id in attributes:
                raise ValueError(
                    f"Duplicate pedestrian attribute ID {track_id!r} in {path}."
                )
            attributes[track_id] = dict(element.attrib)
            element.clear()
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse pedestrian attributes from {path}: {exc}") from exc
    return attributes


def _group_track_boxes(
    boxes: Iterable[PedestrianBox],
) -> dict[str, dict[int, PedestrianBox]]:
    tracks: dict[str, dict[int, PedestrianBox]] = defaultdict(dict)
    for box in boxes:
        if not box.track_id:
            continue
        if box.frame_index in tracks[box.track_id]:
            raise ValueError(
                f"Duplicate spatial annotation for track {box.track_id!r} "
                f"at frame {box.frame_index}."
            )
        tracks[box.track_id][box.frame_index] = box
    return tracks


def _is_visible(box: PedestrianBox | None) -> bool:
    if box is None or box.outside:
        return False
    x1, y1, x2, y2 = box.bbox_xyxy
    return (
        x1 is not None
        and y1 is not None
        and x2 is not None
        and y2 is not None
        and all(math.isfinite(value) for value in (x1, y1, x2, y2))
        and x1 < x2
        and y1 < y2
    )


def construct_target(
    attributes: dict[str, str],
    track_boxes: dict[int, PedestrianBox],
    *,
    horizon_frames: int,
) -> tuple[int | None, str | None]:
    """Construct the audited future-action target without using it as a feature."""
    outcome = attributes.get("crossing")
    if outcome == "-1":
        return None, "crossing_irrelevant"
    if outcome not in {"0", "1"}:
        return None, "invalid_crossing_outcome"
    try:
        experiment_start = int(attributes["exp_start_point"])
        cutoff = int(attributes["critical_point"])
        crossing_point = int(attributes["crossing_point"])
    except (KeyError, TypeError, ValueError):
        return None, "missing_or_invalid_event_points"
    if experiment_start < 0 or cutoff < experiment_start:
        return None, "invalid_event_order"

    if any(
        _is_visible(track_boxes.get(frame_index))
        and track_boxes[frame_index].attributes.get("cross") == "crossing"
        for frame_index in range(experiment_start, cutoff + 1)
    ):
        return None, "already_crossing_by_cutoff"
    if outcome == "1" and crossing_point <= cutoff:
        return None, "positive_event_not_after_cutoff"

    future_labels: list[str] = []
    for frame_index in range(cutoff + 1, cutoff + horizon_frames + 1):
        box = track_boxes.get(frame_index)
        if not _is_visible(box):
            return None, "incomplete_or_ambiguous_horizon"
        label = box.attributes.get("cross")
        if label not in {"crossing", "not-crossing"}:
            return None, "incomplete_or_ambiguous_horizon"
        future_labels.append(label)
    target = int("crossing" in future_labels)
    return target, None


def _clean_category(value: str | None) -> str:
    if value is None or value.strip().lower() in {
        "",
        "undefined",
        "__undefined__",
        "n/a",
    }:
        return _MISSING
    return value.strip().lower()


def build_features(
    track_boxes: dict[int, PedestrianBox],
    *,
    prediction_frame: int,
    image_width: int,
    image_height: int,
    history_window_frames: int,
    max_motion_gap_frames: int,
) -> dict[str, int | float | str | None]:
    prediction_box = track_boxes.get(prediction_frame)
    history_start = max(0, prediction_frame - history_window_frames)
    history = [
        track_boxes[frame_index]
        for frame_index in sorted(track_boxes)
        if history_start <= frame_index <= prediction_frame
        and _is_visible(track_boxes[frame_index])
    ]
    valid_pairs: list[tuple[float, float]] = []
    for previous, current in zip(history, history[1:]):
        previous_center = _center(previous)
        current_center = _center(current)
        frame_gap = current.frame_index - previous.frame_index
        if frame_gap <= 0 or frame_gap > max_motion_gap_frames:
            continue
        valid_pairs.append(
            (
                (current_center[0] - previous_center[0]) / image_width / frame_gap,
                (current_center[1] - previous_center[1]) / image_height / frame_gap,
            )
        )
    speeds = [math.hypot(dx, dy) for dx, dy in valid_pairs]
    history_frames = [box.frame_index for box in history]
    gaps = [
        current - previous
        for previous, current in zip(history_frames, history_frames[1:])
    ]
    features: dict[str, int | float | str | None] = {
        "center_x_norm": None,
        "center_y_norm": None,
        "bbox_width_norm": None,
        "bbox_height_norm": None,
        "bbox_aspect_ratio": None,
        "foot_x_norm": None,
        "foot_y_norm": None,
        "history_observation_count": len(history),
        "history_coverage_ratio": len(history) / (history_window_frames + 1),
        "max_history_gap_frames": max(gaps, default=0),
        "history_motion_pair_count": len(valid_pairs),
        "mean_dx_norm_per_frame": _mean([pair[0] for pair in valid_pairs]),
        "mean_dy_norm_per_frame": _mean([pair[1] for pair in valid_pairs]),
        "mean_speed_norm_per_frame": _mean(speeds),
        "last_dx_norm_per_frame": valid_pairs[-1][0] if valid_pairs else None,
        "last_dy_norm_per_frame": valid_pairs[-1][1] if valid_pairs else None,
        "last_speed_norm_per_frame": speeds[-1] if speeds else None,
        "occluded_at_prediction": None,
        "action_at_prediction": _MISSING,
        "look_at_prediction": _MISSING,
        "gesture_at_prediction": _MISSING,
    }
    if _is_visible(prediction_box):
        assert prediction_box is not None
        x1, y1, x2, y2 = (
            float(value) for value in prediction_box.bbox_xyxy if value is not None
        )
        width = x2 - x1
        height = y2 - y1
        features.update(
            {
                "center_x_norm": ((x1 + x2) / 2) / image_width,
                "center_y_norm": ((y1 + y2) / 2) / image_height,
                "bbox_width_norm": width / image_width,
                "bbox_height_norm": height / image_height,
                "bbox_aspect_ratio": width / height,
                "foot_x_norm": ((x1 + x2) / 2) / image_width,
                "foot_y_norm": y2 / image_height,
                "occluded_at_prediction": (
                    int(prediction_box.occluded_raw == "1")
                    if prediction_box.occluded_raw in {"0", "1"}
                    else None
                ),
                "action_at_prediction": _clean_category(
                    prediction_box.attributes.get("action")
                ),
                "look_at_prediction": _clean_category(
                    prediction_box.attributes.get("look")
                ),
                "gesture_at_prediction": _clean_category(
                    prediction_box.attributes.get("gesture")
                ),
            }
        )
    return features


def _center(box: PedestrianBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box.bbox_xyxy
    if x1 is None or y1 is None or x2 is None or y2 is None:
        raise ValueError("Cannot calculate a center for an invalid pedestrian box.")
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _create_sample(
    *,
    set_id: str,
    video_id: str,
    split: str,
    track_id: str,
    attributes: dict[str, str],
    track_boxes: dict[int, PedestrianBox],
    image_width: int,
    image_height: int,
    config: dict[str, Any],
) -> tuple[CrossingIntentionSample | None, str | None]:
    try:
        cutoff = int(attributes["critical_point"])
    except (KeyError, TypeError, ValueError):
        return None, "missing_or_invalid_event_points"
    target, reason = construct_target(
        attributes,
        track_boxes,
        horizon_frames=config["prediction_horizon_frames"],
    )
    if target is None:
        return None, reason
    features = build_features(
        track_boxes,
        prediction_frame=cutoff,
        image_width=image_width,
        image_height=image_height,
        history_window_frames=config["history_window_frames"],
        max_motion_gap_frames=config["max_motion_gap_frames"],
    )
    return (
        CrossingIntentionSample(
            set_id=set_id,
            video_id=video_id,
            track_id=track_id,
            split=split,
            prediction_frame=cutoff,
            target_start_frame=cutoff + 1,
            target_end_frame=cutoff + config["prediction_horizon_frames"],
            target_crossing=target,
            features=features,
        ),
        None,
    )


def generate_dataset(
    config: dict[str, Any],
) -> tuple[list[CrossingIntentionSample], dict[str, Any]]:
    root: Path = config["dataset_root"]
    assignments: dict[str, list[str]] = config["split_assignments"]
    sample_list: list[CrossingIntentionSample] = []
    exclusions: Counter[str] = Counter()
    videos: list[dict[str, Any]] = []
    discovered_sets: set[str] = set()

    attributes_root = root / "annotations_attributes"
    for attribute_path in sorted(attributes_root.glob("set*/*_attributes.xml")):
        set_id = attribute_path.parent.name
        video_id = attribute_path.name[: -len("_attributes.xml")]
        split = split_for_set(set_id, assignments)
        discovered_sets.add(set_id)
        spatial_path = root / "annotations" / set_id / f"{video_id}_annt.xml"
        if not spatial_path.is_file():
            raise FileNotFoundError(f"Spatial pedestrian XML does not exist: {spatial_path}")
        image_width, image_height, boxes = read_pedestrian_xml(spatial_path)
        track_boxes = _group_track_boxes(boxes)
        attributes_by_track = _read_track_attributes(attribute_path)
        video_samples = 0
        for track_id, attributes in attributes_by_track.items():
            sample, reason = _create_sample(
                set_id=set_id,
                video_id=video_id,
                split=split,
                track_id=track_id,
                attributes=attributes,
                track_boxes=track_boxes.get(track_id, {}),
                image_width=image_width,
                image_height=image_height,
                config=config,
            )
            if sample is None:
                exclusions[reason or "unknown"] += 1
                continue
            sample_list.append(sample)
            video_samples += 1
        videos.append(
            {
                "set_id": set_id,
                "video_id": video_id,
                "split": split,
                "sample_count": video_samples,
            }
        )
    configured_sets = {set_id for values in assignments.values() for set_id in values}
    if discovered_sets != configured_sets:
        missing = sorted(configured_sets - discovered_sets)
        unexpected = sorted(discovered_sets - configured_sets)
        raise ValueError(
            f"Recording-set assignments do not match discovered data; "
            f"missing={missing}, unexpected={unexpected}."
        )
    if not sample_list:
        raise ValueError("No crossing-intention samples were generated.")
    if len({(s.set_id, s.video_id, s.track_id) for s in sample_list}) != len(sample_list):
        raise ValueError("Generated more than one prediction sample for a pedestrian track.")
    _validate_feature_contract()
    return sample_list, {
        "exclusions": dict(sorted(exclusions.items())),
        "videos": videos,
        "recording_sets": sorted(discovered_sets),
    }


def _validate_feature_contract() -> None:
    forbidden = [
        name
        for name in FEATURE_COLUMNS
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Forbidden target/leakage fields in feature schema: {forbidden}")


def _counts(samples: Iterable[CrossingIntentionSample]) -> dict[str, Any]:
    grouped: dict[str, list[CrossingIntentionSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.split].append(sample)
    result: dict[str, Any] = {}
    for split in _SPLITS:
        values = grouped.get(split, [])
        class_counts = Counter(sample.target_crossing for sample in values)
        result[split] = {
            "sample_count": len(values),
            "class_distribution": {
                "negative": class_counts[0],
                "positive": class_counts[1],
            },
            "unique_tracks": len(
                {(sample.set_id, sample.video_id, sample.track_id) for sample in values}
            ),
            "recording_sets": sorted({sample.set_id for sample in values}),
        }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_model_pipeline(kind: str, config: dict[str, Any]) -> Any:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError(
            "Crossing-intention baselines require scikit-learn. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(NUMERIC_FEATURE_INDICES),
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURE_INDICES),
            ),
        ],
        remainder="drop",
    )
    model_config = config["models"][kind]
    if kind == "logistic_regression":
        estimator = LogisticRegression(
            class_weight=model_config["class_weight"],
            max_iter=model_config["max_iter"],
            random_state=config["random_state"],
        )
    elif kind == "random_forest":
        estimator = RandomForestClassifier(
            class_weight=model_config["class_weight"],
            n_estimators=model_config["n_estimators"],
            min_samples_leaf=model_config["min_samples_leaf"],
            random_state=config["random_state"],
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("preprocess", preprocess), ("classifier", estimator)])


def _metrics(
    truth: list[int],
    probabilities: list[float],
    threshold: float,
) -> dict[str, Any]:
    try:
        from sklearn.metrics import (
            average_precision_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )
    except ImportError as exc:
        raise RuntimeError("Crossing-intention evaluation requires scikit-learn.") from exc
    predictions = [int(probability >= threshold) for probability in probabilities]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=list(_LABELS), zero_division=0
    )
    return {
        "threshold": threshold,
        "sample_count": len(truth),
        "positive_count": sum(truth),
        "negative_count": len(truth) - sum(truth),
        "precision": float(precision[1]),
        "recall": float(recall[1]),
        "f1": float(f1[1]),
        "pr_auc_average_precision": float(average_precision_score(truth, probabilities)),
        "confusion_matrix": {
            "rows": "actual [negative, positive]",
            "columns": "predicted [negative, positive]",
            "matrix": confusion_matrix(truth, predictions, labels=list(_LABELS)).tolist(),
        },
        "per_class": {
            "negative": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "positive": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
    }


def _select_validation_threshold(truth: list[int], probabilities: list[float]) -> float:
    candidates = sorted({0.0, 1.0, *probabilities})
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    for threshold in candidates:
        metrics = _metrics(truth, probabilities, threshold)
        positive = metrics["per_class"]["positive"]
        key = (metrics["f1"], positive["precision"], threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def _feature_matrix(samples: list[CrossingIntentionSample]) -> list[list[Any]]:
    return [
        [sample.features[name] for name in FEATURE_COLUMNS]
        for sample in samples
    ]


def _render_report(payload: dict[str, Any]) -> str:
    train = payload["split_manifest"]["train"]
    validation = payload["split_manifest"]["validation"]
    lines = [
        "# PIE crossing prediction baseline",
        "",
        "This report evaluates a one-second-ahead PIE crossing-action target. It is not a direct measure of mental intention.",
        "",
        "## Protocol",
        "",
        "- One example per pedestrian track, prediction moment at `critical_point`.",
        "- Positive: any visible `crossing` frame in the next 30 frames.",
        "- Negative: all next 30 frames visible and labeled `not-crossing`.",
        "- Irrelevant tracks, incomplete/ambiguous horizons, and tracks already crossing by the cutoff are excluded.",
        "- Features use only the prediction-frame box and visible track observations at or before the cutoff.",
        "- Preprocessing and model fitting use training set groups only; thresholds are selected on validation set06.",
        "- The final test group set04 has not been evaluated.",
        "- Image-space movement features are not real-world velocity.",
        "",
        "## Sample counts",
        "",
        "| Split | Recording sets | Samples | Negative | Positive | Unique tracks |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in _SPLITS:
        values = payload["split_manifest"][split]
        distribution = values["class_distribution"]
        lines.append(
            f"| {split} | {', '.join(values['recording_sets'])} | "
            f"{values['sample_count']} | {distribution['negative']} | "
            f"{distribution['positive']} | {values['unique_tracks']} |"
        )
    lines.extend(
        [
            "",
            "Only six independent recording-set groups are available. Validation is one set and test is one set, so measured generalization is uncertain. Validation metrics and tuned thresholds are model-development results, not unbiased final estimates.",
            "",
            "## Validation results",
            "",
            f"Validation set06 contains {validation['sample_count']} samples "
            f"({validation['class_distribution']['positive']} positive, "
            f"{validation['class_distribution']['negative']} negative). "
            f"Training uses {train['sample_count']} samples from four recording sets.",
            "",
            "| Model | Threshold (validation F1) | Precision | Recall | F1 | PR-AUC (average precision) | Confusion matrix [TN,FP;FN,TP] |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name, result in payload["validation_metrics"].items():
        lines.append(
            f"| {name} | {result['threshold']:.4f} | "
            f"{result['precision']:.3f} | {result['recall']:.3f} | "
            f"{result['f1']:.3f} | {result['pr_auc_average_precision']:.3f} | "
            f"`{result['confusion_matrix']['matrix']}` |"
        )
    lines.extend(
        [
            "",
            "PR-AUC is reported as average precision. The majority baseline always predicts the training majority class; its score is the training positive prevalence. Logistic Regression and Random Forest use class weighting and validation-only F1 threshold selection.",
            "",
            "## Limitations",
            "",
            "- The target is future observed crossing action in a fixed one-second horizon, not the soft human-aggregated `intention_prob`.",
            "- Complete visible annotation is required throughout the future horizon, so the resulting subset may be selected and does not represent every pedestrian track.",
            "- PIE labels selected ego-interacting targets and is not exhaustive.",
            "- Current/past `action`, `look`, and `gesture` fields can be missing or weakly informative; gesture uses an explicit missing category.",
            "- Only six recording sets are available. No set-level cross-validation or final test evaluation is included in this run.",
            "- Predictions retain set/video/track/frame identifiers for later dashboard linking.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pipeline(config_path: Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    samples, generation = generate_dataset(config)
    counts = _counts(samples)
    for split in _SPLITS:
        expected = config["expected_sample_counts"][split]
        observed = counts[split]
        actual_distribution = observed["class_distribution"]
        if (
            observed["sample_count"] != expected["total"]
            or actual_distribution["positive"] != expected["positive"]
            or actual_distribution["negative"] != expected["negative"]
        ):
            raise ValueError(
                f"Generated {split} sample counts differ from the audited target: "
                f"expected {expected}, observed "
                f"{{'total': {observed['sample_count']}, "
                f"'positive': {actual_distribution['positive']}, "
                f"'negative': {actual_distribution['negative']}}}. "
                "No models were trained."
            )
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    if not train or not validation or {s.target_crossing for s in train} != {0, 1}:
        raise ValueError("Training and validation data must contain both target classes.")

    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = [sample.to_record() for sample in samples]
    _write_csv(output_dir / "feature_dataset.csv", dataset_rows, DATASET_COLUMNS)

    train_x = _feature_matrix(train)
    train_y = [sample.target_crossing for sample in train]
    validation_x = _feature_matrix(validation)
    validation_y = [sample.target_crossing for sample in validation]
    train_counts = Counter(train_y)
    majority_class = min(
        (label for label in _LABELS if train_counts[label] == max(train_counts.values())),
    )
    majority_prob = train_counts[1] / len(train_y)
    majority_probabilities = [majority_prob] * len(validation)
    validation_metrics: dict[str, Any] = {
        "majority_baseline": {
            **_metrics(validation_y, majority_probabilities, 0.5),
            "threshold_selection": "fixed majority class from training distribution",
            "predicted_class": majority_class,
        }
    }
    predictions: list[dict[str, Any]] = []
    for sample, probability in zip(validation, majority_probabilities):
        predictions.append(
            {
                "model": "majority_baseline",
                "set_id": sample.set_id,
                "video_id": sample.video_id,
                "track_id": sample.track_id,
                "prediction_frame": sample.prediction_frame,
                "actual_crossing": sample.target_crossing,
                "crossing_probability": probability,
                "predicted_crossing": majority_class,
                "threshold": 0.5,
            }
        )

    model_files: dict[str, str] = {}
    for model_name in ("logistic_regression", "random_forest"):
        pipeline = _make_model_pipeline(model_name, config)
        pipeline.fit(train_x, train_y)
        classes = list(pipeline.named_steps["classifier"].classes_)
        positive_index = classes.index(1)
        probabilities = pipeline.predict_proba(validation_x)[:, positive_index].tolist()
        threshold = _select_validation_threshold(validation_y, probabilities)
        metrics = _metrics(validation_y, probabilities, threshold)
        metrics["threshold_selection"] = "validation positive-class F1 maximization"
        validation_metrics[model_name] = metrics
        model_path = output_dir / f"{model_name}.pkl"
        with model_path.open("wb") as destination:
            pickle.dump(pipeline, destination, protocol=pickle.HIGHEST_PROTOCOL)
        model_files[model_name] = model_path.name
        for sample, probability in zip(validation, probabilities):
            predictions.append(
                {
                    "model": model_name,
                    "set_id": sample.set_id,
                    "video_id": sample.video_id,
                    "track_id": sample.track_id,
                    "prediction_frame": sample.prediction_frame,
                    "actual_crossing": sample.target_crossing,
                    "crossing_probability": float(probability),
                    "predicted_crossing": int(probability >= threshold),
                    "threshold": threshold,
                }
            )
    split_manifest = {
        split: {
            **counts[split],
            "assigned_recording_sets": config["split_assignments"][split],
        }
        for split in _SPLITS
    }
    feature_metadata = {
        "schema_version": 1,
        "identifiers_not_used_as_features": list(IDENTIFIER_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric_features": {
            name: "Computed from visible spatial boxes at or before critical_point; image-normalized geometry/motion."
            for name in NUMERIC_FEATURE_COLUMNS
        },
        "categorical_features": {
            "action_at_prediction": "PIE frame-level action at critical_point.",
            "look_at_prediction": "PIE frame-level look at critical_point.",
            "gesture_at_prediction": "PIE frame-level gesture at critical_point; undefined values are missing.",
        },
        "target_and_future_fields_excluded": [
            "future spatial cross labels",
            "crossing_point",
            "critical_point",
            "exp_start_point",
            "track-level crossing outcome",
            "intention_prob",
        ],
        "missing_observation_policy": (
            "No interpolation. History count, coverage, and largest observed gap are features; "
            "missing velocity features are imputed using train-fitted medians."
        ),
        "pixel_motion_note": "Motion features are normalized image-space estimates, not real-world velocity.",
    }
    _write_json(output_dir / "feature_metadata.json", feature_metadata)
    config_snapshot = {
        **{
            key: value
            for key, value in config.items()
            if key not in {"dataset_root", "output_dir", "config_path"}
        },
        "dataset_root": config["dataset_root"].as_posix(),
        "output_dir": output_dir.as_posix(),
        "source_config": config["config_path"].as_posix(),
        "target_definition": {
            "prediction_moment": "critical_point",
            "horizon_frames": config["prediction_horizon_frames"],
            "positive": "at least one visible future frame labeled crossing",
            "negative": "all future horizon frames visible and labeled not-crossing",
            "already_crossing_policy": "exclude if a visible crossing label occurs from exp_start_point through critical_point",
            "crossing_irrelevant_policy": "exclude",
        },
    }
    _write_json(output_dir / "generation_config.json", config_snapshot)
    _write_json(
        output_dir / "split_manifest.json",
        {
            "grouping_unit": "recording_set",
            "assignments": config["split_assignments"],
            "counts": split_manifest,
            "exclusions": generation["exclusions"],
            "videos": generation["videos"],
            "test_evaluation": "not performed",
        },
    )
    _write_csv(
        output_dir / "validation_predictions.csv",
        predictions,
        (
            "model",
            "set_id",
            "video_id",
            "track_id",
            "prediction_frame",
            "actual_crossing",
            "crossing_probability",
            "predicted_crossing",
            "threshold",
        ),
    )
    metrics_payload = {
        "schema_version": 1,
        "evaluation_split": "validation",
        "test_evaluation": "not performed",
        "train_only_preprocessing": True,
        "class_weighting": {
            "logistic_regression": config["models"]["logistic_regression"]["class_weight"],
            "random_forest": config["models"]["random_forest"]["class_weight"],
        },
        "threshold_tuning_split": "validation",
        "validation_metrics": validation_metrics,
        "model_files": model_files,
        "majority_baseline": {
            "training_class_counts": {
                "negative": train_counts[0],
                "positive": train_counts[1],
            },
            "predicted_class": majority_class,
            "positive_probability": majority_prob,
        },
    }
    _write_json(output_dir / "model_metrics.json", metrics_payload)
    report_payload = {
        "split_manifest": split_manifest,
        "validation_metrics": validation_metrics,
    }
    (output_dir / "report.md").write_text(
        _render_report(report_payload), encoding="utf-8"
    )
    return {
        "output_dir": output_dir,
        "counts": split_manifest,
        "validation_metrics": validation_metrics,
        "model_files": model_files,
        "generation": generation,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and validate leakage-safe PIE crossing-prediction baselines."
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        result = run_pipeline(args.config)
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"Crossing-intention pipeline failed: {exc}", file=sys.stderr)
        return 2
    print(f"Generated samples: {result['counts']}")
    for name, metrics in result["validation_metrics"].items():
        print(
            f"{name}: precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, "
            f"PR-AUC={metrics['pr_auc_average_precision']:.3f}"
        )
    print(f"Outputs: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
