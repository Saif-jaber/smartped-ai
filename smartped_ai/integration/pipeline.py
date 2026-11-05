from __future__ import annotations

import csv
import json
import math
import pickle
import re
import statistics
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "integration_analysis.json"
_PREDICTION_COLUMNS = (
    "model",
    "set_id",
    "video_id",
    "track_id",
    "prediction_frame",
    "crossing_probability",
    "predicted_crossing",
    "threshold",
)
_INTEGRATED_FIELDS = (
    "set_id",
    "video_id",
    "frame_index",
    "timestamp_seconds",
    "pedestrian_track_id",
    "vehicle_track_id",
    "vehicle_class",
    "distance_pixels",
    "movement_status",
    "movement_valid",
    "movement_invalid_reason",
    "closest_approach_valid_as_of_frame",
    "closest_approach_invalid_reason",
    "closest_approach_frame_index_as_of_frame",
    "closest_approach_timestamp_seconds_as_of_frame",
    "closest_approach_separation_pixels_as_of_frame",
    "crossing_probability",
    "crossing_prediction",
    "prediction_available",
    "prediction_status",
    "prediction_source",
    "prediction_frame",
    "prediction_timestamp_seconds",
    "prediction_age_frames",
    "prediction_age_seconds",
    "prediction_threshold",
    "prediction_identity_mapping_source",
    "interaction_timestamp_status",
    "prediction_timestamp_status",
    "association_validity",
    "data_quality_warnings",
)
_SUMMARY_FIELDS = (
    "set_id",
    "video_id",
    "pedestrian_track_id",
    "vehicle_track_id",
    "vehicle_class",
    "observation_count",
    "minimum_separation_pixels",
    "median_separation_pixels",
    "movement_status",
    "crossing_probability_mean",
    "crossing_probability_minimum",
    "crossing_probability_maximum",
    "crossing_prediction_positive_count",
    "observations_with_valid_intention_prediction",
    "percentage_with_valid_intention_prediction",
    "data_quality_warnings",
)


@dataclass(frozen=True)
class IntegrationConfig:
    interaction_observations_path: Path
    interaction_summary_path: Path
    interaction_run_config_path: Path
    random_forest_model_path: Path
    feature_metadata_path: Path
    model_metrics_path: Path
    generation_config_path: Path
    validation_predictions_path: Path
    test_predictions_path: Path
    track_mapping_path: Path | None
    output_dir: Path
    expected_model_name: str
    max_prediction_age_frames: int
    fps: float
    timestamp_tolerance_seconds: float
    minimum_observations_for_closest_approach: int

    def validate(self) -> None:
        if not self.expected_model_name:
            raise ValueError("expected_model_name must not be empty.")
        if (
            not isinstance(self.max_prediction_age_frames, int)
            or isinstance(self.max_prediction_age_frames, bool)
            or self.max_prediction_age_frames < 0
        ):
            raise ValueError("max_prediction_age_frames must be a non-negative integer.")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and greater than zero.")
        if (
            not math.isfinite(self.timestamp_tolerance_seconds)
            or self.timestamp_tolerance_seconds < 0
        ):
            raise ValueError("timestamp_tolerance_seconds must be finite and non-negative.")
        if (
            not isinstance(self.minimum_observations_for_closest_approach, int)
            or isinstance(self.minimum_observations_for_closest_approach, bool)
            or self.minimum_observations_for_closest_approach < 1
        ):
            raise ValueError(
                "minimum_observations_for_closest_approach must be a positive integer."
            )


def _resolve_path(config_path: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string.")
    path = Path(value)
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    return path.resolve()


def _read_optional_path(config_path: Path, raw: dict[str, Any], name: str) -> Path | None:
    value = raw.get(name)
    if value is None or value == "":
        return None
    return _resolve_path(config_path, value, name)


def load_config(path: Path = _DEFAULT_CONFIG) -> IntegrationConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read integration config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Integration config must be a JSON object.")

    def integer(name: str) -> int:
        value = raw.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer.")
        return value

    fps = raw.get("fps")
    tolerance = raw.get("timestamp_tolerance_seconds")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool):
        raise ValueError("fps must be numeric.")
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise ValueError("timestamp_tolerance_seconds must be numeric.")

    config = IntegrationConfig(
        interaction_observations_path=_resolve_path(
            path, raw.get("interaction_observations_path"), "interaction_observations_path"
        ),
        interaction_summary_path=_resolve_path(
            path, raw.get("interaction_summary_path"), "interaction_summary_path"
        ),
        interaction_run_config_path=_resolve_path(
            path, raw.get("interaction_run_config_path"), "interaction_run_config_path"
        ),
        random_forest_model_path=_resolve_path(
            path, raw.get("random_forest_model_path"), "random_forest_model_path"
        ),
        feature_metadata_path=_resolve_path(
            path, raw.get("feature_metadata_path"), "feature_metadata_path"
        ),
        model_metrics_path=_resolve_path(
            path, raw.get("model_metrics_path"), "model_metrics_path"
        ),
        generation_config_path=_resolve_path(
            path, raw.get("generation_config_path"), "generation_config_path"
        ),
        validation_predictions_path=_resolve_path(
            path, raw.get("validation_predictions_path"), "validation_predictions_path"
        ),
        test_predictions_path=_resolve_path(
            path, raw.get("test_predictions_path"), "test_predictions_path"
        ),
        track_mapping_path=_read_optional_path(path, raw, "track_mapping_path"),
        output_dir=_resolve_path(path, raw.get("output_dir"), "output_dir"),
        expected_model_name=str(raw.get("expected_model_name", "")).strip(),
        max_prediction_age_frames=integer("max_prediction_age_frames"),
        fps=float(fps),
        timestamp_tolerance_seconds=float(tolerance),
        minimum_observations_for_closest_approach=integer(
            "minimum_observations_for_closest_approach"
        ),
    )
    config.validate()
    return config


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return value


def _read_csv(
    path: Path, required_columns: Iterable[str]
) -> tuple[list[dict[str, str]], int]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                if path.stat().st_size == 0:
                    return [], 0
                raise ValueError(f"CSV has no header: {path}")
            missing = set(required_columns) - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV {path} is missing required columns: "
                    f"{', '.join(sorted(missing))}"
                )
            rows = [
                {key: value or "" for key, value in row.items() if key is not None}
                for row in reader
                if row and any(value and value.strip() for value in row.values() if value)
            ]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read CSV {path}: {exc}") from exc
    return rows, len(rows)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_int(value: Any) -> int | None:
    number = _parse_number(value)
    return int(number) if number is not None and number.is_integer() else None


def _parse_interactions(
    rows: list[dict[str, str]],
    *,
    fps: float,
    timestamp_tolerance: float,
) -> tuple[list[dict[str, Any]], int]:
    parsed: dict[tuple[str, str, int, str, str, str], dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        set_id = row["set_id"].strip()
        video_id = row["video_id"].strip()
        frame_index = _parse_int(row["frame_index"])
        if not set_id or not video_id or frame_index is None or frame_index < 0:
            raise ValueError("Interaction row has invalid recording identity or frame.")
        pedestrian_id = row["pedestrian_track_id"].strip()
        vehicle_id = row["vehicle_track_id"].strip()
        vehicle_class = row["vehicle_class"].strip()
        if not pedestrian_id or not vehicle_id or not vehicle_class:
            raise ValueError("Interaction row is missing a pedestrian or vehicle identity.")
        timestamp = _parse_number(row["timestamp_seconds"])
        if timestamp is None or timestamp < 0:
            timestamp_status = "invalid_timestamp"
        else:
            timestamp_status = row.get("timestamp_validation_status", "unknown")
            if timestamp_status == "consistent" and abs(
                timestamp - frame_index / fps
            ) > timestamp_tolerance:
                timestamp_status = "interaction_timestamp_frame_mismatch"
        distance = _parse_number(row["distance_pixels"])
        if distance is None or distance < 0:
            raise ValueError("Interaction row has an invalid pixel separation.")
        record: dict[str, Any] = {
            "set_id": set_id,
            "video_id": video_id,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp,
            "interaction_timestamp_status": timestamp_status,
            "pedestrian_track_id": pedestrian_id,
            "vehicle_track_id": vehicle_id,
            "vehicle_class": vehicle_class,
            "distance_pixels": distance,
            "movement_valid": _as_bool(row.get("movement_valid", False)),
            "movement_status": row.get("movement_status", "indeterminate"),
            "movement_invalid_reason": row.get("movement_invalid_reason", ""),
        }
        key = (set_id, video_id, frame_index, pedestrian_id, vehicle_class, vehicle_id)
        previous = parsed.get(key)
        if previous is not None:
            if previous != record:
                raise ValueError(
                    "Conflicting duplicate interaction row for "
                    f"{set_id}/{video_id}/{frame_index}/{pedestrian_id}/"
                    f"{vehicle_class}/{vehicle_id}."
                )
            duplicates += 1
            continue
        parsed[key] = record
    return sorted(
        parsed.values(),
        key=lambda item: (
            item["set_id"],
            item["video_id"],
            item["pedestrian_track_id"],
            item["vehicle_class"],
            item["vehicle_track_id"],
            item["frame_index"],
        ),
    ), duplicates


def assess_feature_compatibility(
    feature_metadata: dict[str, Any],
    interaction_fields: Iterable[str],
) -> dict[str, Any]:
    feature_columns = feature_metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(
        isinstance(item, str) for item in feature_columns
    ):
        raise ValueError("feature_metadata.feature_columns must be a string array.")
    available = set(interaction_fields)
    missing_features = [name for name in feature_columns if name not in available]
    required_semantic_inputs = [
        name
        for name in ("action_at_prediction", "look_at_prediction", "gesture_at_prediction")
        if name in feature_columns
    ]
    source_descriptions = feature_metadata.get("categorical_features", {})
    semantic_mismatch = [
        {
            "feature": name,
            "trained_source": source_descriptions.get(name, "annotation-derived"),
            "tracker_observation_available": name in available,
        }
        for name in required_semantic_inputs
        if name not in available
    ]
    return {
        "compatible_for_tracker_inference": not missing_features and not semantic_mismatch,
        "model_feature_count": len(feature_columns),
        "missing_model_feature_columns_from_interaction_output": missing_features,
        "unavailable_annotation_semantics": semantic_mismatch,
        "additional_semantic_mismatch": (
            "The model was fitted on PIE spatial annotation boxes and samples at "
            "annotation critical_point. Interaction inputs use YOLO/BoT-SORT "
            "tracker boxes and tracker-local IDs; their track histories and "
            "sampling semantics are not interchangeable."
        ),
        "inference_performed": False,
    }


def _load_model_contract(
    config: IntegrationConfig,
    metadata: dict[str, Any],
    model_metrics: dict[str, Any],
) -> dict[str, Any]:
    expected_features = metadata.get("feature_columns")
    if not isinstance(expected_features, list) or not all(
        isinstance(name, str) for name in expected_features
    ):
        raise ValueError("Saved feature metadata has invalid feature_columns.")
    thresholds = model_metrics.get("validation_metrics", {}).get(
        config.expected_model_name, {}
    )
    threshold = _parse_number(thresholds.get("threshold"))
    if threshold is None or not 0 <= threshold <= 1:
        raise ValueError("No frozen validation threshold exists for the configured model.")

    caught_warnings: list[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with config.random_forest_model_path.open("rb") as source:
                model = pickle.load(source)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError) as exc:
            raise ValueError(
                f"Unable to load saved {config.expected_model_name} model: {exc}"
            ) from exc
        caught_warnings = list(caught)

    steps = getattr(model, "steps", None)
    if not isinstance(steps, list) or len(steps) != 2:
        raise ValueError("Saved model does not match the expected fitted Pipeline.")
    preprocessor = model.named_steps.get("preprocess")
    classifier = model.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        raise ValueError("Saved model pipeline is missing preprocess/classifier steps.")
    if type(classifier).__name__ != "RandomForestClassifier":
        raise ValueError(
            f"Saved classifier is {type(classifier).__name__}, not RandomForestClassifier."
        )
    transformed = getattr(preprocessor, "transformers_", None)
    if not transformed:
        raise ValueError("Saved model preprocessor has no fitted transformer schema.")
    selected_indices: list[int] = []
    transformer_details: list[dict[str, Any]] = []
    actual_indices_by_name: dict[str, list[int]] = {}
    for name, transformer, columns in transformed:
        if name == "remainder":
            continue
        normalized = [int(value) for value in columns]
        selected_indices.extend(normalized)
        actual_indices_by_name[name] = normalized
        transformer_details.append(
            {
                "name": name,
                "transformer_type": type(transformer).__name__,
                "input_indices": normalized,
                "input_features": [
                    expected_features[index] for index in normalized
                ],
            }
        )
    expected_numeric = metadata.get("numeric_features")
    expected_categorical = metadata.get("categorical_features")
    if not isinstance(expected_numeric, dict) or not isinstance(
        expected_categorical, dict
    ):
        raise ValueError("Feature metadata must define numeric and categorical features.")
    expected_numeric_indices = [
        expected_features.index(name) for name in expected_numeric
    ]
    expected_categorical_indices = [
        expected_features.index(name) for name in expected_categorical
    ]
    schema_matches = (
        len(expected_features) == len(selected_indices)
        and sorted(selected_indices) == list(range(len(expected_features)))
        and actual_indices_by_name.get("numeric") == expected_numeric_indices
        and actual_indices_by_name.get("categorical")
        == expected_categorical_indices
    )
    if not schema_matches:
        raise ValueError(
            "Saved model preprocessing input indices do not match feature metadata."
        )
    warning_details: list[dict[str, str]] = []
    for warning_message in caught_warnings:
        message = str(warning_message.message)
        match = re.search(
            r"Trying to unpickle estimator (\w+) from version ([0-9.]+) "
            r"when using version ([0-9.]+)",
            message,
        )
        if match:
            detail = {
                "warning_type": warning_message.category.__name__,
                "estimator": match.group(1),
                "artifact_version": match.group(2),
                "runtime_version": match.group(3),
            }
        else:
            detail = {
                "warning_type": warning_message.category.__name__,
                "message": message,
            }
        if detail not in warning_details:
            warning_details.append(detail)
    sklearn_versions = sorted(
        {
            detail["artifact_version"]
            for detail in warning_details
            if "artifact_version" in detail
        }
    )
    return {
        "loaded": True,
        "pipeline_type": f"{type(model).__module__}.{type(model).__name__}",
        "classifier_type": f"{type(classifier).__module__}.{type(classifier).__name__}",
        "classifier_count": len(getattr(classifier, "estimators_", [])),
        "classes": [int(value) for value in getattr(classifier, "classes_", [])],
        "feature_metadata_count": len(expected_features),
        "preprocessing_transformers": transformer_details,
        "preprocessing_indices_cover_all_features": schema_matches,
        "validation_threshold": threshold,
        "threshold_source": "model_metrics.json validation_metrics.random_forest.threshold",
        "sklearn_artifact_versions_reported_during_unpickle": sklearn_versions,
        "model_load_warnings": warning_details,
        "inference_performed": False,
        "inference_skip_reason": (
            "Tracker-derived features are semantically incompatible with the "
            "annotation-derived training features, and set01 is a training split."
        ),
    }


def _load_track_mapping(path: Path | None) -> tuple[dict[tuple[str, str, str], dict[str, str]], int]:
    if path is None:
        return {}, 0
    rows, raw_count = _read_csv(
        path,
        (
            "set_id",
            "video_id",
            "interaction_pedestrian_track_id",
            "prediction_track_id",
            "mapping_source",
        ),
    )
    mapping: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row["set_id"].strip(),
            row["video_id"].strip(),
            row["interaction_pedestrian_track_id"].strip(),
        )
        prediction_id = row["prediction_track_id"].strip()
        source = row["mapping_source"].strip()
        if not all(key) or not prediction_id or not source:
            raise ValueError("Track mapping rows must include identity and mapping_source.")
        value = {"prediction_track_id": prediction_id, "mapping_source": source}
        previous = mapping.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"Conflicting track mapping for {key}.")
        mapping[key] = value
    return mapping, raw_count


def _load_predictions(
    paths: Iterable[Path],
    *,
    expected_model_name: str,
    frozen_threshold: float,
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    rows_read = 0
    random_forest_rows = 0
    other_model_rows = 0
    duplicates_removed = 0
    missing_or_invalid_rows = 0
    for path in paths:
        rows, raw_count = _read_csv(path, _PREDICTION_COLUMNS)
        rows_read += raw_count
        unique: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
        for row in rows:
            if row["model"] != expected_model_name:
                other_model_rows += 1
                continue
            random_forest_rows += 1
            set_id = row["set_id"].strip()
            video_id = row["video_id"].strip()
            track_id = row["track_id"].strip()
            frame = _parse_int(row["prediction_frame"])
            probability = _parse_number(row["crossing_probability"])
            predicted = _parse_int(row["predicted_crossing"])
            threshold = _parse_number(row["threshold"])
            if (
                not set_id
                or not video_id
                or not track_id
                or frame is None
                or frame < 0
                or probability is None
                or not 0 <= probability <= 1
                or predicted not in {0, 1}
                or threshold is None
                or not 0 <= threshold <= 1
            ):
                missing_or_invalid_rows += 1
                continue
            if not math.isclose(threshold, frozen_threshold, rel_tol=0, abs_tol=1e-12):
                raise ValueError(
                    f"Prediction threshold {threshold} differs from the frozen "
                    f"validation threshold {frozen_threshold}."
                )
            expected_prediction = int(probability >= frozen_threshold)
            if predicted != expected_prediction:
                raise ValueError(
                    "Saved predicted_crossing does not agree with the frozen threshold."
                )
            record = {
                "set_id": set_id,
                "video_id": video_id,
                "track_id": track_id,
                "prediction_frame": frame,
                "crossing_probability": probability,
                "predicted_crossing": predicted,
                "threshold": threshold,
                "source": path.name,
            }
            key = (set_id, video_id, track_id, frame, expected_model_name)
            previous = unique.get(key)
            if previous is not None:
                if previous != record:
                    raise ValueError(f"Conflicting duplicate prediction {key}.")
                duplicates_removed += 1
                continue
            unique[key] = record
        for record in unique.values():
            grouped[(record["set_id"], record["video_id"], record["track_id"])].append(
                record
            )
    for records in grouped.values():
        records.sort(key=lambda item: item["prediction_frame"])
    return grouped, {
        "files": [str(path) for path in paths],
        "rows_read": rows_read,
        "random_forest_prediction_rows": random_forest_rows,
        "other_model_rows_ignored": other_model_rows,
        "invalid_rows": missing_or_invalid_rows,
        "duplicate_rows_removed": duplicates_removed,
        "unique_prediction_tracks": len(grouped),
        "unique_prediction_records": sum(len(values) for values in grouped.values()),
    }


def _prediction_timestamp_status(
    prediction_frame: int,
    interaction_timestamp: float | None,
    interaction_frame: int,
    *,
    fps: float,
    tolerance: float,
) -> tuple[str, float | None]:
    predicted_time = prediction_frame / fps
    if interaction_timestamp is None:
        return "missing_interaction_timestamp", predicted_time
    interaction_time_from_frame = interaction_frame / fps
    if abs(interaction_timestamp - interaction_time_from_frame) > tolerance:
        return "interaction_timestamp_frame_mismatch", predicted_time
    if abs(predicted_time - prediction_frame / fps) > tolerance:
        return "prediction_timestamp_frame_mismatch", predicted_time
    return "consistent", predicted_time


def _association_for_row(
    row: dict[str, Any],
    predictions: dict[tuple[str, str, str], list[dict[str, Any]]],
    track_mapping: dict[tuple[str, str, str], dict[str, str]],
    *,
    model_name: str,
    frozen_threshold: float,
    max_age_frames: int,
    fps: float,
    timestamp_tolerance: float,
) -> dict[str, Any]:
    mapping_key = (
        row["set_id"],
        row["video_id"],
        row["pedestrian_track_id"],
    )
    mapping = track_mapping.get(mapping_key)
    base = {
        "crossing_probability": None,
        "crossing_prediction": None,
        "prediction_available": False,
        "prediction_status": "no_identity_mapping",
        "prediction_source": model_name,
        "prediction_frame": None,
        "prediction_timestamp_seconds": None,
        "prediction_age_frames": None,
        "prediction_age_seconds": None,
        "prediction_threshold": frozen_threshold,
        "prediction_identity_mapping_source": (
            mapping["mapping_source"] if mapping else ""
        ),
        "prediction_timestamp_status": "unavailable",
        "association_validity": "unavailable",
    }
    if mapping is None:
        return base

    key = (
        row["set_id"],
        row["video_id"],
        mapping["prediction_track_id"],
    )
    records = predictions.get(key, [])
    if not records:
        base["prediction_status"] = "no_prediction_for_mapped_track"
        return base
    frame = row["frame_index"]
    prior = [record for record in records if record["prediction_frame"] <= frame]
    if not prior:
        base["prediction_status"] = "prediction_is_after_interaction_frame"
        base["association_validity"] = "invalid_future_prediction"
        return base
    candidate = prior[-1]
    age = frame - candidate["prediction_frame"]
    if age > max_age_frames:
        timestamp_status, prediction_timestamp = _prediction_timestamp_status(
            candidate["prediction_frame"],
            row["timestamp_seconds"],
            frame,
            fps=fps,
            tolerance=timestamp_tolerance,
        )
        return {
            **base,
            "prediction_status": "prediction_expired",
            "prediction_frame": candidate["prediction_frame"],
            "prediction_timestamp_seconds": prediction_timestamp,
            "prediction_age_frames": age,
            "prediction_age_seconds": age / fps,
            "prediction_timestamp_status": timestamp_status,
            "association_validity": "invalid_stale_prediction",
        }
    if row["interaction_timestamp_status"] != "consistent":
        base["prediction_status"] = "invalid_interaction_timestamp"
        base["association_validity"] = "invalid_timestamp"
        return base
    timestamp_status, prediction_timestamp = _prediction_timestamp_status(
        candidate["prediction_frame"],
        row["timestamp_seconds"],
        frame,
        fps=fps,
        tolerance=timestamp_tolerance,
    )
    if timestamp_status != "consistent":
        base["prediction_status"] = timestamp_status
        base["association_validity"] = "invalid_timestamp"
        base["prediction_frame"] = candidate["prediction_frame"]
        base["prediction_timestamp_seconds"] = prediction_timestamp
        base["prediction_age_frames"] = age
        base["prediction_age_seconds"] = age / fps
        base["prediction_timestamp_status"] = timestamp_status
        return base
    if not math.isclose(
        candidate["threshold"], frozen_threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Prediction uses a threshold other than the frozen threshold.")
    return {
        **base,
        "crossing_probability": candidate["crossing_probability"],
        "crossing_prediction": candidate["predicted_crossing"],
        "prediction_available": True,
        "prediction_status": "matched",
        "prediction_frame": candidate["prediction_frame"],
        "prediction_timestamp_seconds": prediction_timestamp,
        "prediction_age_frames": age,
        "prediction_age_seconds": age / fps,
        "prediction_timestamp_status": "consistent",
        "association_validity": "valid",
    }


def _attach_causal_closest_approach(
    rows: list[dict[str, Any]],
    minimum_observations: int,
) -> None:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["set_id"],
                row["video_id"],
                row["pedestrian_track_id"],
                row["vehicle_class"],
                row["vehicle_track_id"],
            )
        ].append(row)
    for observations in grouped.values():
        observations.sort(key=lambda item: item["frame_index"])
        segment: list[dict[str, Any]] = []
        for row in observations:
            timestamp_valid = (
                row["interaction_timestamp_status"] == "consistent"
                and row["timestamp_seconds"] is not None
            )
            if not timestamp_valid:
                segment = [row]
                row["closest_approach_valid_as_of_frame"] = False
                row["closest_approach_invalid_reason"] = (
                    "invalid_interaction_timestamp"
                )
                row["closest_approach_frame_index_as_of_frame"] = None
                row["closest_approach_timestamp_seconds_as_of_frame"] = None
                row["closest_approach_separation_pixels_as_of_frame"] = None
                continue
            if not row["movement_valid"]:
                segment = [row]
            else:
                segment.append(row)
            if len(segment) < minimum_observations:
                row["closest_approach_valid_as_of_frame"] = False
                row["closest_approach_invalid_reason"] = (
                    "insufficient_consecutive_valid_observations"
                )
                row["closest_approach_frame_index_as_of_frame"] = None
                row["closest_approach_timestamp_seconds_as_of_frame"] = None
                row["closest_approach_separation_pixels_as_of_frame"] = None
                continue
            closest = min(
                segment,
                key=lambda item: (item["distance_pixels"], item["frame_index"]),
            )
            row["closest_approach_valid_as_of_frame"] = True
            row["closest_approach_invalid_reason"] = ""
            row["closest_approach_frame_index_as_of_frame"] = closest["frame_index"]
            row["closest_approach_timestamp_seconds_as_of_frame"] = (
                closest["timestamp_seconds"]
            )
            row["closest_approach_separation_pixels_as_of_frame"] = (
                closest["distance_pixels"]
            )


def associate_predictions(
    interaction_rows: Iterable[dict[str, Any]],
    predictions: dict[tuple[str, str, str], list[dict[str, Any]]],
    track_mapping: dict[tuple[str, str, str], dict[str, str]],
    *,
    model_name: str,
    frozen_threshold: float,
    max_age_frames: int,
    fps: float,
    timestamp_tolerance: float,
    minimum_observations_for_closest_approach: int = 3,
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, int, str, str, str], dict[str, Any]] = {}
    for source_row in interaction_rows:
        row = dict(source_row)
        key = (
            row["set_id"],
            row["video_id"],
            row["frame_index"],
            row["pedestrian_track_id"],
            row["vehicle_class"],
            row["vehicle_track_id"],
        )
        previous = unique.get(key)
        if previous is not None:
            if previous != row:
                raise ValueError(f"Conflicting duplicate interaction observation {key}.")
            continue
        unique[key] = row
    rows = sorted(
        unique.values(),
        key=lambda item: (
            item["set_id"],
            item["video_id"],
            item["pedestrian_track_id"],
            item["vehicle_class"],
            item["vehicle_track_id"],
            item["frame_index"],
        ),
    )
    for row in rows:
        association = _association_for_row(
            row,
            predictions,
            track_mapping,
            model_name=model_name,
            frozen_threshold=frozen_threshold,
            max_age_frames=max_age_frames,
            fps=fps,
            timestamp_tolerance=timestamp_tolerance,
        )
        row.update(association)
        warnings_list: list[str] = []
        if row.get("interaction_timestamp_status") != "consistent":
            warnings_list.append("interaction_timestamp_invalid")
        if not row["prediction_available"]:
            warnings_list.append(row["prediction_status"])
        if not row.get("movement_valid", False):
            warnings_list.append(row.get("movement_invalid_reason") or "movement_invalid")
        row["data_quality_warnings"] = ";".join(sorted(set(warnings_list)))
    _attach_causal_closest_approach(
        rows, minimum_observations_for_closest_approach
    )
    return rows


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        row["set_id"],
        row["video_id"],
        row["pedestrian_track_id"],
        row["vehicle_class"],
        row["vehicle_track_id"],
    )


def _summarize(
    rows: list[dict[str, Any]],
    interaction_summaries: list[dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_pair_key(row)].append(row)
    source_summaries: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for summary in interaction_summaries:
        key = (
            summary["set_id"],
            summary["video_id"],
            summary["pedestrian_track_id"],
            summary["vehicle_class"],
            summary["vehicle_track_id"],
        )
        source_summaries[key] = summary

    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        observations = grouped[key]
        distances = [float(row["distance_pixels"]) for row in observations]
        probabilities = [
            float(row["crossing_probability"])
            for row in observations
            if row["prediction_available"] and row["crossing_probability"] is not None
        ]
        prediction_count = len(probabilities)
        warnings_found = sorted(
            {
                warning
                for row in observations
                for warning in row["data_quality_warnings"].split(";")
                if warning
            }
        )
        source_summary = source_summaries.get(key)
        movement_status = (
            source_summary.get("movement_status", "insufficient_observations")
            if source_summary
            else "insufficient_observations"
        )
        output.append(
            {
                "set_id": key[0],
                "video_id": key[1],
                "pedestrian_track_id": key[2],
                "vehicle_track_id": key[4],
                "vehicle_class": key[3],
                "observation_count": len(observations),
                "minimum_separation_pixels": min(distances) if distances else None,
                "median_separation_pixels": (
                    statistics.median(distances) if distances else None
                ),
                "movement_status": movement_status,
                "crossing_probability_mean": (
                    statistics.mean(probabilities) if probabilities else None
                ),
                "crossing_probability_minimum": min(probabilities) if probabilities else None,
                "crossing_probability_maximum": max(probabilities) if probabilities else None,
                "crossing_prediction_positive_count": sum(
                    int(row["crossing_prediction"] == 1)
                    for row in observations
                    if row["prediction_available"]
                ),
                "observations_with_valid_intention_prediction": prediction_count,
                "percentage_with_valid_intention_prediction": (
                    prediction_count / len(observations) * 100 if observations else 0.0
                ),
                "data_quality_warnings": ";".join(warnings_found),
            }
        )
    return output


def _metrics(
    *,
    interactions: list[dict[str, Any]],
    integrated: list[dict[str, Any]],
    predictions_info: dict[str, Any],
    prediction_records: dict[tuple[str, str, str], list[dict[str, Any]]],
    mapping: dict[tuple[str, str, str], dict[str, str]],
    mapping_rows_read: int,
    compatibility: dict[str, Any],
    model_contract: dict[str, Any],
    interaction_input_count: int,
    summary_input_count: int,
    prediction_interaction_summary: list[dict[str, str]],
    interaction_run_config: dict[str, Any],
    config: IntegrationConfig,
    horizon_frames: int,
) -> dict[str, Any]:
    available = [row for row in integrated if row["prediction_available"]]
    unavailable_reasons = Counter(
        row["prediction_status"] for row in integrated if not row["prediction_available"]
    )
    association_invalid = Counter(
        row["association_validity"]
        for row in integrated
        if str(row["association_validity"]).startswith("invalid")
    )
    matched_pedestrians = {
        (row["set_id"], row["video_id"], row["pedestrian_track_id"])
        for row in available
    }
    interaction_pedestrians = {
        (row["set_id"], row["video_id"], row["pedestrian_track_id"])
        for row in interactions
    }
    interaction_timestamp_counts = Counter(
        row["interaction_timestamp_status"] for row in interactions
    )
    timestamp_track_mismatch_rows = 0
    for row in interactions:
        if row["timestamp_seconds"] is not None and abs(
            row["timestamp_seconds"] - row["frame_index"] / config.fps
        ) > config.timestamp_tolerance_seconds:
            timestamp_track_mismatch_rows += 1

    interaction_recordings = {
        (row["set_id"], row["video_id"]) for row in interactions
    }
    prediction_recordings = {
        (set_id, video_id) for set_id, video_id, _ in prediction_records
    }
    prediction_records_outside_interaction_recordings = sum(
        len(records)
        for key, records in prediction_records.items()
        if key[:2] not in interaction_recordings
    )
    mapped_pedestrians = interaction_pedestrians & set(mapping)
    return {
        "experiment": {
            "interaction_recordings": [
                {"set_id": set_id, "video_id": video_id}
                for set_id, video_id in sorted(interaction_recordings)
            ],
            "prediction_source_model": config.expected_model_name,
            "prediction_age_limit_frames": config.max_prediction_age_frames,
            "model_target_horizon_frames": horizon_frames,
            "frozen_validation_threshold": model_contract["validation_threshold"],
            "threshold_tuned_in_this_task": False,
            "inference_performed": False,
        },
        "input_record_counts": {
            "interaction_observations_rows": interaction_input_count,
            "interaction_observations_unique_rows": len(interactions),
            "interaction_summary_rows": summary_input_count,
            "random_forest_prediction_rows": predictions_info[
                "random_forest_prediction_rows"
            ],
            "random_forest_prediction_records": predictions_info[
                "unique_prediction_records"
            ],
            "track_mapping_rows": mapping_rows_read,
        },
        "pedestrian_and_interaction_counts": {
            "unique_interaction_pedestrians": len(interaction_pedestrians),
            "matched_pedestrians_with_valid_prediction": len(matched_pedestrians),
            "mapped_interaction_pedestrians": len(mapped_pedestrians),
            "interaction_observations": len(interactions),
            "interactions_with_valid_prediction": len(available),
            "interactions_without_valid_prediction": len(integrated) - len(available),
            "unique_interaction_pairs": len({_pair_key(row) for row in integrated}),
        },
        "prediction_association": {
            "available_interaction_predictions": len(available),
            "unavailable_interaction_predictions": len(integrated) - len(available),
            "association_coverage": (
                len(available) / len(integrated) if integrated else None
            ),
            "unavailable_reasons": dict(sorted(unavailable_reasons.items())),
            "invalid_associations": dict(sorted(association_invalid.items())),
            "prediction_records_outside_interaction_recordings": (
                prediction_records_outside_interaction_recordings
            ),
            "candidate_prediction_recordings": [
                {"set_id": set_id, "video_id": video_id}
                for set_id, video_id in sorted(prediction_recordings)
            ],
            "prediction_is_never_associated_before_its_frame": True,
            "prediction_age_capped_by_target_horizon": horizon_frames,
        },
        "feature_compatibility": compatibility,
        "saved_model_contract": model_contract,
        "timestamp_and_track_consistency": {
            "interaction_timestamp_status_counts": dict(
                sorted(interaction_timestamp_counts.items())
            ),
            "interaction_timestamps_not_consistent_with_frame_and_fps": (
                timestamp_track_mismatch_rows
            ),
            "timestamp_tolerance_seconds": config.timestamp_tolerance_seconds,
            "fps_for_prediction_timestamp_derivation": config.fps,
            "explicit_track_mapping_required": True,
            "track_mapping_provided": config.track_mapping_path is not None,
            "interaction_tracks_with_explicit_mapping": len(mapped_pedestrians),
            "numeric_tracker_ids_compared_directly_to_annotation_ids": False,
        },
        "data_quality": {
            "interaction_duplicate_rows_removed": interaction_input_count
            - len(interactions),
            "interaction_summary_pairs_without_observations": len(
                set(
                    (
                        row["set_id"],
                        row["video_id"],
                        row["pedestrian_track_id"],
                        row["vehicle_class"],
                        row["vehicle_track_id"],
                    )
                    for row in prediction_interaction_summary
                )
                - {_pair_key(row) for row in integrated}
            ),
            "thresholds_verified_against_frozen_validation_threshold": True,
            "probability_or_prediction_disagreement_rows": 0,
            "interaction_source_set_ids": interaction_run_config.get("set_ids", []),
            "interaction_source_video_ids": interaction_run_config.get("video_ids", []),
            "interaction_source_max_observation_gap_frames": interaction_run_config.get(
                "max_observation_gap_frames"
            ),
        },
        "interpretation": (
            "The saved model was loaded and its preprocessing contract checked, "
            "but no prediction was produced because tracker-derived interaction "
            "data do not supply compatible annotation-derived features or a "
            "ground-truth-to-tracker identity map. Set01 is also a training split."
        ),
    }


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _build_report(
    metrics: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> str:
    counts = metrics["pedestrian_and_interaction_counts"]
    association = metrics["prediction_association"]
    compatibility = metrics["feature_compatibility"]
    model = metrics["saved_model_contract"]
    lines = [
        "# Crossing-Intention and Interaction Integration",
        "",
        "## Result and scope",
        "",
        "This integration associates only saved Random Forest prediction records "
        "with interaction observations when recording identity, an explicit "
        "pedestrian-ID crosswalk, timestamp, and causal prediction age all validate. "
        "It does not train, tune, or run the model on incompatible tracker features.",
        "",
        f"- Interactions: {counts['interaction_observations']:,} across "
        f"{counts['unique_interaction_pairs']:,} pedestrian–vehicle pairs.",
        f"- Interactions with valid crossing predictions: "
        f"{association['available_interaction_predictions']:,}.",
        f"- Interactions with unavailable crossing predictions: "
        f"{association['unavailable_interaction_predictions']:,}.",
        f"- Association coverage: "
        f"{association['association_coverage'] if association['association_coverage'] is not None else 'n/a'}.",
        f"- Model inference performed: {str(model['inference_performed']).lower()}.",
        "",
        "## Model and feature compatibility",
        "",
        f"The saved model loaded as `{model['pipeline_type']}` with "
        f"`{model['classifier_type']}` ({model['classifier_count']} estimators). "
        f"The saved preprocessing configuration covers all "
        f"{model['feature_metadata_count']} metadata features. The frozen Random "
        f"Forest threshold remains `{model['validation_threshold']:.16g}`, read "
        "from the existing validation metrics; no threshold was tuned here.",
        "",
        f"Tracker feature compatibility: "
        f"`{compatibility['compatible_for_tracker_inference']}`. "
        f"The interaction schema lacks "
        f"{len(compatibility['missing_model_feature_columns_from_interaction_output'])} "
        "of the model's input feature columns, including PIE action/look/gesture "
        "attributes. Remaining model inputs were defined on annotated PIE boxes "
        "and annotation histories at `critical_point`; they are not semantically "
        "equivalent to YOLO/BoT-SORT boxes, pair-conditioned interaction rows, or "
        "tracker-local histories. The engine therefore does not call `predict` or "
        "`predict_proba` and does not fill missing model features.",
        "",
        "The training configuration places set01 in the training split, while "
        "the persisted prediction records are validation set06 and held-out test "
        "set04. They do not provide set01 predictions. Set04 predictions are used "
        "only as saved records and are never used to change the model or threshold.",
        "",
        "The model artifact was serialized with scikit-learn version(s) "
        f"{', '.join(model['sklearn_artifact_versions_reported_during_unpickle']) or 'not recorded'}; "
        "the active unpickler emitted compatibility warnings if versions differ. "
        "Those warnings are recorded in `integration_metrics.json`.",
        "",
        "## Association rules",
        "",
        "A model prediction is matched by `(set_id, video_id, mapped prediction "
        "track ID)` to an interaction pedestrian only through an explicit "
        "track-mapping CSV. Numeric IDs from the PIE annotation/model output and "
        "YOLO/BoT-SORT tracker are never assumed equivalent. A prediction can be "
        "attached only at or after its `prediction_frame`, and only through the "
        "30-frame target horizon (age 0–30). Probability, binary prediction, and "
        "threshold are taken from the saved RF output and checked against the "
        "frozen threshold.",
        "",
        "No crosswalk was available for this run, and all stored RF outputs belong "
        "to set06 or set04 rather than the set01/video_0001 interaction clip. Each "
        "integrated observation therefore contains a null probability/prediction "
        "with an explicit unavailable status; no annotation track is guessed.",
        "",
        "Per-observation closest separation is computed causally from interaction "
        "rows up to that frame, within the current uninterrupted valid segment. "
        "It does not expose a future pair minimum to earlier observations. Pair "
        "summaries report the interaction engine's observed min/median pixel "
        "separation.",
        "",
        "## Results and limitations",
        "",
        f"- Unique interaction pedestrians: {counts['unique_interaction_pedestrians']:,}.",
        f"- Matched pedestrians with a valid intention prediction: "
        f"{counts['matched_pedestrians_with_valid_prediction']:,}.",
        f"- Unavailable reasons: `{json.dumps(association['unavailable_reasons'], sort_keys=True)}`.",
        f"- Timestamp/frame inconsistencies in interaction input: "
        f"{metrics['timestamp_and_track_consistency']['interaction_timestamps_not_consistent_with_frame_and_fps']:,}.",
        "",
        "The intention target remains the existing label definition: at least one "
        "visible PIE `crossing` action in the next 30 frames after the annotation "
        "`critical_point`; this is not collision ground truth. The model was "
        "trained on annotation-derived samples, not validated on tracker-derived "
        "features. Image-space separation and movement are not meters or physical "
        "velocity. No collision-risk score or Low/Medium/High classification is "
        "created.",
        "",
        "The Random Forest was loaded from the existing artifact without retraining. "
        "No model, feature, threshold, interaction, or previous experiment artifact "
        "was modified.",
        "",
        "## Integrated pair summary",
        "",
    ]
    if summaries:
        lines.extend(
            [
                "| Recording | Pedestrian | Vehicle | Observations | Min/median separation (px) | Predictions available | Coverage | Warnings |",
                "|---|---|---|---:|---|---:|---:|---|",
            ]
        )
        for row in summaries[:20]:
            lines.append(
                f"| {row['set_id']}/{row['video_id']} | "
                f"{row['pedestrian_track_id']} | "
                f"{row['vehicle_class']}/{row['vehicle_track_id']} | "
                f"{row['observation_count']} | "
                f"{row['minimum_separation_pixels']:.1f}/"
                f"{row['median_separation_pixels']:.1f} | "
                f"{row['observations_with_valid_intention_prediction']} | "
                f"{row['percentage_with_valid_intention_prediction']:.1f}% | "
                f"{row['data_quality_warnings']} |"
            )
        if len(summaries) > 20:
            lines.append(f"\nShowing 20 of {len(summaries)} pairs.")
    else:
        lines.append("No integrated interaction pairs were available.")
    lines.append("")
    return "\n".join(lines)


def run_integration(config_path: Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    required_paths = (
        "interaction_observations_path",
        "interaction_summary_path",
        "interaction_run_config_path",
        "random_forest_model_path",
        "feature_metadata_path",
        "model_metrics_path",
        "generation_config_path",
        "validation_predictions_path",
        "test_predictions_path",
    )
    for name in required_paths:
        source_path = getattr(config, name)
        if not source_path.is_file():
            raise FileNotFoundError(f"Configured {name} does not exist: {source_path}")
    if config.track_mapping_path is not None and not config.track_mapping_path.is_file():
        raise FileNotFoundError(
            f"Configured track_mapping_path does not exist: {config.track_mapping_path}"
        )

    interaction_rows_raw, interaction_input_count = _read_csv(
        config.interaction_observations_path,
        (
            "set_id",
            "video_id",
            "frame_index",
            "timestamp_seconds",
            "timestamp_validation_status",
            "pedestrian_track_id",
            "vehicle_track_id",
            "vehicle_class",
            "distance_pixels",
            "movement_status",
            "movement_valid",
            "movement_invalid_reason",
        ),
    )
    interactions, duplicate_count = _parse_interactions(
        interaction_rows_raw,
        fps=config.fps,
        timestamp_tolerance=config.timestamp_tolerance_seconds,
    )
    if duplicate_count:
        interaction_input_count = len(interactions) + duplicate_count
    summaries_raw, summary_input_count = _read_csv(
        config.interaction_summary_path,
        (
            "set_id",
            "video_id",
            "pedestrian_track_id",
            "vehicle_class",
            "vehicle_track_id",
            "observation_count",
            "minimum_separation_pixels",
            "median_separation_pixels",
            "movement_status",
        ),
    )
    interaction_run_config = _read_json(
        config.interaction_run_config_path, "interaction run configuration"
    )
    feature_metadata = _read_json(config.feature_metadata_path, "feature metadata")
    model_metrics = _read_json(config.model_metrics_path, "model metrics")
    generation_config = _read_json(config.generation_config_path, "generation config")
    horizon = generation_config.get("target_definition", {}).get("horizon_frames")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("Saved generation config has no valid target horizon.")
    if config.max_prediction_age_frames > horizon:
        raise ValueError(
            "max_prediction_age_frames cannot exceed the saved target horizon."
        )
    saved_threshold = model_metrics.get("validation_metrics", {}).get(
        config.expected_model_name, {}
    ).get("threshold")
    model_contract = _load_model_contract(config, feature_metadata, model_metrics)
    frozen_threshold = model_contract["validation_threshold"]
    if saved_threshold != frozen_threshold:
        raise ValueError("Loaded threshold does not match the saved validation metrics.")
    predictions, predictions_info = _load_predictions(
        (config.validation_predictions_path, config.test_predictions_path),
        expected_model_name=config.expected_model_name,
        frozen_threshold=frozen_threshold,
    )
    track_mapping, mapping_rows_read = _load_track_mapping(
        config.track_mapping_path
    )
    interaction_fields = (
        interaction_rows_raw[0].keys()
        if interaction_rows_raw
        else _INTEGRATED_FIELDS
    )
    compatibility = assess_feature_compatibility(
        feature_metadata, interaction_fields
    )

    integrated = associate_predictions(
        interactions,
        predictions,
        track_mapping,
        model_name=config.expected_model_name,
        frozen_threshold=frozen_threshold,
        max_age_frames=config.max_prediction_age_frames,
        fps=config.fps,
        timestamp_tolerance=config.timestamp_tolerance_seconds,
        minimum_observations_for_closest_approach=(
            config.minimum_observations_for_closest_approach
        ),
    )
    integrated_summaries = _summarize(integrated, summaries_raw)
    metrics = _metrics(
        interactions=interactions,
        integrated=integrated,
        predictions_info=predictions_info,
        prediction_records=predictions,
        mapping=track_mapping,
        mapping_rows_read=mapping_rows_read,
        compatibility=compatibility,
        model_contract=model_contract,
        interaction_input_count=interaction_input_count,
        summary_input_count=summary_input_count,
        prediction_interaction_summary=summaries_raw,
        interaction_run_config=interaction_run_config,
        config=config,
        horizon_frames=horizon,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        config.output_dir / "integrated_interactions.csv",
        _INTEGRATED_FIELDS,
        integrated,
    )
    _write_csv(
        config.output_dir / "integrated_summary.csv",
        _SUMMARY_FIELDS,
        integrated_summaries,
    )
    (config.output_dir / "integration_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_snapshot = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
    }
    (config.output_dir / "integration_run_config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "INTEGRATION_REPORT.md").write_text(
        _build_report(metrics, integrated_summaries),
        encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "integrated_observations": integrated,
        "integrated_summaries": integrated_summaries,
        "report_path": config.output_dir / "INTEGRATION_REPORT.md",
    }
