from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

from .comparison import compare_predictions_to_pie
from .features import (
    coverage_diagnostics,
    extract_track_features,
    summarize_tracks,
)
from .models import (
    CSV_FIELDS,
    MOVEMENT_STANDING,
    MOVEMENT_UNKNOWN,
    MOVEMENT_WALKING,
    TrajectoryConfig,
    TrajectoryFeature,
)
from .readers import load_observations
from .report import render_trajectory_report, trajectory_limitations

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "trajectory_extraction.json"
)
_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "artifacts" / "trajectory_extraction"
)


def _as_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field {name!r} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValueError(f"Config field {name!r} must be finite.")
    return result


def _as_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field {name!r} must be an integer.") from exc
    return result


def load_trajectory_config(path: Path) -> TrajectoryConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read trajectory config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Trajectory config must be a JSON object.")

    config = TrajectoryConfig(
        fps=_as_float(raw.get("fps", 30.0), "fps"),
        max_gap_frames=_as_int(raw.get("max_gap_frames", 15), "max_gap_frames"),
        standing_speed_threshold_px_s=_as_float(
            raw.get("standing_speed_threshold_px_s", 10.0),
            "standing_speed_threshold_px_s",
        ),
        smoothing_window_observations=_as_int(
            raw.get("smoothing_window_observations", 9),
            "smoothing_window_observations",
        ),
        smoothing_min_observations=_as_int(
            raw.get("smoothing_min_observations", 3),
            "smoothing_min_observations",
        ),
    )
    config.validate()
    return config


def _write_csv(path: Path, features: list[TrajectoryFeature]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for feature in features:
            writer.writerow(feature.to_csv_row())


def _write_jsonl(path: Path, features: list[TrajectoryFeature]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for feature in features:
            destination.write(
                json.dumps(
                    feature.to_dict(),
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _build_summary(
    source: Path,
    input_format: str,
    config: TrajectoryConfig,
    features: list[TrajectoryFeature],
    track_summaries: list[dict[str, Any]],
    issues: list[Any],
) -> dict[str, Any]:
    states = [feature.smoothed_movement_state for feature in features]
    durations = [
        float(summary["duration_seconds"])
        for summary in track_summaries
        if summary["duration_seconds"] is not None
    ]
    expected_observations = sum(
        summary["expected_observation_count"] for summary in track_summaries
    )
    observation_coverage = (
        len(features) / expected_observations if expected_observations else None
    )
    return {
        "schema_version": 1,
        "input": source.as_posix(),
        "input_format": input_format,
        "config": config.to_dict(),
        "counts": {
            "tracks": len(track_summaries),
            "observations": len(features),
            "expected_observations_within_track_spans": expected_observations,
            "missing_observations_within_track_spans": sum(
                summary["missing_observation_count"] for summary in track_summaries
            ),
            "tracks_with_missing_observations": sum(
                summary["missing_observation_count"] > 0
                for summary in track_summaries
            ),
            "observation_coverage_ratio": observation_coverage,
            "valid_velocity_observations": sum(
                feature.velocity_valid for feature in features
            ),
            "long_gap_observations": sum(
                feature.gap_exceeded for feature in features
            ),
            "walking_observations": states.count(MOVEMENT_WALKING),
            "standing_observations": states.count(MOVEMENT_STANDING),
            "unknown_state_observations": states.count(MOVEMENT_UNKNOWN),
            "track_duration_seconds": (
                {
                    "total": sum(durations),
                    "median": median(durations),
                    "min": min(durations),
                    "max": max(durations),
                }
                if durations
                else None
            ),
            "error_issues": sum(issue.severity == "error" for issue in issues),
            "warning_issues": sum(issue.severity == "warning" for issue in issues),
            "coverage_diagnostics": coverage_diagnostics(features),
        },
        "limitations": trajectory_limitations(),
        "tracks": track_summaries,
        "issues": [issue.to_dict() for issue in issues],
    }


def run_extraction(
    source: Path,
    *,
    config_path: Path = _DEFAULT_CONFIG,
    output_dir: Path | None = None,
    input_format: str = "auto",
    default_set_id: str | None = None,
    default_video_id: str | None = None,
    model_path: Path | None = None,
    tracker_config: Path | None = None,
    confidence: float = 0.25,
    fps_override: float | None = None,
    max_seconds: float | None = None,
    annotations: Path | None = None,
    comparison_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    config = load_trajectory_config(config_path)
    if fps_override is not None:
        config = replace(config, fps=_as_float(fps_override, "fps"))
        config.validate()
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1.")
    if not 0 <= comparison_iou_threshold <= 1:
        raise ValueError("comparison_iou_threshold must be between 0 and 1.")

    observations, issues, resolved_format = load_observations(
        source,
        config,
        input_format=input_format,
        default_set_id=default_set_id,
        default_video_id=default_video_id,
        model_path=model_path,
        tracker_config=tracker_config,
        confidence=confidence,
        max_seconds=max_seconds,
    )
    features = extract_track_features(observations, config)
    summaries = summarize_tracks(features)
    track_summaries = [summary.to_dict() for summary in summaries]
    inference = (
        {
            "detector_weights": (
                model_path.resolve().as_posix() if model_path is not None else None
            ),
            "tracker_config": (
                tracker_config.resolve().as_posix()
                if tracker_config is not None
                else None
            ),
            "confidence": confidence,
            "max_seconds": max_seconds,
        }
        if resolved_format == "video"
        else None
    )
    comparison = (
        compare_predictions_to_pie(
            features,
            annotations,
            iou_threshold=comparison_iou_threshold,
        )
        if annotations is not None
        else None
    )
    if comparison is not None and inference is not None:
        comparison["predictions"].update(inference)

    destination = (output_dir or _DEFAULT_OUTPUT_DIR).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "trajectory_features.csv"
    jsonl_path = destination / "trajectory_features.jsonl"
    summary_path = destination / "trajectory_summary.json"
    report_path = destination / "trajectory_report.md"
    comparison_path = destination / "trajectory_annotation_comparison.json"

    _write_csv(csv_path, features)
    _write_jsonl(jsonl_path, features)
    summary = _build_summary(
        source.resolve(),
        resolved_format,
        config,
        features,
        track_summaries,
        issues,
    )
    if inference is not None:
        summary["inference"] = inference
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if comparison is not None:
        comparison_path.write_text(
            json.dumps(comparison, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    report_path.write_text(
        render_trajectory_report(
            summary,
            comparison,
            comparison_path.as_posix() if comparison is not None else None,
        ),
        encoding="utf-8",
    )

    return {
        "output_dir": destination,
        "csv_path": csv_path,
        "jsonl_path": jsonl_path,
        "summary_path": summary_path,
        "report_path": report_path,
        "comparison_path": comparison_path if comparison is not None else None,
        "summary": summary,
        "comparison": comparison,
        "issues": issues,
        "input_format": resolved_format,
    }
