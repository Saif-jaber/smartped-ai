from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from ..data.prepare_detection import read_pedestrian_xml
from .calibration import (
    choose_speed_threshold,
    classification_metrics,
)
from .camera_motion import (
    CameraMotionConfig,
    CameraMotionSequence,
    estimate_camera_motion,
)
from .comparison import compare_predictions_to_pie
from .ego_telemetry import summarize_ego_telemetry
from .features import (
    coverage_diagnostics,
    extract_track_features,
    summarize_tracks,
)
from .models import (
    MOVEMENT_STANDING,
    MOVEMENT_WALKING,
    TrackObservation,
    TrajectoryConfig,
    TrajectoryFeature,
)
from .pipeline import _build_summary, _write_csv, _write_jsonl
from .readers import load_observations

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = (
    _PROJECT_ROOT / "configs" / "camera_motion_experiment.json"
)


@dataclass(frozen=True)
class ExperimentClip:
    set_id: str
    video_id: str
    video_path: Path
    annotation_path: Path
    ego_path: Path
    max_seconds: float


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


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Config field {name!r} must be a boolean.")


def _resolve_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field {name!r} must be a non-empty path string.")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _clip_from_mapping(
    root: Path,
    raw: Any,
    *,
    default_max_seconds: float,
    name: str,
) -> ExperimentClip:
    if not isinstance(raw, dict):
        raise ValueError(f"Config field {name!r} must be an object.")
    set_id = str(raw.get("set_id", "")).strip()
    video_id = str(raw.get("video_id", "")).strip()
    if not set_id or not video_id:
        raise ValueError(f"Config field {name!r} requires set_id and video_id.")
    video_path = _resolve_path(
        root,
        raw.get(
            "video",
            f"dataset/PIE_clips/{set_id}/{video_id}.mp4",
        ),
        f"{name}.video",
    )
    annotation_path = _resolve_path(
        root,
        raw.get(
            "annotations",
            f"dataset/annotations/{set_id}/{video_id}_annt.xml",
        ),
        f"{name}.annotations",
    )
    ego_path = _resolve_path(
        root,
        raw.get(
            "ego_telemetry",
            f"dataset/annotations_vehicle/{set_id}/{video_id}_obd.xml",
        ),
        f"{name}.ego_telemetry",
    )
    max_seconds = _as_float(
        raw.get("max_seconds", default_max_seconds),
        f"{name}.max_seconds",
    )
    if max_seconds <= 0:
        raise ValueError(f"Config field {name!r}.max_seconds must be positive.")
    return ExperimentClip(
        set_id=set_id,
        video_id=video_id,
        video_path=video_path,
        annotation_path=annotation_path,
        ego_path=ego_path,
        max_seconds=max_seconds,
    )


def _camera_config_from_mapping(raw: Any) -> CameraMotionConfig:
    if not isinstance(raw, dict):
        raise ValueError("Config field 'camera_motion' must be an object.")
    config = CameraMotionConfig(
        max_width_px=_as_int(raw.get("max_width_px", 960), "max_width_px"),
        feature_max_corners=_as_int(
            raw.get("feature_max_corners", 2500), "feature_max_corners"
        ),
        feature_quality_level=_as_float(
            raw.get("feature_quality_level", 0.01), "feature_quality_level"
        ),
        feature_min_distance_px=_as_float(
            raw.get("feature_min_distance_px", 10.0),
            "feature_min_distance_px",
        ),
        feature_block_size_px=_as_int(
            raw.get("feature_block_size_px", 7), "feature_block_size_px"
        ),
        lk_window_size_px=_as_int(
            raw.get("lk_window_size_px", 21), "lk_window_size_px"
        ),
        lk_max_level=_as_int(raw.get("lk_max_level", 3), "lk_max_level"),
        lk_backward_error_px=_as_float(
            raw.get("lk_backward_error_px", 1.0), "lk_backward_error_px"
        ),
        local_radius_px=_as_float(
            raw.get("local_radius_px", 160.0), "local_radius_px"
        ),
        min_local_flow_samples=_as_int(
            raw.get("min_local_flow_samples", 8), "min_local_flow_samples"
        ),
        mask_padding_px=_as_float(
            raw.get("mask_padding_px", 6.0), "mask_padding_px"
        ),
        ransac_reprojection_threshold_px=_as_float(
            raw.get("ransac_reprojection_threshold_px", 3.0),
            "ransac_reprojection_threshold_px",
        ),
        min_global_inliers=_as_int(
            raw.get("min_global_inliers", 20), "min_global_inliers"
        ),
    )
    config.validate()
    return config


def load_experiment_config(
    path: Path,
    *,
    root: Path = _PROJECT_ROOT,
) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read camera-motion config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Camera-motion experiment config must be a JSON object.")

    trajectory = TrajectoryConfig(
        fps=_as_float(raw.get("fps", 30.0), "fps"),
        max_gap_frames=_as_int(raw.get("max_gap_frames", 15), "max_gap_frames"),
        standing_speed_threshold_px_s=_as_float(
            raw.get("original_standing_speed_threshold_px_s", 10.0),
            "original_standing_speed_threshold_px_s",
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
    trajectory.validate()

    calibration_raw = raw.get("calibration")
    if not isinstance(calibration_raw, list) or not calibration_raw:
        raise ValueError("Config requires a non-empty 'calibration' list.")
    default_max_seconds = _as_float(raw.get("calibration_max_seconds", 60.0), "calibration_max_seconds")
    calibration = [
        _clip_from_mapping(
            root,
            item,
            default_max_seconds=default_max_seconds,
            name=f"calibration[{index}]",
        )
        for index, item in enumerate(calibration_raw)
    ]
    evaluation = _clip_from_mapping(
        root,
        raw.get("evaluation"),
        default_max_seconds=_as_float(
            raw.get("evaluation_max_seconds", 60.0),
            "evaluation_max_seconds",
        ),
        name="evaluation",
    )
    if any(
        clip.set_id == evaluation.set_id and clip.video_id == evaluation.video_id
        for clip in calibration
    ):
        raise ValueError(
            "The evaluation recording cannot also be used for calibration."
        )

    features_csv = _resolve_path(
        root,
        raw.get("evaluation_features_csv"),
        "evaluation_features_csv",
    )
    output_dir = _resolve_path(
        root,
        raw.get("output_dir", "artifacts/trajectory_extraction/camera_motion"),
        "output_dir",
    )
    radius_candidates = [
        _as_float(value, "local_radius_candidates_px")
        for value in raw.get(
            "local_radius_candidates_px", [80.0, 120.0, 160.0, 220.0, 300.0]
        )
    ]
    if not radius_candidates or any(radius <= 0 for radius in radius_candidates):
        raise ValueError("local_radius_candidates_px must contain positive values.")
    iou_threshold = _as_float(raw.get("iou_threshold", 0.5), "iou_threshold")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("Config field 'iou_threshold' must be between 0 and 1.")

    return {
        "trajectory_config": trajectory,
        "camera_config": _camera_config_from_mapping(raw.get("camera_motion", {})),
        "calibration": calibration,
        "evaluation": evaluation,
        "evaluation_features_csv": features_csv,
        "output_dir": output_dir,
        "local_radius_candidates_px": radius_candidates,
        "iou_threshold": iou_threshold,
        "write_raw_features": _as_bool(
            raw.get("write_raw_features", True), "write_raw_features"
        ),
    }


def _load_pie_observations(
    clip: ExperimentClip,
    config: TrajectoryConfig,
) -> tuple[list[TrackObservation], dict[tuple[str, str, str, int], str]]:
    _width, _height, boxes = read_pedestrian_xml(clip.annotation_path)
    max_frames = math.ceil(clip.max_seconds * config.fps)
    observations: list[TrackObservation] = []
    actions: dict[tuple[str, str, str, int], str] = {}
    for box in boxes:
        if box.outside or not box.track_id:
            continue
        if box.frame_index >= max_frames:
            continue
        try:
            observation = TrackObservation(
                track_id=box.track_id,
                frame_index=box.frame_index,
                bbox_xyxy=box.bbox_xyxy,
                timestamp_seconds=box.frame_index / config.fps,
                set_id=clip.set_id,
                video_id=clip.video_id,
                occluded=box.occluded_raw == "1",
                source=clip.annotation_path.as_posix(),
            )
        except ValueError:
            continue
        observations.append(observation)
        action = (box.attributes.get("action") or "").strip().lower()
        if action:
            actions[
                (clip.set_id, clip.video_id, box.track_id, box.frame_index)
            ] = action
    return observations, actions


def _action_for_feature(
    feature: TrajectoryFeature,
    actions: dict[tuple[str, str, str, int], str],
) -> str | None:
    return actions.get(
        (
            feature.set_id or "",
            feature.video_id or "",
            feature.track_id,
            feature.frame_index,
        )
    )


def _speed_samples(
    features: Iterable[TrajectoryFeature],
    actions: dict[tuple[str, str, str, int], str],
) -> list[tuple[str, float | None]]:
    samples: list[tuple[str, float | None]] = []
    for feature in features:
        action = _action_for_feature(feature, actions)
        if (
            action not in {MOVEMENT_WALKING, MOVEMENT_STANDING}
            or not feature.velocity_valid
            or feature.speed_px_s is None
        ):
            continue
        samples.append((action, feature.speed_px_s))
    return samples


def _classification_for_features(
    features: Iterable[TrajectoryFeature],
    actions: dict[tuple[str, str, str, int], str],
) -> dict[str, Any]:
    ground_truth: list[str] = []
    predictions: list[str] = []
    for feature in features:
        action = _action_for_feature(feature, actions)
        state = feature.smoothed_movement_state
        if action in {MOVEMENT_WALKING, MOVEMENT_STANDING} and state in {
            MOVEMENT_WALKING,
            MOVEMENT_STANDING,
        }:
            ground_truth.append(action)
            predictions.append(state)
    return classification_metrics(ground_truth, predictions)


def _merge_actions(
    target: dict[tuple[str, str, str, int], str],
    source: dict[tuple[str, str, str, int], str],
) -> None:
    target.update(source)


def _calibration_payload(
    clips: list[ExperimentClip],
    config: dict[str, Any],
) -> dict[str, Any]:
    trajectory_config: TrajectoryConfig = config["trajectory_config"]
    camera_config: CameraMotionConfig = config["camera_config"]
    loaded: list[
        tuple[ExperimentClip, list[TrackObservation], dict[tuple[str, str, str, int], str]]
    ] = []
    actions: dict[tuple[str, str, str, int], str] = {}
    sequences: dict[tuple[str, str], CameraMotionSequence] = {}
    clip_records: list[dict[str, Any]] = []
    for clip in clips:
        observations, clip_actions = _load_pie_observations(clip, trajectory_config)
        if not observations:
            raise ValueError(
                f"No visible PIE pedestrian observations for {clip.set_id}/"
                f"{clip.video_id} in the calibration window."
            )
        _merge_actions(actions, clip_actions)
        loaded.append((clip, observations, clip_actions))
        sequence = estimate_camera_motion(
            clip.video_path,
            observations,
            camera_config,
            max_frame=max(observation.frame_index for observation in observations) + 1,
        )
        sequences[(clip.set_id, clip.video_id)] = sequence
        clip_records.append(
            {
                "set_id": clip.set_id,
                "video_id": clip.video_id,
                "video": clip.video_path.as_posix(),
                "annotations": clip.annotation_path.as_posix(),
                "max_seconds": clip.max_seconds,
                "observations": len(observations),
                "tracks": len(
                    {
                        (observation.set_id, observation.video_id, observation.track_id)
                        for observation in observations
                    }
                ),
                "camera_motion": sequence.quality_summary(observations),
            }
        )

    baseline_samples: list[tuple[str, float | None]] = []
    baseline_features: list[TrajectoryFeature] = []
    for clip, observations, clip_actions in loaded:
        features = extract_track_features(observations, trajectory_config)
        baseline_features.extend(features)
        baseline_samples.extend(_speed_samples(features, clip_actions))
    baseline_threshold = choose_speed_threshold(baseline_samples)
    calibrated_features: list[TrajectoryFeature] = []
    for clip, observations, _clip_actions in loaded:
        calibrated_features.extend(
            extract_track_features(
                observations,
                replace(
                    trajectory_config,
                    standing_speed_threshold_px_s=baseline_threshold[
                        "threshold_px_s"
                    ],
                ),
            )
        )
    baseline_threshold["smoothed_calibration_metrics"] = (
        _classification_for_features(calibrated_features, actions)
    )

    radius_records: list[dict[str, Any]] = []
    for radius in config["local_radius_candidates_px"]:
        samples: list[tuple[str, float | None]] = []
        for clip, observations, clip_actions in loaded:
            sequence = sequences[(clip.set_id, clip.video_id)].with_local_radius(radius)
            features = extract_track_features(
                observations,
                trajectory_config,
                camera_motion=sequence,
            )
            samples.extend(_speed_samples(features, clip_actions))
        threshold = choose_speed_threshold(samples)
        evaluated_features: list[TrajectoryFeature] = []
        for clip, observations, _clip_actions in loaded:
            sequence = sequences[(clip.set_id, clip.video_id)].with_local_radius(radius)
            evaluated_features.extend(
                extract_track_features(
                    observations,
                    replace(
                        trajectory_config,
                        standing_speed_threshold_px_s=threshold["threshold_px_s"],
                    ),
                    camera_motion=sequence,
                )
            )
        smoothed_metrics = _classification_for_features(
            evaluated_features, actions
        )
        radius_records.append(
            {
                "local_radius_px": radius,
                "threshold_px_s": threshold["threshold_px_s"],
                "raw_threshold_metrics": threshold["metrics"],
                "smoothed_calibration_metrics": smoothed_metrics,
                "speed_samples": len(samples),
            }
        )

    def radius_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
        metrics = record["smoothed_calibration_metrics"]
        radius = float(record["local_radius_px"])
        return (
            metrics["balanced_accuracy"] or 0.0,
            metrics["macro_f1"] or 0.0,
            metrics["accuracy"] or 0.0,
            -abs(radius - 160.0),
        )

    selected = max(radius_records, key=radius_key)
    ego_records = [
        summarize_ego_telemetry(
            clip.ego_path,
            max_frame=math.ceil(clip.max_seconds * trajectory_config.fps),
        )
        for clip in clips
        if clip.ego_path.is_file()
    ]
    return {
        "clips": clip_records,
        "speed_sample_count": len(baseline_samples),
        "baseline_threshold": baseline_threshold,
        "radius_selection": {
            "candidates": radius_records,
            "selected": selected,
        },
        "ego_telemetry": ego_records,
    }


def _comparison_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    action = comparison["walking_standing_action_comparison"]
    metrics = action["metrics"]
    counts = comparison["counts"]
    return {
        "threshold_px_s": None,
        "comparable_matched_observations": action[
            "comparable_matched_observations"
        ],
        "agreeing_observations": action["agreeing_observations"],
        "agreement_ratio": action["agreement_ratio"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"]["matrix"],
        "per_class": metrics["per_class"],
        "ground_truth_action_counts": action["ground_truth_action_counts"],
        "predicted_state_counts": action["predicted_state_counts"],
        "matched_boxes": counts["matched_boxes"],
        "ground_truth_box_match_rate": counts["ground_truth_box_match_rate"],
        "mean_iou_of_matched_boxes": counts["mean_iou_of_matched_boxes"],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _render_report(experiment: dict[str, Any]) -> str:
    trajectory = experiment["trajectory_config"]
    calibration = experiment["calibration"]
    evaluation = experiment["evaluation"]
    selected = calibration["radius_selection"]["selected"]
    lines = [
        "# Camera-Motion Compensation and Threshold Calibration",
        "",
        "This experiment estimates background camera motion from sparse Lucas-Kanade "
        "optical flow with tracked pedestrian boxes masked. It then compares the "
        "original threshold, a separately calibrated baseline, and the compensated "
        "method on the same evaluation recordings, frames, and matched boxes.",
        "",
        "Pixel velocities remain image-space measurements, not real-world speeds.",
        "",
        "## Protocol",
        "",
        f"- Calibration recordings: "
        + ", ".join(
            f"`{clip['set_id']}/{clip['video_id']}`"
            for clip in calibration["clips"]
        ),
        f"- Evaluation recording: "
        f"`{evaluation['clip']['set_id']}/{evaluation['clip']['video_id']}`",
        f"- Calibration and evaluation recordings are disjoint: "
        f"`{experiment['evaluation_protocol']['recordings_disjoint']}`",
        f"- Original standing threshold: "
        f"`{trajectory['standing_speed_threshold_px_s']}` px/s",
        f"- Maximum valid frame gap: `{trajectory['max_gap_frames']}` frames",
        f"- Smoothing: `{trajectory['smoothing_window_observations']}` observations, "
        f"minimum `{trajectory['smoothing_min_observations']}`",
        f"- IoU matching threshold: `{experiment['iou_threshold']}`",
        "",
        "## Calibration",
        "",
        "The local-flow radius and standing threshold were chosen on the calibration "
        "recordings only. Threshold selection maximizes balanced walking/standing "
        "accuracy on raw speeds; the table reports the resulting smoothed features.",
        "",
        "| Local radius px | Threshold px/s | Samples | Accuracy | Balanced accuracy | Macro F1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in calibration["radius_selection"]["candidates"]:
        metrics = record["smoothed_calibration_metrics"]
        lines.append(
            f"| {record['local_radius_px']:.0f} | "
            f"{record['threshold_px_s']:.2f} | "
            f"{record['speed_samples']:,} | "
            f"{_format(metrics['accuracy'])} | "
            f"{_format(metrics['balanced_accuracy'])} | "
            f"{_format(metrics['macro_f1'])} |"
        )
    baseline = calibration["baseline_threshold"]
    baseline_metrics = baseline["smoothed_calibration_metrics"]
    lines.extend(
        [
            "",
            f"- Selected local radius: `{selected['local_radius_px']:.0f}` px",
            f"- Selected compensated threshold: "
            f"`{selected['threshold_px_s']:.2f}` px/s",
            f"- Calibrated baseline threshold: "
            f"`{baseline['threshold_px_s']:.2f}` px/s "
            f"(calibration balanced accuracy {_format(baseline_metrics['balanced_accuracy'])})",
            "",
            "## Evaluation",
            "",
            f"Prediction features were loaded from "
            f"`{evaluation['features_source']}` and compared with "
            f"`{evaluation['clip']['annotations']}`. All variants use the same "
            "observations.",
            "",
            "| Method | Threshold px/s | Comparable | Agreement | Accuracy | Balanced accuracy | Macro F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("original_baseline", "Original baseline"),
        ("calibrated_baseline", "Calibrated baseline"),
        ("compensated", "Camera-motion compensated"),
    ):
        summary = evaluation["methods"][key]
        lines.append(
            f"| {label} | {summary['threshold_px_s']:.2f} | "
            f"{summary['comparable_matched_observations']:,} | "
            f"{_format(summary['agreement_ratio'])} | "
            f"{_format(summary['accuracy'])} | "
            f"{_format(summary['balanced_accuracy'])} | "
            f"{_format(summary['macro_f1'])} |"
        )
    lines.extend(
        [
            "",
            "| Method | Standing precision / recall / F1 | Walking precision / recall / F1 | Confusion matrix `[standing, walking]` |",
            "|---|---|---|---|",
        ]
    )
    for key, label in (
        ("original_baseline", "Original baseline"),
        ("calibrated_baseline", "Calibrated baseline"),
        ("compensated", "Camera-motion compensated"),
    ):
        summary = evaluation["methods"][key]
        standing = summary["per_class"]["standing"]
        walking = summary["per_class"]["walking"]
        lines.append(
            f"| {label} | {_format(standing['precision'])} / "
            f"{_format(standing['recall'])} / {_format(standing['f1'])} | "
            f"{_format(walking['precision'])} / "
            f"{_format(walking['recall'])} / {_format(walking['f1'])} | "
            f"`{summary['confusion_matrix']}` |"
        )
    quality = evaluation["camera_motion_quality"]
    observation_quality = quality.get("observation_compensation", {})
    lines.extend(
        [
            "",
            "## Camera-motion quality",
            "",
            f"- Frame pairs: `{quality['frame_pairs']:,}`",
            f"- Mean background flow points per pair: "
            f"`{_format(quality['mean_flow_point_count'])}`",
            f"- Pairs with a valid global affine fallback: "
            f"`{quality['pairs_with_global_affine']:,}`",
            f"- Observations compensated: "
            f"`{observation_quality.get('applied', 0):,}` / "
            f"`{observation_quality.get('observations_evaluated', 0):,}`",
            f"- Local-only / global-only / mixed: "
            f"`{observation_quality.get('local_only', 0):,}` / "
            f"`{observation_quality.get('global_fallback_only', 0):,}` / "
            f"`{observation_quality.get('mixed', 0):,}`",
            f"- Compensation unavailable: "
            f"`{observation_quality.get('unavailable', 0):,}`",
            "",
            "## Coverage diagnostics",
            "",
            f"- Missing frames within track spans: "
            f"`{evaluation['coverage_diagnostics']['total_missing_frames']:,}`",
            f"- Missing-frame share from gaps above 15 frames: "
            f"`{_format(evaluation['coverage_diagnostics']['long_gap_missing_frame_share'])}`",
            f"- Tracks with long gaps: "
            f"`{evaluation['coverage_diagnostics']['tracks_with_long_gaps']:,}`",
            "",
            "## Ego telemetry",
            "",
            "Ego telemetry is reported for context only. It has no camera intrinsics, "
            "extrinsics, mounting geometry, or road-plane homography, so it is not "
            "used to convert speed into pixel displacement.",
            "",
        ]
    )
    ego = evaluation.get("ego_telemetry")
    if ego:
        gps = ego.get("gps_speed_kmh")
        lines.extend(
            [
                f"- Frames: `{ego['frame_count']:,}`",
                f"- Mean GPS speed: "
                f"`{_format(gps['mean']) if gps else 'n/a'}` km/h",
                f"- Mean absolute heading change per frame: "
                f"`{_format(ego['mean_absolute_heading_change_deg_per_frame'])}` degrees",
                f"- Usable for direct pixel compensation: "
                f"`{ego['usable_for_pixel_compensation']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Caveats",
            "",
            "- Compensation estimates apparent background motion; it does not recover "
            "depth, scale, or real-world pedestrian speed.",
            "- Masked tracked boxes can hide background regions. Local flow needs enough "
            "unmasked features near the pedestrian; otherwise the global affine fallback "
            "is used.",
            "- Threshold and radius should be recalibrated for a different camera, scene, "
            "or recording setup.",
            "- PIE annotations are not exhaustive scene ground truth, and action labels "
            "are a different measurement source from velocity-derived states.",
            "",
            "## Files",
            "",
            "- `camera_motion_experiment.json`: full configuration and results.",
            "- `camera_motion_features.csv` / `.jsonl`: compensated per-observation features.",
            "- `camera_motion_summary.json`: compensated feature summary.",
            "- `comparison_original_baseline.json`",
            "- `comparison_calibrated_baseline.json`",
            "- `comparison_compensated.json`",
            "- `camera_motion_experiment_report.md`: this report.",
            "",
        ]
    )
    return "\n".join(lines)


def _format(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def run_experiment(config_path: Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    trajectory_config: TrajectoryConfig = config["trajectory_config"]
    camera_config: CameraMotionConfig = config["camera_config"]
    calibration = _calibration_payload(config["calibration"], config)
    selected = calibration["radius_selection"]["selected"]
    selected_radius = float(selected["local_radius_px"])
    compensated_threshold = float(selected["threshold_px_s"])
    baseline_threshold = float(
        calibration["baseline_threshold"]["threshold_px_s"]
    )
    evaluation_clip: ExperimentClip = config["evaluation"]

    observations, issues, resolved_format = load_observations(
        config["evaluation_features_csv"],
        trajectory_config,
        input_format="csv",
    )
    if resolved_format != "csv":
        raise ValueError("Evaluation features must be supplied as CSV.")
    if not observations:
        raise ValueError("Evaluation feature CSV contains no observations.")
    if any(issue.severity == "error" for issue in issues):
        raise ValueError(
            "Evaluation feature CSV contained errors; inspect it before running "
            "the experiment."
        )

    max_frame = max(observation.frame_index for observation in observations) + 1
    sequence = estimate_camera_motion(
        evaluation_clip.video_path,
        observations,
        replace(camera_config, local_radius_px=selected_radius),
        max_frame=max_frame,
    )
    original_features = extract_track_features(observations, trajectory_config)
    calibrated_features = extract_track_features(
        observations,
        replace(
            trajectory_config,
            standing_speed_threshold_px_s=baseline_threshold,
        ),
    )
    compensated_config = replace(
        trajectory_config,
        standing_speed_threshold_px_s=compensated_threshold,
    )
    compensated_features = extract_track_features(
        observations,
        compensated_config,
        camera_motion=sequence,
    )

    comparisons = {
        "original_baseline": compare_predictions_to_pie(
            original_features,
            evaluation_clip.annotation_path,
            iou_threshold=config["iou_threshold"],
        ),
        "calibrated_baseline": compare_predictions_to_pie(
            calibrated_features,
            evaluation_clip.annotation_path,
            iou_threshold=config["iou_threshold"],
        ),
        "compensated": compare_predictions_to_pie(
            compensated_features,
            evaluation_clip.annotation_path,
            iou_threshold=config["iou_threshold"],
        ),
    }
    methods: dict[str, Any] = {}
    for key, comparison in comparisons.items():
        summary = _comparison_summary(comparison)
        summary["threshold_px_s"] = (
            trajectory_config.standing_speed_threshold_px_s
            if key == "original_baseline"
            else baseline_threshold
            if key == "calibrated_baseline"
            else compensated_threshold
        )
        methods[key] = summary

    destination = config["output_dir"].resolve()
    destination.mkdir(parents=True, exist_ok=True)
    comparison_paths: dict[str, Path] = {}
    for key, comparison in comparisons.items():
        path = destination / f"comparison_{key}.json"
        _write_json(path, comparison)
        comparison_paths[key] = path

    csv_path = destination / "camera_motion_features.csv"
    jsonl_path = destination / "camera_motion_features.jsonl"
    summary_path = destination / "camera_motion_summary.json"
    if config["write_raw_features"]:
        _write_csv(csv_path, compensated_features)
        _write_jsonl(jsonl_path, compensated_features)
        feature_summary = _build_summary(
            config["evaluation_features_csv"].resolve(),
            "prediction_csv_with_camera_motion",
            compensated_config,
            compensated_features,
            [summary.to_dict() for summary in summarize_tracks(compensated_features)],
            [],
        )
        feature_summary["camera_motion"] = {
            "config": sequence.config.to_dict(),
            "quality": sequence.quality_summary(observations),
        }
        _write_json(summary_path, feature_summary)

    ego_telemetry = (
        summarize_ego_telemetry(
            evaluation_clip.ego_path,
            max_frame=max_frame,
        )
        if evaluation_clip.ego_path.is_file()
        else None
    )
    experiment: dict[str, Any] = {
        "schema_version": 1,
        "experiment_type": "camera_motion_compensation_and_threshold_calibration",
        "evaluation_protocol": {
            "calibration_recordings": [
                {"set_id": clip.set_id, "video_id": clip.video_id}
                for clip in config["calibration"]
            ],
            "evaluation_recording": {
                "set_id": evaluation_clip.set_id,
                "video_id": evaluation_clip.video_id,
            },
            "recordings_disjoint": True,
            "same_evaluation_observations_across_methods": True,
            "same_matched_boxes_across_methods": True,
            "threshold_calibration_source": "PIE walking/standing actions",
            "evaluation_source": "PIE walking/standing actions",
        },
        "trajectory_config": trajectory_config.to_dict(),
        "camera_motion_config": sequence.config.to_dict(),
        "iou_threshold": config["iou_threshold"],
        "calibration": calibration,
        "evaluation": {
            "clip": {
                "set_id": evaluation_clip.set_id,
                "video_id": evaluation_clip.video_id,
                "video": evaluation_clip.video_path.as_posix(),
                "annotations": evaluation_clip.annotation_path.as_posix(),
                "ego_telemetry": (
                    evaluation_clip.ego_path.as_posix()
                    if evaluation_clip.ego_path.is_file()
                    else None
                ),
            },
            "features_source": config["evaluation_features_csv"].as_posix(),
            "observation_count": len(observations),
            "frame_range": {
                "first": min(observation.frame_index for observation in observations),
                "last": max(observation.frame_index for observation in observations),
            },
            "methods": methods,
            "camera_motion_quality": sequence.quality_summary(observations),
            "coverage_diagnostics": coverage_diagnostics(original_features),
            "ego_telemetry": ego_telemetry,
        },
        "outputs": {
            "directory": destination.as_posix(),
            "experiment_json": (destination / "camera_motion_experiment.json").as_posix(),
            "report": (destination / "camera_motion_experiment_report.md").as_posix(),
            "features_csv": csv_path.as_posix() if config["write_raw_features"] else None,
            "features_jsonl": (
                jsonl_path.as_posix() if config["write_raw_features"] else None
            ),
            "features_summary": (
                summary_path.as_posix() if config["write_raw_features"] else None
            ),
            "comparisons": {
                key: path.as_posix() for key, path in comparison_paths.items()
            },
        },
        "notes": [
            "Camera motion is estimated from background optical flow with tracked boxes masked.",
            "Compensation is a heuristic image-space correction, not real-world velocity.",
            "Calibration and evaluation recordings are separate.",
            "The evaluation CSV observations, frames, and box matches are identical across methods.",
        ],
    }
    _write_json(destination / "camera_motion_experiment.json", experiment)
    (destination / "camera_motion_experiment_report.md").write_text(
        _render_report(experiment),
        encoding="utf-8",
    )
    return experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate camera-motion compensation and walking/standing thresholds "
            "on separate PIE recordings, then evaluate on an existing prediction CSV."
        )
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        experiment = run_experiment(args.config)
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"Camera-motion experiment failed: {exc}", file=sys.stderr)
        return 2

    output_dir = experiment["outputs"]["directory"]
    methods = experiment["evaluation"]["methods"]
    selected = experiment["calibration"]["radius_selection"]["selected"]
    print(
        f"Selected local radius {selected['local_radius_px']:.0f} px and "
        f"threshold {selected['threshold_px_s']:.2f} px/s."
    )
    for key, label in (
        ("original_baseline", "Original baseline"),
        ("calibrated_baseline", "Calibrated baseline"),
        ("compensated", "Camera-motion compensated"),
    ):
        summary = methods[key]
        print(
            f"{label}: balanced accuracy "
            f"{_format(summary['balanced_accuracy'])}, "
            f"agreement {_format(summary['agreement_ratio'])}, "
            f"macro F1 {_format(summary['macro_f1'])}"
        )
    print(f"Outputs: {output_dir}")
    print(f"Report: {experiment['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

