from __future__ import annotations

import json
import gzip
import math
import shutil
import subprocess
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any

from .models import SpatialSummary, ValidationIssue, VideoMetadata
from .parsing import (
    compare_frame_sets,
    frame_set_to_ranges,
    parse_annotated_frames,
    parse_ego_telemetry,
    parse_pedestrian_attributes,
    parse_spatial_xml,
)
from .report import render_validation_report


@dataclass(frozen=True)
class DatasetConfig:
    config_path: Path
    dataset_root: Path
    output_dir: Path
    video_dir: str
    spatial_dir: str
    attributes_dir: str
    telemetry_dir: str
    ffprobe_executable: str
    ffprobe_timeout_seconds: int
    split_assignments: dict[str, list[str]]


def load_config(path: Path) -> DatasetConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read config {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Dataset config must be a JSON object.")
    base = path.resolve().parent
    required_paths = ("dataset_root", "output_dir")
    for key in required_paths:
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ValueError(f"Config field {key!r} must be a non-empty path string.")

    split_assignments = raw.get("split_assignments")
    if not isinstance(split_assignments, dict):
        raise ValueError("Config field 'split_assignments' must be an object.")
    normalised_splits: dict[str, list[str]] = {}
    for split_name in ("train", "validation", "test"):
        values = split_assignments.get(split_name)
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError(f"Split {split_name!r} must be a list of set names.")
        normalised_splits[split_name] = values

    try:
        timeout = int(raw.get("ffprobe_timeout_seconds", 20))
    except (TypeError, ValueError) as exc:
        raise ValueError("ffprobe_timeout_seconds must be an integer.") from exc
    if timeout <= 0:
        raise ValueError("ffprobe_timeout_seconds must be greater than zero.")

    return DatasetConfig(
        config_path=path.resolve(),
        dataset_root=(base / raw["dataset_root"]).resolve(),
        output_dir=(base / raw["output_dir"]).resolve(),
        video_dir=str(raw.get("video_dir", "PIE_clips")),
        spatial_dir=str(raw.get("spatial_dir", "annotations")),
        attributes_dir=str(raw.get("attributes_dir", "annotations_attributes")),
        telemetry_dir=str(raw.get("telemetry_dir", "annotations_vehicle")),
        ffprobe_executable=str(raw.get("ffprobe_executable", "ffprobe")),
        ffprobe_timeout_seconds=timeout,
        split_assignments=normalised_splits,
    )


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def probe_video(
    path: Path, executable: str, timeout_seconds: int
) -> tuple[VideoMetadata | None, str | None]:
    command = shutil.which(executable)
    if command is None:
        return None, f"Executable {executable!r} was not found on PATH."
    try:
        completed = subprocess.run(
            [
                command,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"ffprobe failed for {path.name}: {exc}"

    try:
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        stream = next((item for item in streams if item.get("codec_type", "video") == "video"), None)
        if stream is None:
            return None, "ffprobe returned no video stream."
        width = int(stream["width"]) if stream.get("width") is not None else None
        height = int(stream["height"]) if stream.get("height") is not None else None
        duration = float(stream["duration"]) if stream.get("duration") not in (None, "N/A") else None
        fps: float | None = None
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        if rate and rate != "0/0":
            fps_value = float(Fraction(rate))
            fps = fps_value if math.isfinite(fps_value) and fps_value > 0 else None
        frame_count: int | None = None
        raw_count = stream.get("nb_frames")
        if raw_count not in (None, "N/A"):
            try:
                frame_count = int(raw_count)
            except ValueError:
                frame_count = None
        if frame_count is None and duration is not None and fps is not None:
            frame_count = round(duration * fps)
        if frame_count is not None and frame_count <= 0:
            frame_count = None
        return (
            VideoMetadata(width, height, fps, frame_count, duration, "ffprobe"),
            None,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
        return None, f"Unable to interpret ffprobe output for {path.name}: {exc}"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _expected_paths(config: DatasetConfig, set_id: str, video_id: str) -> dict[str, Path]:
    return {
        "video": config.dataset_root / config.video_dir / set_id / f"{video_id}.mp4",
        "spatial": config.dataset_root
        / config.spatial_dir
        / set_id
        / f"{video_id}_annt.xml",
        "attributes": config.dataset_root
        / config.attributes_dir
        / set_id
        / f"{video_id}_attributes.xml",
        "telemetry": config.dataset_root
        / config.telemetry_dir
        / set_id
        / f"{video_id}_obd.xml",
    }


def _read_frame_csvs(
    config: DatasetConfig, set_ids: set[str], issues: list[ValidationIssue]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for set_id in sorted(set_ids):
        path = (
            config.dataset_root
            / config.spatial_dir
            / set_id
            / f"{set_id}_annotated_frames.csv"
        )
        result[set_id] = parse_annotated_frames(path, set_id, issues)
    return result


def _check_split_assignments(
    assignments: dict[str, list[str]],
    set_ids: set[str],
    issues: list[ValidationIssue],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split_name, values in assignments.items():
        local_seen: set[str] = set()
        for set_id in values:
            if set_id in local_seen:
                issues.append(
                    ValidationIssue(
                        "error",
                        "duplicate_set_in_split",
                        f"{set_id} is repeated in the {split_name} split.",
                    )
                )
            local_seen.add(set_id)
            previous = mapping.get(set_id)
            if previous is not None:
                issues.append(
                    ValidationIssue(
                        "error",
                        "set_in_multiple_splits",
                        f"{set_id} is assigned to both {previous} and {split_name}.",
                    )
                )
            else:
                mapping[set_id] = split_name

    for set_id in sorted(set_ids - mapping.keys()):
        issues.append(
            ValidationIssue(
                "error",
                "unassigned_recording_set",
                f"{set_id} has no train/validation/test assignment.",
            )
        )
    for set_id in sorted(mapping.keys() - set_ids):
        issues.append(
            ValidationIssue(
                "warning",
                "unknown_set_in_split",
                f"Split assignment references set {set_id}, which has no local videos.",
            )
        )
    return mapping


def _check_frame_bounds(
    indices: set[int] | list[int],
    frame_count: int | None,
    issues: list[ValidationIssue],
    path: Path,
    video_id: str,
    source: str,
) -> None:
    if frame_count is None:
        return
    for index in indices:
        if index < 0 or index >= frame_count:
            issues.append(
                ValidationIssue(
                    "error",
                    "frame_index_out_of_bounds",
                    f"{source} frame index {index} is outside video range [0, {frame_count - 1}].",
                    str(path),
                    video_id,
                    index,
                )
            )


def _empty_spatial_summary() -> SpatialSummary:
    return SpatialSummary(None, None, None, set(), set(), {}, {}, {}, {}, {})


def _summary_totals(per_video: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "videos": len(per_video),
        "video_frames": 0,
        "annotated_frames": 0,
        "pedestrian_tracks": 0,
        "vehicle_tracks": 0,
        "pedestrian_boxes": 0,
        "vehicle_boxes": 0,
        "intention_annotations": 0,
        "ego_telemetry_frames": 0,
        "object_boxes_by_category": {},
        "action_labels": {},
        "crossing_labels": {},
        "crossing_outcomes": {},
        "intention_probability": {"count": 0, "mean": None, "median": None, "min": None, "max": None},
        "sets": {},
        "metadata_frame_counts": {},
    }
    probabilities: list[float] = []
    for video in per_video:
        set_id = video["set_id"]
        set_stats = totals["sets"].setdefault(
            set_id,
            {
                "videos": 0,
                "video_frames": 0,
                "annotated_frames": 0,
                "pedestrian_tracks": 0,
                "vehicle_tracks": 0,
                "intention_annotations": 0,
            },
        )
        set_stats["videos"] += 1
        frame_count = video.get("metadata", {}).get("frame_count")
        if frame_count is not None:
            totals["video_frames"] += frame_count
            set_stats["video_frames"] += frame_count
            totals["metadata_frame_counts"][video["metadata"]["source"]] = (
                totals["metadata_frame_counts"].get(video["metadata"]["source"], 0) + 1
            )
        annotated_count = video.get("annotated_frame_count", 0)
        totals["annotated_frames"] += annotated_count
        set_stats["annotated_frames"] += annotated_count
        pedestrian_tracks = video.get("track_counts", {}).get("pedestrian", 0)
        vehicle_tracks = video.get("track_counts", {}).get("vehicle", 0)
        totals["pedestrian_tracks"] += pedestrian_tracks
        totals["vehicle_tracks"] += vehicle_tracks
        set_stats["pedestrian_tracks"] += pedestrian_tracks
        set_stats["vehicle_tracks"] += vehicle_tracks
        totals["pedestrian_boxes"] += video.get("visible_box_counts", {}).get("pedestrian", 0)
        totals["vehicle_boxes"] += video.get("visible_box_counts", {}).get("vehicle", 0)
        for category, count in video.get("visible_box_counts", {}).items():
            totals["object_boxes_by_category"][category] = (
                totals["object_boxes_by_category"].get(category, 0) + count
            )
        for name, count in video.get("action_counts", {}).items():
            totals["action_labels"][name] = totals["action_labels"].get(name, 0) + count
        for name, count in video.get("crossing_label_counts", {}).items():
            totals["crossing_labels"][name] = totals["crossing_labels"].get(name, 0) + count
        for name, count in video.get("crossing_outcome_counts", {}).items():
            totals["crossing_outcomes"][name] = totals["crossing_outcomes"].get(name, 0) + count
        intention_count = video.get("intention_count", 0)
        totals["intention_annotations"] += intention_count
        set_stats["intention_annotations"] += intention_count
        totals["ego_telemetry_frames"] += video.get("ego_telemetry", {}).get("unique_frames", 0)
        probabilities.extend(video.get("intention_probabilities", []))

    if probabilities:
        totals["intention_probability"] = {
            "count": len(probabilities),
            "mean": sum(probabilities) / len(probabilities),
            "median": median(probabilities),
            "min": min(probabilities),
            "max": max(probabilities),
        }
    return totals


def run_pipeline(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if not config.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {config.dataset_root}")
    if _is_within(config.output_dir, config.dataset_root):
        raise ValueError("Output directory must be outside the original dataset directory.")

    issues: list[ValidationIssue] = []
    video_root = config.dataset_root / config.video_dir
    videos = sorted(video_root.glob("set*/video_*.mp4")) if video_root.is_dir() else []
    if not video_root.is_dir():
        issues.append(
            ValidationIssue("error", "missing_video_directory", "Video directory is missing.", str(video_root))
        )
    if not videos:
        issues.append(
            ValidationIssue("error", "no_videos_found", "No PIE MP4 videos were found.", str(video_root))
        )

    set_ids = {video.parent.name for video in videos}
    split_for_set = _check_split_assignments(config.split_assignments, set_ids, issues)
    frame_csvs = _read_frame_csvs(config, set_ids, issues)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {
        "objects": "object_annotations.jsonl.gz",
        "attributes": "pedestrian_attributes.jsonl.gz",
        "telemetry": "ego_telemetry.jsonl.gz",
        "annotated_frames": "annotated_frames.jsonl.gz",
    }
    per_video: list[dict[str, Any]] = []
    manifest_videos: list[dict[str, Any]] = []

    with ExitStack() as stack:
        outputs = {
            name: stack.enter_context(
                gzip.open(
                    config.output_dir / file_name,
                    "wt",
                    encoding="utf-8",
                    newline="\n",
                    compresslevel=6,
                )
            )
            for name, file_name in output_names.items()
        }

        for video_path in videos:
            set_id = video_path.parent.name
            video_id = video_path.stem
            paths = _expected_paths(config, set_id, video_id)
            video_issues_start = len(issues)
            for role in ("spatial", "attributes", "telemetry"):
                if not paths[role].is_file():
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"missing_{role}_annotation",
                            f"Required {role} annotation file is missing.",
                            str(paths[role]),
                            video_id,
                        )
                    )

            metadata, probe_error = probe_video(
                video_path, config.ffprobe_executable, config.ffprobe_timeout_seconds
            )
            if probe_error:
                issues.append(
                    ValidationIssue(
                        "warning", "video_metadata_unavailable", probe_error, str(video_path), video_id
                    )
                )

            spatial = (
                parse_spatial_xml(
                    paths["spatial"], set_id, video_id, outputs["objects"], issues
                )
                if paths["spatial"].is_file()
                else _empty_spatial_summary()
            )
            if metadata is None:
                metadata = VideoMetadata(
                    spatial.image_width,
                    spatial.image_height,
                    None,
                    spatial.task_size,
                    None,
                    "spatial_xml_task_size" if spatial.task_size is not None else "unavailable",
                )
            elif metadata.width is None or metadata.height is None:
                metadata = VideoMetadata(
                    metadata.width or spatial.image_width,
                    metadata.height or spatial.image_height,
                    metadata.fps,
                    metadata.frame_count or spatial.task_size,
                    metadata.duration_seconds,
                    metadata.source
                    if metadata.frame_count is not None
                    else ("spatial_xml_task_size" if spatial.task_size is not None else metadata.source),
                )

            if (
                metadata.frame_count is not None
                and spatial.task_size is not None
                and metadata.frame_count != spatial.task_size
            ):
                issues.append(
                    ValidationIssue(
                        "warning",
                        "video_task_size_mismatch",
                        f"Video metadata reports {metadata.frame_count} frames; spatial XML task size "
                        f"is {spatial.task_size}.",
                        str(paths["spatial"]),
                        video_id,
                    )
                )

            csv_entry = frame_csvs.get(set_id, {}).get(video_id)
            csv_frames = csv_entry.frame_indices if csv_entry else []
            if csv_entry is None:
                issues.append(
                    ValidationIssue(
                        "error",
                        "video_missing_from_frame_csv",
                        f"No annotated-frame CSV row exists for {video_id}.",
                        str(
                            config.dataset_root
                            / config.spatial_dir
                            / set_id
                            / f"{set_id}_annotated_frames.csv"
                        ),
                        video_id,
                    )
                )
            else:
                compare_frame_sets(
                    spatial.visible_frames,
                    set(csv_frames),
                    config.dataset_root
                    / config.spatial_dir
                    / set_id
                    / f"{set_id}_annotated_frames.csv",
                    video_id,
                    issues,
                )

            pedestrian_attribute_ids: set[str] = set()
            crossing_outcomes: dict[str, int] = {}
            intention_probabilities: list[float] = []
            if paths["attributes"].is_file():
                (
                    pedestrian_attribute_ids,
                    crossing_outcomes,
                    intention_probabilities,
                ) = parse_pedestrian_attributes(
                    paths["attributes"],
                    set_id,
                    video_id,
                    outputs["attributes"],
                    issues,
                )
            spatial_pedestrian_ids = spatial.track_ids.get("pedestrian", set())
            for track_id in sorted(spatial_pedestrian_ids - pedestrian_attribute_ids):
                issues.append(
                    ValidationIssue(
                        "warning",
                        "pedestrian_attributes_missing",
                        "Spatial pedestrian track has no matching pedestrian attribute record.",
                        str(paths["attributes"]),
                        video_id,
                        track_id=track_id,
                    )
                )
            for track_id in sorted(pedestrian_attribute_ids - spatial_pedestrian_ids):
                issues.append(
                    ValidationIssue(
                        "warning",
                        "orphan_pedestrian_attributes",
                        "Pedestrian attribute record has no matching spatial pedestrian track.",
                        str(paths["attributes"]),
                        video_id,
                        track_id=track_id,
                    )
                )

            ego_frame_ids: set[int] = set()
            telemetry_summary: dict[str, int] = {
                "records": 0,
                "unique_frames": 0,
                "duplicate_ids": 0,
                "invalid_ids": 0,
            }
            if paths["telemetry"].is_file():
                ego_frame_ids, telemetry_summary = parse_ego_telemetry(
                    paths["telemetry"],
                    set_id,
                    video_id,
                    outputs["telemetry"],
                    issues,
                )

            frame_count = metadata.frame_count
            _check_frame_bounds(
                spatial.all_box_frames,
                frame_count,
                issues,
                paths["spatial"],
                video_id,
                "Spatial annotation",
            )
            _check_frame_bounds(
                csv_frames, frame_count, issues, paths["spatial"], video_id, "CSV"
            )
            _check_frame_bounds(
                ego_frame_ids,
                frame_count,
                issues,
                paths["telemetry"],
                video_id,
                "Ego telemetry",
            )
            if frame_count is not None and telemetry_summary["unique_frames"] != frame_count:
                missing_ego_frames = sorted(set(range(frame_count)) - ego_frame_ids)
                issues.append(
                    ValidationIssue(
                        "warning",
                        "incomplete_ego_telemetry",
                        f"Found {telemetry_summary['unique_frames']} unique ego frame records for "
                        f"a {frame_count}-frame video. Missing frame sample: "
                        f"{missing_ego_frames[:20]}.",
                        str(paths["telemetry"]),
                        video_id,
                    )
                )

            annotated_frames = sorted(set(csv_frames))
            for frame_index in annotated_frames:
                outputs["annotated_frames"].write(
                    json.dumps(
                        {
                            "set_id": set_id,
                            "video_id": video_id,
                            "frame_index": frame_index,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            action_counts = spatial.attribute_counts.get("action", {})
            crossing_label_counts = spatial.attribute_counts.get("cross", {})
            video_summary = {
                "set_id": set_id,
                "video_id": video_id,
                "metadata": metadata.to_dict(),
                "annotated_frame_count": len(annotated_frames),
                "track_counts": spatial.track_counts,
                "unique_track_ids": {
                    label: len(ids) for label, ids in spatial.track_ids.items()
                },
                "box_counts": spatial.box_counts,
                "visible_box_counts": spatial.visible_box_counts,
                "action_counts": action_counts,
                "crossing_label_counts": crossing_label_counts,
                "crossing_outcome_counts": crossing_outcomes,
                "intention_count": len(intention_probabilities),
                "intention_probabilities": intention_probabilities,
                "ego_telemetry": telemetry_summary,
                "frame_csv_xml_difference": {
                    "xml_only": len(spatial.visible_frames - set(csv_frames)),
                    "csv_only": len(set(csv_frames) - spatial.visible_frames),
                },
                "issue_count": len(issues) - video_issues_start,
            }
            per_video.append(video_summary)

            manifest_videos.append(
                {
                    "set_id": set_id,
                    "video_id": video_id,
                    "split": split_for_set.get(set_id),
                    "paths": {
                        role: _relative(path, config.dataset_root)
                        for role, path in paths.items()
                    },
                    "metadata": metadata.to_dict(),
                    "task_size": spatial.task_size,
                    "annotated_frame_count": len(annotated_frames),
                    "annotated_frame_ranges_inclusive": frame_set_to_ranges(annotated_frames),
                    "track_counts": spatial.track_counts,
                    "unique_track_ids": {
                        label: len(ids) for label, ids in spatial.track_ids.items()
                    },
                    "box_counts": spatial.box_counts,
                    "visible_box_counts": spatial.visible_box_counts,
                    "attribute_counts": spatial.attribute_counts,
                    "ego_telemetry": telemetry_summary,
                }
            )

    for set_id, rows in frame_csvs.items():
        known_videos = {video["video_id"] for video in manifest_videos if video["set_id"] == set_id}
        for video_id in sorted(rows.keys() - known_videos):
            issues.append(
                ValidationIssue(
                    "warning",
                    "frame_csv_video_without_video_file",
                    f"Frame CSV lists {video_id}, but no matching local video was found.",
                    str(
                        config.dataset_root
                        / config.spatial_dir
                        / set_id
                        / f"{set_id}_annotated_frames.csv"
                    ),
                    video_id,
                )
            )

    known_video_keys = {(video.parent.name, video.stem) for video in videos}
    annotation_patterns = (
        (config.spatial_dir, "*_annt.xml", "_annt.xml", "spatial"),
        (config.attributes_dir, "*_attributes.xml", "_attributes.xml", "attributes"),
        (config.telemetry_dir, "*_obd.xml", "_obd.xml", "telemetry"),
    )
    for relative_dir, pattern, suffix, role in annotation_patterns:
        root = config.dataset_root / relative_dir
        if not root.is_dir():
            continue
        for annotation_path in root.glob(f"set*/{pattern}"):
            set_id = annotation_path.parent.name
            video_id = annotation_path.name[: -len(suffix)]
            if (set_id, video_id) not in known_video_keys:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "annotation_without_video",
                        f"{role} annotation has no matching local video.",
                        str(annotation_path),
                        video_id,
                    )
                )

    totals = _summary_totals(per_video)
    split_summary: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        selected_sets = [
            set_id for set_id, assigned_split in split_for_set.items() if assigned_split == split_name
        ]
        split_summary[split_name] = {
            "sets": sorted(selected_sets),
            "videos": sum(
                video["set_id"] in selected_sets for video in per_video
            ),
            "video_frames": sum(
                stats["video_frames"] for set_id, stats in totals["sets"].items() if set_id in selected_sets
            ),
            "pedestrian_tracks": sum(
                stats["pedestrian_tracks"]
                for set_id, stats in totals["sets"].items()
                if set_id in selected_sets
            ),
            "vehicle_tracks": sum(
                stats["vehicle_tracks"]
                for set_id, stats in totals["sets"].items()
                if set_id in selected_sets
            ),
        }

    split_manifest = {
        "schema_version": 1,
        "grouping_unit": "recording_set",
        "assignments": {
            split_name: sorted(
                set_id for set_id, assigned_split in split_for_set.items() if assigned_split == split_name
            )
            for split_name in ("train", "validation", "test")
        },
        "statistics": split_summary,
        "limitations": [
            "PIE has only six recording sets, so each validation/test partition contains very few independent groups.",
            "Clip/frame-level random splitting is intentionally not used because clips within a set are continuous footage.",
            "Report grouped cross-validation or per-set results alongside this fixed split; validation and test estimates may have high variance.",
        ],
    }

    manifest = {
        "schema_version": 1,
        "dataset": "PIE",
        "dataset_root": ".",
        "generated_from_config": _relative(config.config_path, config.config_path.parent.parent),
        "record_outputs": output_names,
        "video_count": len(manifest_videos),
        "videos": manifest_videos,
    }
    issue_payload = {
        "schema_version": 1,
        "error_count": sum(issue.severity == "error" for issue in issues),
        "warning_count": sum(issue.severity == "warning" for issue in issues),
        "issues": [issue.to_dict() for issue in issues],
    }
    statistics = {
        "schema_version": 1,
        "totals": totals,
        "split_statistics": split_summary,
    }

    (config.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (config.output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (config.output_dir / "statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (config.output_dir / "validation_issues.json").write_text(
        json.dumps(issue_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    report = render_validation_report(
        totals=totals,
        issues=issues,
        split_manifest=split_manifest,
        output_dir=config.output_dir,
    )
    (config.output_dir / "validation_report.md").write_text(report, encoding="utf-8")

    return {
        "output_dir": config.output_dir,
        "statistics": totals,
        "issues": issues,
        "error_count": issue_payload["error_count"],
        "warning_count": issue_payload["warning_count"],
    }
