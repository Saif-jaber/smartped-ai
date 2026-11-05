from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from ..integration.pipeline import run_integration
from ..interaction.pipeline import run_analysis
from ..trajectory.pipeline import run_extraction
from ..vehicle_tracking.pipeline import run_experiment as run_vehicle_experiment

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL = _PROJECT_ROOT / "yolo11s.pt"
_DEFAULT_PEDESTRIAN_TRACKER = _PROJECT_ROOT / "smartped_botsort.yaml"
_DEFAULT_VEHICLE_TRACKER = _PROJECT_ROOT / "smartped_vehicle_botsort.yaml"
_TRAJECTORY_CONFIG = _PROJECT_ROOT / "configs" / "trajectory_extraction.json"
_VEHICLE_CONFIG = _PROJECT_ROOT / "configs" / "vehicle_tracking.json"
_INTERACTION_CONFIG = _PROJECT_ROOT / "configs" / "interaction_analysis.json"
_INTEGRATION_CONFIG = _PROJECT_ROOT / "configs" / "integration_analysis.json"

_PEDESTRIAN_DETECTION_FIELDS = (
    "source_kind",
    "set_id",
    "video_id",
    "frame_index",
    "timestamp_seconds",
    "class_name",
    "track_id",
    "tracked",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
)


@dataclass(frozen=True)
class VideoInfo:
    source: Path
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float


def validate_runner_inputs(
    source: Path,
    model_path: Path,
    pedestrian_tracker_config: Path,
    vehicle_tracker_config: Path,
    confidence: float,
    duration_seconds: float | None,
    vehicle_model_path: Path | None = None,
    vehicle_confidence: float | None = None,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist or is not a file: {source}")
    if source.suffix.lower() != ".mp4":
        raise ValueError(f"Input must be an MP4 file: {source}")
    for path, name in (
        (model_path, "detector checkpoint"),
        (pedestrian_tracker_config, "pedestrian tracker configuration"),
        (vehicle_tracker_config, "vehicle tracker configuration"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Configured {name} does not exist: {path}")
    if vehicle_model_path is not None and not vehicle_model_path.is_file():
        raise FileNotFoundError(
            f"Configured vehicle detector checkpoint does not exist: {vehicle_model_path}"
        )
    for name, value in (
        ("confidence", confidence),
        ("vehicle_confidence", vehicle_confidence),
    ):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be finite and between 0 and 1.")
    if duration_seconds is not None and (
        not math.isfinite(duration_seconds) or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be finite and greater than zero.")


def inspect_video(source: Path) -> VideoInfo:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video processing requires OpenCV.") from exc
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open input video: {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if (
            not math.isfinite(fps)
            or fps <= 0
            or width <= 0
            or height <= 0
            or frame_count <= 0
        ):
            raise ValueError(
                "Input video reports invalid FPS, dimensions, or frame count."
            )
        return VideoInfo(
            source=source.resolve(),
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_seconds=frame_count / fps,
        )
    finally:
        capture.release()


def compute_frame_limit(
    video: VideoInfo,
    duration_seconds: float | None,
) -> int:
    if duration_seconds is None:
        return video.frame_count
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and greater than zero.")
    return min(video.frame_count, math.ceil(duration_seconds * video.fps))


def _recording_identity(source: Path) -> tuple[str, str]:
    video_id = source.stem
    set_id = source.parent.name
    if not set_id.startswith("set") or not set_id[3:].isdigit():
        set_id = "custom"
    return set_id, video_id


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            return list(reader), list(reader.fieldnames)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read generated CSV {path}: {exc}") from exc


def _write_csv(
    path: Path,
    fields: Iterable[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _make_pedestrian_detection_csv(
    trajectory_path: Path,
    destination: Path,
) -> tuple[int, list[dict[str, str]]]:
    rows, _ = _read_csv(trajectory_path)
    detections = [
        {
            "source_kind": "YOLO11s_detection_plus_BoT-SORT_track",
            "set_id": row["set_id"],
            "video_id": row["video_id"],
            "frame_index": row["frame_index"],
            "timestamp_seconds": row["timestamp_seconds"],
            "class_name": "pedestrian",
            "track_id": row["track_id"],
            "tracked": "true",
            "confidence": row["confidence"],
            "bbox_x1": row["bbox_x1"],
            "bbox_y1": row["bbox_y1"],
            "bbox_x2": row["bbox_x2"],
            "bbox_y2": row["bbox_y2"],
        }
        for row in rows
    ]
    _write_csv(destination, _PEDESTRIAN_DETECTION_FIELDS, detections)
    return len(detections), rows


def _prepare_vehicle_config(
    *,
    source: Path,
    output_dir: Path,
    model_path: Path,
    tracker_config: Path,
    duration_seconds: float,
    confidence: float,
    set_id: str,
    video_id: str,
    empty_annotation_path: Path,
    config_path: Path,
) -> None:
    try:
        raw = json.loads(_VEHICLE_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load vehicle configuration: {exc}") from exc
    raw.update(
        {
            "video_path": str(source.resolve()),
            "model_path": str(model_path.resolve()),
            "tracker_config_path": str(tracker_config.resolve()),
            "output_dir": str(output_dir.resolve()),
            "set_id": set_id,
            "video_id": video_id,
            "duration_seconds": duration_seconds,
            "confidence_threshold": confidence,
            "visual_sample_count": 1,
        }
    )
    candidate_annotation = (
        source.parents[2] / "annotations" / set_id / f"{video_id}_annt.xml"
        if len(source.parents) > 2
        else Path()
    )
    if candidate_annotation.is_file():
        raw["annotation_path"] = str(candidate_annotation.resolve())
    else:
        empty_annotation_path.write_text("<annotations />\n", encoding="utf-8")
        raw["annotation_path"] = str(empty_annotation_path.resolve())
    _write_config(config_path, raw)


def _prepare_interaction_config(
    *,
    pedestrian_trajectory: Path,
    vehicle_detections: Path,
    vehicle_trajectories: Path,
    source: Path,
    output_dir: Path,
    set_id: str,
    video_id: str,
    config_path: Path,
) -> None:
    try:
        raw = json.loads(_INTERACTION_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load interaction configuration: {exc}") from exc
    raw.update(
        {
            "pedestrian_trajectory_path": str(pedestrian_trajectory.resolve()),
            "vehicle_detection_path": str(vehicle_detections.resolve()),
            "vehicle_trajectory_path": str(vehicle_trajectories.resolve()),
            "video_path": str(source.resolve()),
            "output_dir": str(output_dir.resolve()),
            "set_ids": [set_id],
            "video_ids": [video_id],
            "visual_sample_count": 0,
        }
    )
    _write_config(config_path, raw)


def _prepare_integration_config(
    *,
    interaction_dir: Path,
    output_dir: Path,
    config_path: Path,
) -> None:
    try:
        raw = json.loads(_INTEGRATION_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load integration configuration: {exc}") from exc
    for key, value in tuple(raw.items()):
        if key.endswith("_path") and isinstance(value, str) and value:
            path = Path(value)
            if not path.is_absolute():
                raw[key] = str((_INTEGRATION_CONFIG.resolve().parent / path).resolve())
    raw.update(
        {
            "interaction_observations_path": str(
                (interaction_dir / "interaction_observations.csv").resolve()
            ),
            "interaction_summary_path": str(
                (interaction_dir / "interaction_summary.csv").resolve()
            ),
            "interaction_run_config_path": str(
                (interaction_dir / "interaction_run_config.json").resolve()
            ),
            "output_dir": str(output_dir.resolve()),
        }
    )
    _write_config(config_path, raw)


def _group_by_frame(
    rows: Iterable[dict[str, str]],
) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        frame = int(row["frame_index"])
        grouped[frame].append(row)
    return grouped


def _merge_prediction_display_fields(
    interaction_rows: list[dict[str, str]],
    integrated_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    prediction_by_identity = {
        (
            row["set_id"],
            row["video_id"],
            row["frame_index"],
            row["pedestrian_track_id"],
            row["vehicle_track_id"],
        ): row
        for row in integrated_rows
    }
    merged: list[dict[str, str]] = []
    for interaction in interaction_rows:
        identity = (
            interaction["set_id"],
            interaction["video_id"],
            interaction["frame_index"],
            interaction["pedestrian_track_id"],
            interaction["vehicle_track_id"],
        )
        prediction = prediction_by_identity.get(identity)
        row = dict(interaction)
        row["prediction_available"] = (
            prediction.get("prediction_available", "false")
            if prediction is not None
            else "false"
        )
        row["crossing_probability"] = (
            prediction.get("crossing_probability", "")
            if prediction is not None
            else ""
        )
        merged.append(row)
    return merged


def timestamps_aligned(
    frame_index: int,
    pedestrian_timestamp: str | float | None,
    vehicle_timestamp: str | float | None,
    fps: float,
    tolerance_seconds: float = 1e-6,
) -> bool:
    if pedestrian_timestamp is None or vehicle_timestamp is None:
        return False
    try:
        pedestrian_time = float(pedestrian_timestamp)
        vehicle_time = float(vehicle_timestamp)
    except (TypeError, ValueError):
        return False
    expected = frame_index / fps
    return (
        math.isfinite(pedestrian_time)
        and math.isfinite(vehicle_time)
        and abs(pedestrian_time - expected) <= tolerance_seconds
        and abs(vehicle_time - expected) <= tolerance_seconds
    )


def _intersects(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    gap = 3
    return not (
        first[2] + gap <= second[0]
        or second[2] + gap <= first[0]
        or first[3] + gap <= second[1]
        or second[3] + gap <= first[1]
    )


def _fit_text(cv2: Any, text: str, max_width: int, scale: float, thickness: int) -> str:
    if max_width <= 0:
        return ""
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
        return text
    ellipsis = "..."
    fitted = text
    while fitted and cv2.getTextSize(
        fitted + ellipsis,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )[0][0] > max_width:
        fitted = fitted[:-1]
    return fitted + ellipsis if fitted else ""


def _draw_compact_label(
    cv2: Any,
    image: Any,
    text: str,
    *,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    occupied: list[tuple[int, int, int, int]],
    scale: float,
) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    padding_x = max(4, round(6 * scale))
    padding_y = max(2, round(4 * scale))
    thickness = 1
    available_width = max(1, width - 2 * padding_x)
    text = _fit_text(cv2, text, available_width, scale, thickness)
    text_size, baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    label_width = min(width, text_size[0] + padding_x * 2)
    label_height = min(height, text_size[1] + baseline + padding_y * 2)
    x1, y1, x2, y2 = bbox
    x_candidates = (
        x1,
        x2 - label_width,
        x1 - label_width,
        x2,
    )
    y_candidates = (
        y1 - label_height - 2,
        y2 + 2,
        y1 + 2,
        y2 - label_height - 2,
    )
    candidates: list[tuple[int, int, int, int]] = []
    for candidate_y in y_candidates:
        for candidate_x in x_candidates:
            left = min(max(candidate_x, 0), max(0, width - label_width))
            top = min(max(candidate_y, 0), max(0, height - label_height))
            rectangle = (left, top, left + label_width, top + label_height)
            if rectangle not in candidates:
                candidates.append(rectangle)
    free_candidate = next(
        (
            candidate
            for candidate in candidates
            if not any(_intersects(candidate, previous) for previous in occupied)
        ),
        None,
    )
    label_rect = free_candidate or min(
        candidates,
        key=lambda candidate: sum(
            _intersects(candidate, previous) for previous in occupied
        ),
    )
    left, top, right, bottom = label_rect
    overlay = image.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)
    baseline_y = min(
        height - padding_y,
        top + padding_y + text_size[1],
    )
    cv2.putText(
        image,
        text,
        (left + padding_x, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    occupied.append(label_rect)
    return label_rect


def _draw_information_panel(
    cv2: Any,
    image: Any,
    lines: list[str],
    *,
    right_aligned: bool,
    scale: float,
) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    margin = max(8, round(16 * scale))
    padding = max(6, round(10 * scale))
    font_scale = max(0.34, scale * 0.86)
    line_height = max(14, round(25 * scale))
    thickness = 1
    max_panel_width = max(1, width - margin * 2)
    fitted_lines = [
        _fit_text(
            cv2,
            line,
            max_panel_width - padding * 2,
            font_scale,
            thickness,
        )
        for line in lines
    ]
    text_width = max(
        (
            cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )[0][0]
            for line in fitted_lines
        ),
        default=1,
    )
    panel_width = min(max_panel_width, text_width + padding * 2)
    panel_height = min(height, len(fitted_lines) * line_height + padding * 2)
    left = (
        max(margin, width - margin - panel_width)
        if right_aligned
        else margin
    )
    top = margin
    right = min(width - margin, left + panel_width)
    bottom = min(height - margin, top + panel_height)
    overlay = image.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (15, 19, 25), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)
    baseline = top + padding + max(10, round(15 * scale))
    for index, text in enumerate(fitted_lines):
        cv2.putText(
            image,
            text,
            (left + padding, min(height - 2, baseline + index * line_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (235, 239, 244),
            thickness,
            cv2.LINE_AA,
        )
    return left, top, right, bottom


def _select_nearby_pairs(
    rows: Iterable[dict[str, str]],
    pixel_distance_threshold: float,
    limit: int = 3,
) -> list[dict[str, str]]:
    closest_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("proximity_status") != "nearby":
            continue
        try:
            distance = float(row["distance_pixels"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(distance) or distance > pixel_distance_threshold:
            continue
        identity = (row["pedestrian_track_id"], row["vehicle_track_id"])
        previous = closest_by_pair.get(identity)
        if previous is None or distance < float(previous["distance_pixels"]):
            closest_by_pair[identity] = row
    return sorted(
        closest_by_pair.values(),
        key=lambda row: (
            float(row["distance_pixels"]),
            row["pedestrian_track_id"],
            row["vehicle_track_id"],
        ),
    )[: max(0, limit)]


def _draw_legend(
    cv2: Any,
    image: Any,
    *,
    scale: float = 0.48,
) -> tuple[int, int, int, int]:
    return _draw_information_panel(
        cv2,
        image,
        ["P pedestrian   V vehicle   px = image-space separation"],
        right_aligned=False,
        scale=scale,
    )


def _render_annotated_video(
    *,
    source: Path,
    destination: Path,
    video: VideoInfo,
    frame_limit: int,
    pedestrian_rows: list[dict[str, str]],
    vehicle_rows: list[dict[str, str]],
    interaction_rows: list[dict[str, str]],
    pixel_distance_threshold: float,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Annotated video rendering requires OpenCV.") from exc

    pedestrians_by_frame = _group_by_frame(pedestrian_rows)
    vehicles_by_frame = _group_by_frame(vehicle_rows)
    interactions_by_frame = _group_by_frame(interaction_rows)
    capture = cv2.VideoCapture(str(source))
    writer = None
    frames_written = 0
    try:
        if not capture.isOpened():
            raise OSError(f"Could not reopen source video for annotation: {source}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(destination),
            fourcc,
            video.fps,
            (video.width, video.height),
        )
        if not writer.isOpened():
            raise OSError(f"Could not create annotated MP4: {destination}")
        for frame_index in range(frame_limit):
            ok, image = capture.read()
            if not ok or image is None:
                raise OSError(
                    f"Source video ended or failed to decode at frame {frame_index}."
                )
            frame_time = frame_index / video.fps
            scale = min(0.54, max(0.36, video.width / 1920.0 * 0.54))
            label_occupied: list[tuple[int, int, int, int]] = []
            legend_rect = _draw_legend(cv2, image, scale=scale)
            label_occupied.append(legend_rect)
            frame_interactions = interactions_by_frame.get(frame_index, [])
            nearby_pairs = _select_nearby_pairs(
                frame_interactions,
                pixel_distance_threshold,
                limit=max(1, len(frame_interactions)),
            )
            panel_lines: list[str] = []
            if nearby_pairs:
                panel_lines.append("NEAREST NEARBY PAIRS")
                for row in nearby_pairs[:3]:
                    distance = float(row["distance_pixels"])
                    movement = (
                        row.get("movement_status", "indeterminate")
                        .replace("_", " ")
                        .strip()
                        or "indeterminate"
                    )
                    label = (
                        f"P{row['pedestrian_track_id']} / "
                        f"V{row['vehicle_track_id']}  {distance:.0f}px  {movement}"
                    )
                    integrated_prediction = row.get("crossing_probability")
                    prediction_available = (
                        row.get("prediction_available", "").strip().lower() == "true"
                    )
                    if prediction_available:
                        if integrated_prediction in (None, ""):
                            raise ValueError(
                                "An interaction is marked prediction_available without "
                                "a crossing probability."
                            )
                        probability = float(integrated_prediction)
                        if not math.isfinite(probability) or not 0 <= probability <= 1:
                            raise ValueError(
                                "Crossing probability must be finite and between zero and one."
                            )
                        label += f"  crossing {probability:.2f}"
                    else:
                        label += "  crossing unavailable"
                    panel_lines.append(label)
                if len(nearby_pairs) > 3:
                    panel_lines.append(f"+{len(nearby_pairs) - 3} more")
                label_occupied.append(
                    _draw_information_panel(
                        cv2,
                        image,
                        panel_lines,
                        right_aligned=True,
                        scale=scale,
                    )
                )

            pedestrian_boxes: list[tuple[int, int, int, int]] = []
            vehicle_boxes: list[tuple[int, int, int, int]] = []
            for rows, color, prefix, target_boxes, is_pedestrian in (
                (
                    pedestrians_by_frame.get(frame_index, []),
                    (55, 215, 75),
                    "P",
                    pedestrian_boxes,
                    True,
                ),
                (
                    vehicles_by_frame.get(frame_index, []),
                    (255, 120, 25),
                    "V",
                    vehicle_boxes,
                    False,
                ),
            ):
                for row in rows:
                    raw_box = tuple(
                        float(row[key])
                        for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
                    )
                    if not all(math.isfinite(value) for value in raw_box):
                        continue
                    x1 = min(max(int(round(raw_box[0])), 0), video.width - 1)
                    y1 = min(max(int(round(raw_box[1])), 0), video.height - 1)
                    x2 = min(max(int(round(raw_box[2])), 0), video.width - 1)
                    y2 = min(max(int(round(raw_box[3])), 0), video.height - 1)
                    if x1 >= x2 or y1 >= y2:
                        continue
                    bbox = (x1, y1, x2, y2)
                    target_boxes.append(bbox)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    try:
                        confidence = float(row.get("confidence") or 0)
                    except (TypeError, ValueError):
                        confidence = 0.0
                    confidence_text = (
                        f"{confidence:.2f}" if math.isfinite(confidence) else "n/a"
                    )
                    track_id = row.get("track_id") or "?"
                    if is_pedestrian:
                        state = (
                            row.get("smoothed_movement_state")
                            or row.get("movement_state")
                            or ""
                        ).strip().lower()
                        state = "" if state in {"", "unknown", "unavailable", "none"} else state
                        label = f"{prefix}{track_id} {confidence_text}"
                        if state:
                            label += f" {state}"
                    else:
                        vehicle_class = row.get("class_name", "vehicle") or "vehicle"
                        label = f"{prefix}{track_id} {vehicle_class} {confidence_text}"
                    _draw_compact_label(
                        cv2,
                        image,
                        label,
                        bbox=bbox,
                        color=color,
                        occupied=label_occupied,
                        scale=scale,
                    )

            for row in nearby_pairs[:2]:
                start = (
                    min(max(int(round(float(row["pedestrian_footpoint_x_px"]))), 0), video.width - 1),
                    min(max(int(round(float(row["pedestrian_footpoint_y_px"]))), 0), video.height - 1),
                )
                end = (
                    min(max(int(round(float(row["nearest_vehicle_point_x_px"]))), 0), video.width - 1),
                    min(max(int(round(float(row["nearest_vehicle_point_y_px"]))), 0), video.height - 1),
                )
                cv2.line(image, start, end, (0, 80, 255), 2, cv2.LINE_AA)

            cv2.putText(
                image,
                f"{frame_index}  {frame_time:.2f}s",
                (max(8, round(16 * scale)), video.height - max(10, round(18 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.36, scale * 0.86),
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            writer.write(image)
            frames_written += 1
            if (
                frame_index + 1 == frame_limit
                or (frame_index + 1) % max(1, frame_limit // 10) == 0
            ):
                progress(
                    f"Annotated video: {frame_index + 1}/{frame_limit} frames "
                    f"({(frame_index + 1) * 100 / frame_limit:.0f}%)."
                )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frames_written != frame_limit:
        raise RuntimeError(
            f"Annotated video contains {frames_written} frames; expected {frame_limit}."
        )
    return {
        "frames_written": frames_written,
        "fps": video.fps,
        "width": video.width,
        "height": video.height,
        "codec": "mp4v",
    }


def verify_annotated_video(
    path: Path,
    *,
    expected_frames: int,
    expected_fps: float,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video verification requires OpenCV.") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"Annotated MP4 cannot be opened: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count != expected_frames:
            raise ValueError(
                f"Annotated MP4 has {frame_count} frames; expected {expected_frames}."
            )
        if not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=0.05):
            raise ValueError(
                f"Annotated MP4 FPS is {fps}; expected approximately {expected_fps}."
            )
        if (
            expected_width is not None
            and expected_height is not None
            and (width != expected_width or height != expected_height)
        ):
            raise ValueError(
                f"Annotated MP4 resolution is {width}x{height}; expected "
                f"{expected_width}x{expected_height}."
            )
        decoded_frame_count = 0
        for frame_index in range(expected_frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise OSError(
                    f"Annotated MP4 failed to decode frame {frame_index}: {path}"
                )
            decoded_frame_count += 1
        return {
            "opens_and_decodes": True,
            "frame_count": frame_count,
            "decoded_frame_count": decoded_frame_count,
            "fully_decoded": decoded_frame_count == expected_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "first_frame_decodes": True,
            "resolution_matches_source": (
                expected_width is None
                or expected_height is None
                or (width == expected_width and height == expected_height)
            ),
        }
    finally:
        capture.release()


def _runtime_device() -> str:
    try:
        import torch
    except ImportError:
        return "Ultralytics default device selection (torch status unavailable)"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _render_report(metrics: dict[str, Any]) -> str:
    pedestrian = metrics["pedestrian"]
    vehicle = metrics["vehicle"]
    interaction = metrics["interaction"]
    integration = metrics["integration"]
    files = metrics["outputs"]
    warnings_list = metrics["warnings"]
    lines = [
        "# SmartPed AI Video Processing Report",
        "",
        "## Run summary",
        "",
        f"- Input: `{metrics['input']['source']}`",
        f"- Recording identity: `{metrics['input']['set_id']}/{metrics['input']['video_id']}`",
        f"- Source video: {metrics['input']['fps']:.3f} FPS, "
        f"{metrics['input']['width']}x{metrics['input']['height']}, "
        f"{metrics['input']['source_frame_count']:,} frames "
        f"({metrics['input']['source_duration_seconds']:.2f}s).",
        f"- Processed: {metrics['input']['processed_frame_count']:,} frames "
        f"({metrics['input']['processed_duration_seconds']:.2f}s).",
        f"- Annotated video verification: {metrics['annotated_video']['decoded_frame_count']:,}/"
        f"{metrics['annotated_video']['frame_count']:,} frames fully decoded, "
        f"frames, {metrics['annotated_video']['fps']:.3f} FPS, "
        f"{metrics['annotated_video']['width']}x{metrics['annotated_video']['height']}, "
        f"opens and decodes: {metrics['annotated_video']['opens_and_decodes']}.",
        f"- Processing wall time: {metrics['timing']['total_seconds']:.2f}s; "
        f"pipeline throughput: {metrics['timing']['processed_frames_per_second']:.2f} "
        "source frames/s; real-time factor: "
        f"{metrics['timing']['real_time_factor']:.3f}x.",
        f"- Detector execution device: `{metrics['runtime']['device']}` "
        "(Ultralytics selects an available accelerator by default).",
        f"- Pedestrian detector: `{metrics['runtime']['pedestrian_detector_checkpoint']}` "
        f"at confidence {metrics['input']['pedestrian_confidence_threshold']:.6f}; "
        f"vehicle detector: `{metrics['runtime']['vehicle_detector_checkpoint']}` "
        f"at confidence {metrics['input']['vehicle_confidence_threshold']:.6f}.",
        "",
        "## Detection and tracking",
        "",
        f"- Pedestrian tracked observations: {pedestrian['observation_count']:,}; "
        f"unique tracker IDs: {pedestrian['unique_tracks']:,}.",
        f"- Vehicle detections: {vehicle['detection_count']:,}; "
        f"unique class-aware tracks: {vehicle['unique_tracks']:,}.",
        f"- Vehicle classes: `{json.dumps(vehicle['class_counts'], sort_keys=True)}`.",
        "- Detection/trajectory gaps were retained as observed; no positions were interpolated.",
        "",
        "## Interaction and intention integration",
        "",
        f"- Interaction pair observations: {interaction['pair_observations']:,}; "
        f"unique track pairs: {interaction['unique_pairs']:,}.",
        f"- Interactions with valid crossing predictions: "
        f"{integration['available_predictions']:,}; unavailable: "
        f"{integration['unavailable_predictions']:,}.",
        f"- Intention unavailable reason(s): "
        f"`{json.dumps(integration['unavailable_reasons'], sort_keys=True)}`.",
        "- The RF uses annotation-derived PIE features. No tracker-derived RF inference is claimed; unavailable probabilities remain null.",
        "- Pixel distances and displacements are image-space only. No real-world speed, TTC, collision, or risk level is inferred.",
        "",
        "## Warnings",
        "",
    ]
    if warnings_list:
        lines.extend(f"- {warning}" for warning in warnings_list)
    else:
        lines.append("- None.")
    lines.extend(["", "## Output files", ""])
    lines.extend(f"- `{name}`: `{path}`" for name, path in files.items())
    lines.append("")
    return "\n".join(lines)


def render_existing_run(
    source: Path,
    source_run_dir: Path,
    output_dir: Path,
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    source_run_dir = source_run_dir.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if not source_run_dir.is_dir():
        raise FileNotFoundError(f"Existing runner output directory not found: {source_run_dir}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Render output directory is not empty; refusing to overwrite: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    metrics_path = source_run_dir / "processing_metrics.json"
    try:
        prior_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read existing processing metrics: {exc}") from exc
    expected_source = Path(prior_metrics["input"]["source"]).resolve()
    if expected_source != source:
        raise ValueError(
            f"Source video does not match the existing run: {expected_source}"
        )
    video = inspect_video(source)
    frame_limit = int(prior_metrics["input"]["processed_frame_count"])
    if frame_limit <= 0 or frame_limit > video.frame_count:
        raise ValueError(
            f"Existing run frame limit {frame_limit} is invalid for this video."
        )

    pedestrian_rows, _ = _read_csv(source_run_dir / "pedestrian_trajectories.csv")
    vehicle_rows, _ = _read_csv(source_run_dir / "vehicle_detections.csv")
    interaction_rows, _ = _read_csv(source_run_dir / "interaction_observations.csv")
    integrated_rows, _ = _read_csv(source_run_dir / "integrated_interactions.csv")
    display_interactions = _merge_prediction_display_fields(
        interaction_rows,
        integrated_rows,
    )
    destination_video = destination / "annotated_video.mp4"
    progress(
        f"Rendering {frame_limit} frames from saved detections and interactions; "
        "no detector or analysis pipeline will run."
    )
    render_metrics = _render_annotated_video(
        source=source,
        destination=destination_video,
        video=video,
        frame_limit=frame_limit,
        pedestrian_rows=pedestrian_rows,
        vehicle_rows=vehicle_rows,
        interaction_rows=display_interactions,
        pixel_distance_threshold=float(
            prior_metrics["interaction"]["pixel_distance_threshold"]
        ),
        progress=progress,
    )
    verification = verify_annotated_video(
        destination_video,
        expected_frames=frame_limit,
        expected_fps=video.fps,
        expected_width=video.width,
        expected_height=video.height,
    )
    result_metrics = {
        "source_video": str(source),
        "source_run_dir": str(source_run_dir),
        "rendered_video": str(destination_video),
        "frame_count": frame_limit,
        "fps": video.fps,
        "width": video.width,
        "height": video.height,
        "pedestrian_rows": len(pedestrian_rows),
        "vehicle_rows": len(vehicle_rows),
        "interaction_rows": len(interaction_rows),
        "nearby_pixel_distance_threshold": float(
            prior_metrics["interaction"]["pixel_distance_threshold"]
        ),
        "display_rules": {
            "pedestrian_box_color": "green",
            "vehicle_box_color": "blue",
            "interaction_panel": "up to three nearest distinct nearby pairs",
            "unavailable_crossing_prediction_text": "crossing unavailable",
            "risk_classification_displayed": False,
            "measurements": "image-space pixels only",
        },
        "render": render_metrics,
        "verification": verification,
    }
    (destination / "render_metrics.json").write_text(
        json.dumps(result_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    progress(f"Render complete. Output: {destination_video}")
    return {
        "output_dir": destination,
        "annotated_video_path": destination_video,
        "metrics": result_metrics,
    }


def run_video(
    source: Path,
    *,
    output_dir: Path | None = None,
    duration_seconds: float | None = None,
    confidence: float = 0.25,
    model_path: Path = _DEFAULT_MODEL,
    vehicle_model_path: Path | None = None,
    vehicle_confidence: float | None = None,
    pedestrian_tracker_config: Path = _DEFAULT_PEDESTRIAN_TRACKER,
    vehicle_tracker_config: Path = _DEFAULT_VEHICLE_TRACKER,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    vehicle_model_path = (vehicle_model_path or model_path).expanduser().resolve()
    vehicle_confidence = confidence if vehicle_confidence is None else vehicle_confidence
    pedestrian_tracker_config = pedestrian_tracker_config.expanduser().resolve()
    vehicle_tracker_config = vehicle_tracker_config.expanduser().resolve()
    validate_runner_inputs(
        source,
        model_path,
        pedestrian_tracker_config,
        vehicle_tracker_config,
        confidence,
        duration_seconds,
        vehicle_model_path,
        vehicle_confidence,
    )
    video = inspect_video(source)
    frame_limit = compute_frame_limit(video, duration_seconds)
    set_id, video_id = _recording_identity(source)
    if set_id == "custom":
        raise ValueError(
            "A recording under a PIE `setNN` directory is required by the existing "
            "vehicle and interaction pipelines."
        )
    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else _PROJECT_ROOT
        / "artifacts"
        / "video_runs"
        / f"{set_id}_{video_id}_{time.time_ns()}"
    )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    selected_duration = frame_limit / video.fps
    warnings_list: list[str] = []
    if duration_seconds is not None and duration_seconds > video.duration_seconds:
        warnings_list.append(
            f"Requested duration {duration_seconds:.3f}s exceeds video duration "
            f"{video.duration_seconds:.3f}s; processed the full video."
        )
    pie_annotation = (
        source.parents[2] / "annotations" / set_id / f"{video_id}_annt.xml"
        if len(source.parents) > 2
        else None
    )
    if pie_annotation is None or not pie_annotation.is_file():
        warnings_list.append(
            "No corresponding PIE annotation XML was found. The vehicle pipeline "
            "received an empty annotation document only to satisfy its evaluation "
            "input contract; no annotation-based evaluation is meaningful for this run."
        )
    progress(
        f"Input ready: {source.name}, {video.fps:.3f} FPS, "
        f"{video.width}x{video.height}, {video.frame_count} frames; "
        f"processing {frame_limit} frames."
    )

    total_started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="smartped_video_runner_") as temporary:
        work = Path(temporary)
        pedestrian_work = work / "pedestrian"
        vehicle_work = work / "vehicle"
        interaction_work = work / "interaction"
        integration_work = work / "integration"
        for path in (pedestrian_work, vehicle_work, interaction_work, integration_work):
            path.mkdir(parents=True, exist_ok=True)

        progress("Stage 1/5: pedestrian YOLO11s detection, tracking, and trajectories.")
        stage_started = time.perf_counter()
        pedestrian_result = run_extraction(
            source,
            config_path=_TRAJECTORY_CONFIG,
            output_dir=pedestrian_work,
            input_format="video",
            default_set_id=set_id,
            default_video_id=video_id,
            model_path=model_path,
            tracker_config=pedestrian_tracker_config,
            confidence=confidence,
            max_seconds=selected_duration,
        )
        stage_seconds["pedestrian_detection_tracking_and_trajectory"] = (
            time.perf_counter() - stage_started
        )
        pedestrian_detections_path = work / "pedestrian_detections.csv"
        pedestrian_count, pedestrian_rows = _make_pedestrian_detection_csv(
            pedestrian_result["csv_path"],
            pedestrian_detections_path,
        )
        pedestrian_trajectories_path = work / "pedestrian_trajectories.csv"
        shutil.copy2(
            pedestrian_result["csv_path"],
            pedestrian_trajectories_path,
        )
        pedestrian_summary = pedestrian_result["summary"]["counts"]
        progress(
            f"Pedestrians complete: {pedestrian_count} tracked detections, "
            f"{pedestrian_summary['tracks']} tracks."
        )

        progress("Stage 2/5: vehicle YOLO11s detection and separate BoT-SORT tracking.")
        stage_started = time.perf_counter()
        vehicle_config_path = work / "vehicle_config.json"
        _prepare_vehicle_config(
            source=source,
            output_dir=vehicle_work,
            model_path=vehicle_model_path,
            tracker_config=vehicle_tracker_config,
            duration_seconds=selected_duration,
            confidence=vehicle_confidence,
            set_id=set_id,
            video_id=video_id,
            empty_annotation_path=work / "no_pie_annotations.xml",
            config_path=vehicle_config_path,
        )
        vehicle_result = run_vehicle_experiment(vehicle_config_path)
        stage_seconds["vehicle_detection_tracking_and_trajectory"] = (
            time.perf_counter() - stage_started
        )
        vehicle_detections_path = work / "vehicle_detections.csv"
        vehicle_trajectories_path = work / "vehicle_trajectories.csv"
        shutil.copy2(vehicle_work / "vehicle_detections.csv", vehicle_detections_path)
        shutil.copy2(
            vehicle_work / "vehicle_trajectories.csv",
            vehicle_trajectories_path,
        )
        vehicle_rows, _ = _read_csv(vehicle_detections_path)
        vehicle_metrics = vehicle_result
        vehicle_prediction_summary = vehicle_metrics.get("predictions", {})
        progress(
            f"Vehicles complete: {vehicle_prediction_summary.get('detection_count', len(vehicle_rows))} "
            "detections; vehicle tracks retained."
        )

        progress("Stage 3/5: pedestrian–vehicle interaction analysis.")
        stage_started = time.perf_counter()
        interaction_config_path = work / "interaction_config.json"
        _prepare_interaction_config(
            pedestrian_trajectory=pedestrian_result["csv_path"],
            vehicle_detections=vehicle_work / "vehicle_detections.csv",
            vehicle_trajectories=vehicle_work / "vehicle_trajectories.csv",
            source=source,
            output_dir=interaction_work,
            set_id=set_id,
            video_id=video_id,
            config_path=interaction_config_path,
        )
        interaction_result = run_analysis(interaction_config_path)
        stage_seconds["interaction_analysis"] = time.perf_counter() - stage_started
        interaction_observations_path = work / "interaction_observations.csv"
        interaction_summary_path = work / "interaction_summary.csv"
        shutil.copy2(
            interaction_work / "interaction_observations.csv",
            interaction_observations_path,
        )
        shutil.copy2(
            interaction_work / "interaction_summary.csv",
            interaction_summary_path,
        )
        interaction_rows, _ = _read_csv(interaction_observations_path)
        interaction_metrics = interaction_result["metrics"]
        progress(
            f"Interactions complete: "
            f"{interaction_metrics['candidate_pair_observations']} pair observations."
        )

        progress("Stage 4/5: crossing-intention integration (saved model only).")
        stage_started = time.perf_counter()
        integration_config_path = work / "integration_config.json"
        _prepare_integration_config(
            interaction_dir=interaction_work,
            output_dir=integration_work,
            config_path=integration_config_path,
        )
        integration_result = run_integration(integration_config_path)
        stage_seconds["intention_integration"] = time.perf_counter() - stage_started
        integrated_path = work / "integrated_interactions.csv"
        shutil.copy2(
            integration_work / "integrated_interactions.csv",
            integrated_path,
        )
        integrated_rows, _ = _read_csv(integrated_path)
        integration_metrics = integration_result["metrics"]
        progress(
            "Integration complete: "
            f"{integration_metrics['prediction_association']['available_interaction_predictions']} "
            "valid crossing predictions."
        )

        progress("Stage 5/5: rendering annotated MP4.")
        stage_started = time.perf_counter()
        interaction_config_raw = json.loads(
            _INTERACTION_CONFIG.read_text(encoding="utf-8")
        )
        annotated_path = work / "annotated_video.mp4"
        render_metrics = _render_annotated_video(
            source=source,
            destination=annotated_path,
            video=video,
            frame_limit=frame_limit,
            pedestrian_rows=pedestrian_rows,
            vehicle_rows=vehicle_rows,
            interaction_rows=_merge_prediction_display_fields(
                interaction_rows,
                integrated_rows,
            ),
            pixel_distance_threshold=float(
                interaction_config_raw["pixel_distance_threshold"]
            ),
            progress=progress,
        )
        stage_seconds["annotated_video_rendering"] = time.perf_counter() - stage_started
        verification = verify_annotated_video(
            annotated_path,
            expected_frames=frame_limit,
            expected_fps=video.fps,
            expected_width=video.width,
            expected_height=video.height,
        )

        total_seconds = time.perf_counter() - total_started
        pedestrian_track_ids = {
            (row["set_id"], row["video_id"], row["track_id"])
            for row in pedestrian_rows
        }
        vehicle_track_ids = {
            (row["set_id"], row["video_id"], row["class_name"], row["track_id"])
            for row in vehicle_rows
            if row["track_id"]
        }
        class_counts: dict[str, int] = defaultdict(int)
        for row in vehicle_rows:
            class_counts[row["class_name"]] += 1
        unavailable_reasons = integration_metrics["prediction_association"][
            "unavailable_reasons"
        ]
        if integration_metrics["prediction_association"][
            "unavailable_interaction_predictions"
        ]:
            warnings_list.append(
                "Crossing predictions are unavailable: the saved model expects "
                "annotation-derived features and has no verified tracker-ID crosswalk; "
                "probabilities remain null."
            )
        if integration_metrics["saved_model_contract"].get("model_load_warnings"):
            warnings_list.append(
                "The saved Random Forest emitted scikit-learn version compatibility "
                "warnings during unpickling; see integration metrics."
            )
        if not pedestrian_rows:
            warnings_list.append("No pedestrian detections were produced.")
        if not vehicle_rows:
            warnings_list.append("No vehicle detections were produced.")

        processing_metrics = {
            "schema_version": 1,
            "input": {
                "source": str(source),
                "set_id": set_id,
                "video_id": video_id,
                "fps": video.fps,
                "width": video.width,
                "height": video.height,
                "source_frame_count": video.frame_count,
                "source_duration_seconds": video.duration_seconds,
                "processed_frame_count": frame_limit,
                "processed_duration_seconds": frame_limit / video.fps,
                "requested_duration_seconds": duration_seconds,
                "confidence_threshold": confidence,
                "pedestrian_confidence_threshold": confidence,
                "vehicle_confidence_threshold": vehicle_confidence,
            },
            "runtime": {
                "device": _runtime_device(),
                "shared_detector_checkpoint": (
                    str(model_path) if model_path == vehicle_model_path else None
                ),
                "pedestrian_detector_checkpoint": str(model_path),
                "vehicle_detector_checkpoint": str(vehicle_model_path),
                "pedestrian_tracker_config": str(pedestrian_tracker_config),
                "vehicle_tracker_config": str(vehicle_tracker_config),
                "models_loaded_by_existing_pipelines": 2,
                "model_load_note": (
                    "The pedestrian and vehicle pipelines each instantiate their configured "
                    "checkpoint once; they run sequentially, not concurrently."
                ),
            },
            "pedestrian": {
                "observation_count": pedestrian_count,
                "unique_tracks": len(pedestrian_track_ids),
                "trajectory_summary": pedestrian_summary,
            },
            "vehicle": {
                "detection_count": len(vehicle_rows),
                "unique_tracks": len(vehicle_track_ids),
                "class_counts": dict(sorted(class_counts.items())),
                "existing_pipeline_metrics": vehicle_metrics,
            },
            "interaction": {
                "pair_observations": len(interaction_rows),
                "unique_pairs": interaction_metrics["candidate_unique_track_pairs"],
                "valid_movement_estimates": interaction_metrics[
                    "valid_movement_estimates"
                ],
                "invalid_movement_estimates": interaction_metrics[
                    "invalid_movement_estimates"
                ],
                "timestamp_validation": interaction_metrics["timestamp_validation"],
                "missing_observation_statistics": interaction_metrics[
                    "missing_observation_statistics"
                ],
                "pixel_distance_threshold": interaction_metrics["thresholds"][
                    "nearby_pixel_distance_threshold"
                ],
            },
            "integration": {
                "interaction_rows": len(integrated_rows),
                "available_predictions": integration_metrics[
                    "prediction_association"
                ]["available_interaction_predictions"],
                "unavailable_predictions": integration_metrics[
                    "prediction_association"
                ]["unavailable_interaction_predictions"],
                "unavailable_reasons": unavailable_reasons,
                "frozen_model_threshold": integration_metrics[
                    "saved_model_contract"
                ]["validation_threshold"],
                "tracker_feature_compatible": integration_metrics[
                    "feature_compatibility"
                ]["compatible_for_tracker_inference"],
            },
            "annotated_video": {
                **render_metrics,
                **verification,
                "output_path": str(destination / "annotated_video.mp4"),
            },
            "timing": {
                "stage_seconds": stage_seconds,
                "total_seconds": total_seconds,
                "processed_frames_per_second": (
                    frame_limit / total_seconds if total_seconds > 0 else None
                ),
                "real_time_factor": (
                    (frame_limit / video.fps) / total_seconds
                    if total_seconds > 0
                    else None
                ),
            },
            "warnings": warnings_list,
            "outputs": {},
        }
        files_to_publish = (
            annotated_path,
            pedestrian_detections_path,
            vehicle_detections_path,
            pedestrian_trajectories_path,
            vehicle_trajectories_path,
            interaction_observations_path,
            interaction_summary_path,
            integrated_path,
        )
        processing_metrics["outputs"] = {
            path.name: str(destination / path.name)
            for path in files_to_publish
        }
        for path in files_to_publish:
            shutil.copy2(path, destination / path.name)
        metrics_path = destination / "processing_metrics.json"
        metrics_path.write_text(
            json.dumps(processing_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_path = destination / "PROCESSING_REPORT.md"
        report_path.write_text(
            _render_report(processing_metrics),
            encoding="utf-8",
        )
    progress(f"Run complete. Outputs: {destination}")
    return {
        "output_dir": destination,
        "metrics": processing_metrics,
        "report_path": destination / "PROCESSING_REPORT.md",
        "annotated_video_path": destination / "annotated_video.mp4",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SmartPed AI pedestrian/vehicle analysis on an MP4."
    )
    parser.add_argument("--source", type=Path, required=True, help="Input MP4 video.")
    parser.add_argument(
        "--render-from-existing-run",
        type=Path,
        help="Rerender annotations from a prior run directory without rerunning inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New or empty output directory (default: artifacts/video_runs/<recording>_<time>).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Process the first N seconds; omit to process the full video.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help=(
            "Pedestrian confidence threshold (default: 0.25; also used for vehicles "
            "unless --vehicle-confidence is set)."
        ),
    )
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument(
        "--vehicle-model",
        type=Path,
        help="Optional vehicle detector checkpoint (defaults to --model).",
    )
    parser.add_argument(
        "--vehicle-confidence",
        type=float,
        help="Optional vehicle confidence threshold (defaults to --confidence).",
    )
    parser.add_argument(
        "--pedestrian-tracker-config",
        type=Path,
        default=_DEFAULT_PEDESTRIAN_TRACKER,
    )
    parser.add_argument(
        "--vehicle-tracker-config",
        type=Path,
        default=_DEFAULT_VEHICLE_TRACKER,
    )
    args = parser.parse_args(argv)
    try:
        if args.render_from_existing_run is not None:
            if args.output_dir is None:
                raise ValueError(
                    "--output-dir is required with --render-from-existing-run."
                )
            result = render_existing_run(
                args.source,
                args.render_from_existing_run,
                args.output_dir,
            )
            print(f"Annotated video: {result['annotated_video_path']}")
            return 0
        result = run_video(
            args.source,
            output_dir=args.output_dir,
            duration_seconds=args.duration,
            confidence=args.confidence,
            model_path=args.model,
            vehicle_model_path=args.vehicle_model,
            vehicle_confidence=args.vehicle_confidence,
            pedestrian_tracker_config=args.pedestrian_tracker_config,
            vehicle_tracker_config=args.vehicle_tracker_config,
        )
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"Video processing failed: {exc}", file=sys.stderr)
        return 2
    print(f"Report: {result['report_path']}")
    print(f"Annotated video: {result['annotated_video_path']}")
    return 0
