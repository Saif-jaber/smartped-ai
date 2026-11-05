from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from ..data.models import ValidationIssue
from ..data.parsing import parse_spatial_xml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "vehicle_tracking.json"


@dataclass(frozen=True)
class VehicleDetection:
    frame_index: int
    timestamp_seconds: float
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    track_id: str | None


@dataclass(frozen=True)
class VehicleAnnotation:
    frame_index: int
    track_id: str
    pie_type: str
    class_name: str | None
    bbox_xyxy: tuple[float, float, float, float] | None
    outside: bool
    occluded: bool | None


def _resolve_path(config_path: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string.")
    path = Path(value)
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    return path.resolve()


def load_config(path: Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read vehicle tracking config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Vehicle tracking config must be a JSON object.")

    classes = raw.get("coco_classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("coco_classes must be a non-empty object.")
    normalized_classes: dict[str, int] = {}
    for name, class_id in classes.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(class_id, int)
            or isinstance(class_id, bool)
            or class_id < 0
        ):
            raise ValueError("Each COCO class needs a name and non-negative integer ID.")
        normalized_classes[name] = class_id

    mappings = raw.get("pie_type_to_coco_class")
    if not isinstance(mappings, dict):
        raise ValueError("pie_type_to_coco_class must be an object.")
    for pie_type, class_name in mappings.items():
        if class_name is not None and (
            not isinstance(class_name, str) or class_name not in normalized_classes
        ):
            raise ValueError(
                f"PIE type {pie_type!r} maps to unsupported COCO class {class_name!r}."
            )

    duration = raw.get("duration_seconds")
    confidence = raw.get("confidence_threshold")
    iou_threshold = raw.get("iou_match_threshold")
    max_gap = raw.get("max_track_gap_frames")
    sample_count = raw.get("visual_sample_count")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_seconds must be finite and greater than zero.")
    for name, value in (
        ("confidence_threshold", confidence),
        ("iou_match_threshold", iou_threshold),
    ):
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one.")
    if not isinstance(max_gap, int) or isinstance(max_gap, bool) or max_gap < 1:
        raise ValueError("max_track_gap_frames must be a positive integer.")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
        raise ValueError("visual_sample_count must be a positive integer.")

    return {
        "config_path": path.resolve(),
        "video_path": _resolve_path(path, raw.get("video_path"), "video_path"),
        "annotation_path": _resolve_path(
            path, raw.get("annotation_path"), "annotation_path"
        ),
        "model_path": _resolve_path(path, raw.get("model_path"), "model_path"),
        "tracker_config_path": _resolve_path(
            path, raw.get("tracker_config_path"), "tracker_config_path"
        ),
        "output_dir": _resolve_path(path, raw.get("output_dir"), "output_dir"),
        "set_id": _required_text(raw.get("set_id"), "set_id"),
        "video_id": _required_text(raw.get("video_id"), "video_id"),
        "duration_seconds": float(duration),
        "confidence_threshold": float(confidence),
        "iou_match_threshold": float(iou_threshold),
        "max_track_gap_frames": max_gap,
        "visual_sample_count": sample_count,
        "coco_classes": normalized_classes,
        "pie_type_to_coco_class": dict(mappings),
        "pie_unmapped_reasons": raw.get("pie_unmapped_reasons", {}),
        "annotation_policy": raw.get("annotation_policy", {}),
    }


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _track_id(value: Any) -> str | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None
    if not math.isfinite(numeric):
        return None
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def parse_result_detections(
    result: Any,
    *,
    frame_index: int,
    fps: float,
    coco_classes: dict[str, int],
) -> list[VehicleDetection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    class_values = _as_list(boxes.cls)
    coordinates = _as_list(boxes.xyxy)
    confidences = _as_list(boxes.conf)
    raw_ids = getattr(boxes, "id", None)
    track_ids = _as_list(raw_ids) if raw_ids is not None else [None] * len(class_values)
    if not (
        len(class_values)
        == len(coordinates)
        == len(confidences)
        == len(track_ids)
    ):
        raise ValueError("Ultralytics result arrays have inconsistent lengths.")

    allowed = {class_id: name for name, class_id in coco_classes.items()}
    detections: list[VehicleDetection] = []
    for class_value, bbox_value, confidence_value, track_value in zip(
        class_values, coordinates, confidences, track_ids
    ):
        class_id = int(class_value)
        class_name = allowed.get(class_id)
        if class_name is None:
            continue
        bbox = tuple(float(value) for value in bbox_value)
        confidence = float(confidence_value)
        if (
            len(bbox) != 4
            or not all(math.isfinite(value) for value in bbox)
            or bbox[0] >= bbox[2]
            or bbox[1] >= bbox[3]
            or not math.isfinite(confidence)
        ):
            continue
        detections.append(
            VehicleDetection(
                frame_index=frame_index,
                timestamp_seconds=frame_index / fps,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox_xyxy=bbox,  # type: ignore[arg-type]
                track_id=_track_id(track_value),
            )
        )
    return detections


def read_vehicle_annotations(
    path: Path,
    *,
    frame_limit: int,
    pie_type_to_coco_class: dict[str, str | None],
) -> list[VehicleAnnotation]:
    issues: list[ValidationIssue] = []
    output = StringIO()
    parse_spatial_xml(
        path,
        path.parent.name,
        path.name.removesuffix("_annt.xml"),
        output,
        issues,
    )
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise ValueError(
            f"Vehicle annotations contain {len(errors)} parsing error(s): "
            f"{errors[0].message}"
        )
    annotations: list[VehicleAnnotation] = []
    for line in output.getvalue().splitlines():
        record = json.loads(line)
        if (
            record["object_category"] != "vehicle"
            or not isinstance(record["frame_index"], int)
            or not 0 <= record["frame_index"] < frame_limit
        ):
            continue
        attributes = record["attributes"]
        pie_type = attributes.get("type", "")
        raw_box = record["bbox_xyxy"]
        bbox: tuple[float, float, float, float] | None = None
        if not record["outside"]:
            try:
                coordinates = tuple(
                    float(raw_box[key]) for key in ("x1", "y1", "x2", "y2")
                )
            except (TypeError, ValueError, KeyError):
                coordinates = ()
            if (
                len(coordinates) == 4
                and all(math.isfinite(value) for value in coordinates)
                and coordinates[0] < coordinates[2]
                and coordinates[1] < coordinates[3]
            ):
                bbox = coordinates  # type: ignore[assignment]
        annotations.append(
            VehicleAnnotation(
                frame_index=record["frame_index"],
                track_id=record.get("track_id") or "",
                pie_type=pie_type,
                class_name=pie_type_to_coco_class.get(pie_type),
                bbox_xyxy=bbox,
                outside=bool(record["outside"]),
                occluded=(
                    record["occluded"] == "1"
                    if record.get("occluded") in {"0", "1"}
                    else None
                ),
            )
        )
    return annotations


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_annotations(
    detections: list[VehicleDetection],
    annotations: list[VehicleAnnotation],
    *,
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detections_by_frame_class: dict[tuple[int, str], list[VehicleDetection]] = defaultdict(list)
    annotations_by_frame_class: dict[tuple[int, str], list[VehicleAnnotation]] = defaultdict(list)
    for detection in detections:
        detections_by_frame_class[(detection.frame_index, detection.class_name)].append(
            detection
        )
    for annotation in annotations:
        if not annotation.outside and annotation.bbox_xyxy is not None and annotation.class_name:
            annotations_by_frame_class[
                (annotation.frame_index, annotation.class_name)
            ].append(annotation)

    rows: list[dict[str, Any]] = []
    matched_ious: list[float] = []
    matched_annotation_count = 0
    for key in sorted(set(detections_by_frame_class) | set(annotations_by_frame_class)):
        predicted = detections_by_frame_class.get(key, [])
        truth = annotations_by_frame_class.get(key, [])
        candidates = sorted(
            (
                (box_iou(detection.bbox_xyxy, annotation.bbox_xyxy), di, ai)
                for di, detection in enumerate(predicted)
                for ai, annotation in enumerate(truth)
            ),
            reverse=True,
        )
        used_detections: set[int] = set()
        used_annotations: set[int] = set()
        assignment: dict[int, tuple[int, float]] = {}
        for iou, detection_index, annotation_index in candidates:
            if iou < iou_threshold:
                break
            if detection_index in used_detections or annotation_index in used_annotations:
                continue
            used_detections.add(detection_index)
            used_annotations.add(annotation_index)
            assignment[detection_index] = (annotation_index, iou)
            annotation = truth[annotation_index]
            matched_annotation_count += 1
            matched_ious.append(iou)
        for index, detection in enumerate(predicted):
            assigned = assignment.get(index)
            annotation = truth[assigned[0]] if assigned is not None else None
            rows.append(
                {
                    "frame_index": detection.frame_index,
                    "timestamp_seconds": detection.timestamp_seconds,
                    "class_name": detection.class_name,
                    "predicted_track_id": detection.track_id or "",
                    "annotation_track_id": annotation.track_id if annotation else "",
                    "iou": assigned[1] if assigned is not None else "",
                    "matched": assigned is not None,
                }
            )

    eligible_count = sum(
        not annotation.outside
        and annotation.bbox_xyxy is not None
        and annotation.class_name is not None
        for annotation in annotations
    )
    return rows, {
        "matching_method": "greedy one-to-one maximum-IoU assignment per frame and mapped class",
        "iou_threshold": iou_threshold,
        "eligible_annotation_boxes": eligible_count,
        "matched_annotation_boxes": matched_annotation_count,
        "annotation_match_rate": (
            matched_annotation_count / eligible_count if eligible_count else None
        ),
        "mean_iou_matched_pairs": (
            sum(matched_ious) / len(matched_ious) if matched_ious else None
        ),
        "matched_pairs": len(matched_ious),
        "unmatched_detections_not_counted_as_false_positives": (
            "PIE vehicle annotations may not be exhaustive."
        ),
    }


def extract_trajectory_rows(
    detections: Iterable[VehicleDetection],
    *,
    fps: float,
    max_track_gap_frames: int,
    set_id: str,
    video_id: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[VehicleDetection]] = defaultdict(list)
    for detection in detections:
        if detection.track_id is not None:
            grouped[(detection.class_name, detection.track_id)].append(detection)
    rows: list[dict[str, Any]] = []
    for (class_name, track_id), track_detections in sorted(grouped.items()):
        ordered = sorted(track_detections, key=lambda detection: detection.frame_index)
        previous: VehicleDetection | None = None
        for observation_index, detection in enumerate(ordered):
            x1, y1, x2, y2 = detection.bbox_xyxy
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            frame_gap: int | None = None
            displacement_x: float | None = None
            displacement_y: float | None = None
            displacement: float | None = None
            image_speed: float | None = None
            gap_exceeded = False
            if previous is not None:
                frame_gap = detection.frame_index - previous.frame_index
                previous_center_x = (previous.bbox_xyxy[0] + previous.bbox_xyxy[2]) / 2.0
                previous_center_y = (previous.bbox_xyxy[1] + previous.bbox_xyxy[3]) / 2.0
                displacement_x = center_x - previous_center_x
                displacement_y = center_y - previous_center_y
                displacement = math.hypot(displacement_x, displacement_y)
                gap_exceeded = frame_gap > max_track_gap_frames
                if frame_gap > 0 and not gap_exceeded:
                    image_speed = displacement / (frame_gap / fps)
            rows.append(
                {
                    "set_id": set_id,
                    "video_id": video_id,
                    "source_kind": "tracker_estimated_vehicle_observation",
                    "class_name": class_name,
                    "track_id": track_id,
                    "observation_index": observation_index,
                    "frame_index": detection.frame_index,
                    "timestamp_seconds": detection.timestamp_seconds,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "previous_frame_index": previous.frame_index if previous else None,
                    "frame_gap": frame_gap,
                    "missing_frames_between_observations": (
                        max(0, frame_gap - 1) if frame_gap is not None else None
                    ),
                    "displacement_x_px": displacement_x,
                    "displacement_y_px": displacement_y,
                    "displacement_px": displacement,
                    "image_speed_px_s": image_speed,
                    "gap_exceeded": gap_exceeded,
                    "confidence": detection.confidence,
                    "identity_is_tracker_estimate": True,
                }
            )
            previous = detection
    return rows


def _read_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def _draw_samples(
    result: Any,
    detections: list[VehicleDetection],
    destination: Path,
) -> None:
    import cv2

    image = result.orig_img.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
        color = (40, 210, 40)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        identity = detection.track_id if detection.track_id is not None else "untracked"
        label = f"{detection.class_name} ID={identity} {detection.confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not write vehicle visual sample: {destination}")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(metrics: dict[str, Any]) -> str:
    prediction = metrics["predictions"]
    annotations = metrics["annotations"]
    matching = metrics["matching"]
    lines = [
        "# Vehicle Detection and Tracking Experiment",
        "",
        "## Protocol",
        "",
        f"- Input: `{metrics['input']['video_path']}`",
        f"- Evaluation interval: frames 0 through {metrics['input']['frame_limit'] - 1} "
        f"({metrics['input']['duration_seconds']:.2f} seconds at "
        f"{metrics['input']['fps']:.3f} FPS).",
        f"- Detector: pretrained `{metrics['input']['model_path']}`; no model training.",
        f"- Tracker: BoT-SORT using the separate `{metrics['input']['tracker_config']}`.",
        "- Detections and tracker IDs are model-generated predictions. PIE XML boxes and IDs are annotations.",
        "- Trajectory displacement and speed are image-space estimates only.",
        "",
        "## Predicted detections and tracks",
        "",
        f"- Vehicle detections: {prediction['detection_count']}",
        f"- Unique predicted vehicle tracks: {prediction['unique_track_count']}",
        f"- Detections without tracker IDs: {prediction['detections_without_track_id']}",
        "",
        "| Predicted class | Detections |",
        "|---|---:|",
    ]
    for class_name, count in sorted(prediction["detection_counts_by_class"].items()):
        lines.append(f"| {class_name} | {count} |")
    lines.extend(
        [
            "",
            "## PIE annotations in the same interval",
            "",
            f"- Vehicle annotation boxes (including outside states): {annotations['box_count_all_states']}",
            f"- Visible boxes with valid geometry: {annotations['visible_valid_box_count']}",
            f"- Unique annotated vehicle tracks: {annotations['unique_track_count']}",
            f"- Visible boxes marked occluded: {annotations['visible_occluded_box_count']}",
            f"- Boxes whose PIE types map to requested COCO classes: {annotations['mapped_visible_box_count']}",
            f"- Unmapped/out-of-scope boxes: {annotations['unmapped_or_out_of_scope_box_count']}",
            f"- Tracker observation gaps: {metrics['trajectory']['gaps_between_observations']}; "
            f"missing frame slots: {metrics['trajectory']['missing_frames_between_observations']}; "
            f"gaps exceeding {metrics['trajectory']['max_track_gap_frames']} frames: "
            f"{metrics['trajectory']['gaps_exceeding_limit']}",
            "",
            "| PIE type | Annotation boxes | Unique tracks | Mapped COCO class |",
            "|---|---:|---:|---|",
        ]
    )
    for pie_type, values in sorted(annotations["by_pie_type"].items()):
        class_name = values["mapped_coco_class"] or "not mapped"
        lines.append(
            f"| {pie_type} | {values['box_count']} | {values['unique_track_count']} | {class_name} |"
        )
    lines.extend(
        [
            "",
            "## Frame-level annotation matching",
            "",
            f"- Method: {matching['matching_method']}.",
            f"- Eligible mapped annotation boxes: {matching['eligible_annotation_boxes']}",
            f"- Matched annotation boxes at IoU >= {matching['iou_threshold']:.2f}: "
            f"{matching['matched_annotation_boxes']}",
            f"- Annotation match rate (matched / eligible mapped annotation boxes): "
            f"{_format_metric(matching['annotation_match_rate'])}",
            f"- Mean IoU over matched pairs: {_format_metric(matching['mean_iou_matched_pairs'])}",
            "- This is annotation recall/coverage, not detection precision: PIE labels are not exhaustive, so unmatched predictions are not treated as false positives.",
            "",
            "## Limitations",
            "",
            "- PIE `bike` is left unmapped; it is not automatically equated with COCO bicycle or motorcycle. PIE `train` is counted but outside the requested COCO class set.",
            "- Occluded boxes are included in matching when valid geometry is present; outside boxes are excluded from matching.",
            "- Moving-camera motion, parallax, occlusion, detector misses, and tracker ID switches affect frame matching and trajectories. Frame-level IoU does not validate track identity.",
            "- Tracker IDs are estimates, not guaranteed persistent ground truth. Gaps are retained explicitly; no missing observations are interpolated.",
            "- Pixel displacement/image speed is not real-world distance or velocity. No collision, near-miss, TTC, or risk labels are created.",
            "- YOLO detections may include visible objects absent from PIE annotations; these are not scored as false positives.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def run_experiment(config_path: Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    for key in ("video_path", "annotation_path", "model_path", "tracker_config_path"):
        if not config[key].is_file():
            raise FileNotFoundError(f"Configured {key} does not exist: {config[key]}")
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Vehicle inference requires OpenCV and Ultralytics in the active environment."
        ) from exc

    capture = cv2.VideoCapture(str(config["video_path"]))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open video: {config['video_path']}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
            raise ValueError("Video reports invalid FPS or frame count.")
    finally:
        capture.release()
    frame_limit = min(total_frames, math.ceil(config["duration_seconds"] * fps))
    if frame_limit <= 0:
        raise ValueError("The configured evaluation interval contains no frames.")

    model = YOLO(str(config["model_path"]))
    model_names = _read_names(model.names)
    for class_name, class_id in config["coco_classes"].items():
        if model_names.get(class_id) != class_name:
            raise ValueError(
                f"Model class ID {class_id} is {model_names.get(class_id)!r}, "
                f"not configured class {class_name!r}."
            )
    class_ids = sorted(config["coco_classes"].values())
    track_options: dict[str, Any] = {
        "source": str(config["video_path"]),
        "stream": True,
        "persist": True,
        "conf": config["confidence_threshold"],
        "classes": class_ids,
        "tracker": str(config["tracker_config_path"]),
        "save": False,
        "show": False,
        "verbose": False,
    }
    detections: list[VehicleDetection] = []
    visual_frames = {
        min(frame_limit - 1, round(index * (frame_limit - 1) / max(1, config["visual_sample_count"] - 1)))
        for index in range(config["visual_sample_count"])
    }
    output_dir: Path = config["output_dir"]
    visual_manifest: list[dict[str, Any]] = []
    for frame_index, result in enumerate(
        islice(model.track(**track_options), frame_limit)
    ):
        current = parse_result_detections(
            result,
            frame_index=frame_index,
            fps=fps,
            coco_classes=config["coco_classes"],
        )
        detections.extend(current)
        if frame_index in visual_frames:
            sample_path = output_dir / "visual_samples" / f"frame_{frame_index:06d}.jpg"
            _draw_samples(result, current, sample_path)
            visual_manifest.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_index / fps,
                    "detection_count": len(current),
                    "path": sample_path.relative_to(output_dir).as_posix(),
                }
            )

    annotations = read_vehicle_annotations(
        config["annotation_path"],
        frame_limit=frame_limit,
        pie_type_to_coco_class=config["pie_type_to_coco_class"],
    )
    match_rows, matching_metrics = match_annotations(
        detections,
        annotations,
        iou_threshold=config["iou_match_threshold"],
    )
    trajectory_rows = extract_trajectory_rows(
        detections,
        fps=fps,
        max_track_gap_frames=config["max_track_gap_frames"],
        set_id=config["set_id"],
        video_id=config["video_id"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_rows = [
        {
            "source_kind": "YOLO11s_detection_plus_BoT-SORT_track",
            "set_id": config["set_id"],
            "video_id": config["video_id"],
            "frame_index": detection.frame_index,
            "timestamp_seconds": detection.timestamp_seconds,
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "track_id": detection.track_id or "",
            "tracked": detection.track_id is not None,
            "confidence": detection.confidence,
            "bbox_x1": detection.bbox_xyxy[0],
            "bbox_y1": detection.bbox_xyxy[1],
            "bbox_x2": detection.bbox_xyxy[2],
            "bbox_y2": detection.bbox_xyxy[3],
        }
        for detection in detections
    ]
    _write_csv(
        output_dir / "vehicle_detections.csv",
        detection_rows,
        [
            "source_kind",
            "set_id",
            "video_id",
            "frame_index",
            "timestamp_seconds",
            "class_id",
            "class_name",
            "track_id",
            "tracked",
            "confidence",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
        ],
    )
    trajectory_fields = list(trajectory_rows[0]) if trajectory_rows else [
        "set_id", "video_id", "source_kind", "class_name", "track_id",
        "observation_index", "frame_index",
        "timestamp_seconds", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "center_x", "center_y", "previous_frame_index", "frame_gap",
        "missing_frames_between_observations", "displacement_x_px",
        "displacement_y_px", "displacement_px", "image_speed_px_s",
        "gap_exceeded", "confidence", "identity_is_tracker_estimate",
    ]
    _write_csv(output_dir / "vehicle_trajectories.csv", trajectory_rows, trajectory_fields)
    _write_csv(
        output_dir / "annotation_matches.csv",
        match_rows,
        [
            "frame_index",
            "timestamp_seconds",
            "class_name",
            "predicted_track_id",
            "annotation_track_id",
            "iou",
            "matched",
        ],
    )

    pie_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    pie_type_track_ids: dict[str, set[str]] = defaultdict(set)
    for annotation in annotations:
        pie_type_counts[annotation.pie_type]["box_count"] += 1
        if annotation.track_id:
            pie_type_track_ids[annotation.pie_type].add(annotation.track_id)
        if annotation.outside:
            pie_type_counts[annotation.pie_type]["outside_box_count"] += 1
        elif annotation.bbox_xyxy is not None:
            pie_type_counts[annotation.pie_type]["visible_valid_box_count"] += 1
            if annotation.occluded:
                pie_type_counts[annotation.pie_type]["visible_occluded_box_count"] += 1
            if annotation.class_name:
                pie_type_counts[annotation.pie_type]["mapped_visible_box_count"] += 1
            else:
                pie_type_counts[annotation.pie_type]["unmapped_box_count"] += 1
    annotation_summary = {
        "box_count_all_states": len(annotations),
        "unique_track_count": len(
            {(annotation.pie_type, annotation.track_id) for annotation in annotations if annotation.track_id}
        ),
        "visible_valid_box_count": sum(
            not annotation.outside and annotation.bbox_xyxy is not None
            for annotation in annotations
        ),
        "visible_occluded_box_count": sum(
            not annotation.outside
            and annotation.bbox_xyxy is not None
            and annotation.occluded is True
            for annotation in annotations
        ),
        "mapped_visible_box_count": sum(
            not annotation.outside
            and annotation.bbox_xyxy is not None
            and annotation.class_name is not None
            for annotation in annotations
        ),
        "unmapped_or_out_of_scope_box_count": sum(
            not annotation.outside
            and annotation.bbox_xyxy is not None
            and annotation.class_name is None
            for annotation in annotations
        ),
        "by_pie_type": {
            pie_type: {
                **dict(counts),
                "unique_track_count": len(pie_type_track_ids[pie_type]),
                "mapped_coco_class": config["pie_type_to_coco_class"].get(pie_type),
                "unmapped_reason": config["pie_unmapped_reasons"].get(pie_type),
            }
            for pie_type, counts in sorted(pie_type_counts.items())
        },
    }
    prediction_summary = {
        "detection_count": len(detections),
        "unique_track_count": len(
            {(detection.class_name, detection.track_id) for detection in detections if detection.track_id}
        ),
        "detections_without_track_id": sum(
            detection.track_id is None for detection in detections
        ),
        "detection_counts_by_class": {
            class_name: sum(
                detection.class_name == class_name for detection in detections
            )
            for class_name in config["coco_classes"]
        },
        "unique_tracks_by_class": {
            class_name: len(
                {
                    detection.track_id
                    for detection in detections
                    if detection.class_name == class_name and detection.track_id
                }
            )
            for class_name in config["coco_classes"]
        },
    }
    metrics = {
        "schema_version": 1,
        "evaluation": "set01/video_0001 first 60 seconds; one held-out experiment clip",
        "input": {
            "video_path": config["video_path"].as_posix(),
            "annotation_path": config["annotation_path"].as_posix(),
            "model_path": config["model_path"].as_posix(),
            "tracker_config": config["tracker_config_path"].as_posix(),
            "fps": fps,
            "source_frame_count": total_frames,
            "frame_limit": frame_limit,
            "duration_seconds": frame_limit / fps,
        },
        "models": {
            "detector": "pretrained YOLO11s (no retraining)",
            "tracker": "BoT-SORT with separate smartped_vehicle_botsort.yaml",
            "allowed_coco_classes": config["coco_classes"],
            "pie_type_to_coco_class": config["pie_type_to_coco_class"],
            "pie_unmapped_reasons": config["pie_unmapped_reasons"],
        },
        "predictions": prediction_summary,
        "annotations": annotation_summary,
        "matching": matching_metrics,
        "trajectory": {
            "observation_rows_with_track_id": len(trajectory_rows),
            "max_track_gap_frames": config["max_track_gap_frames"],
            "gaps_between_observations": sum(
                row["frame_gap"] is not None and row["frame_gap"] > 1
                for row in trajectory_rows
            ),
            "missing_frames_between_observations": sum(
                row["missing_frames_between_observations"] or 0
                for row in trajectory_rows
            ),
            "gaps_exceeding_limit": sum(
                row["gap_exceeded"] for row in trajectory_rows
            ),
            "uninterpolated_gaps": True,
            "coordinate_semantics": "image-space pixels; not real-world distances or velocities",
        },
        "visual_samples": visual_manifest,
        "outputs": {
            "detections": "vehicle_detections.csv",
            "trajectories": "vehicle_trajectories.csv",
            "annotation_matches": "annotation_matches.csv",
            "visual_samples_dir": "visual_samples/",
            "metrics": "vehicle_metrics.json",
            "report": "VEHICLE_TRACKING_REPORT.md",
        },
    }
    (output_dir / "vehicle_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (output_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                key: (value.as_posix() if isinstance(value, Path) else value)
                for key, value in config.items()
                if key != "config_path"
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "VEHICLE_TRACKING_REPORT.md").write_text(
        _render_report(metrics), encoding="utf-8"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect and track PIE vehicles using pretrained YOLO11s and BoT-SORT."
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        metrics = run_experiment(args.config)
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"Vehicle tracking failed: {exc}", file=sys.stderr)
        return 2
    prediction = metrics["predictions"]
    matching = metrics["matching"]
    print(
        f"Processed {prediction['detection_count']} vehicle detections across "
        f"{prediction['unique_track_count']} predicted tracks."
    )
    print(
        f"Annotation match rate: {_format_metric(matching['annotation_match_rate'])}; "
        f"mean matched IoU: {_format_metric(matching['mean_iou_matched_pairs'])}."
    )
    print(f"Report: {metrics['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
