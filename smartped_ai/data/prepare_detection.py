from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import DatasetConfig, load_config
from .parsing import parse_annotated_frames


_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "pie_detection_preparation.json"
_COORDINATES = ("xtl", "ytl", "xbr", "ybr")


@dataclass(frozen=True)
class PedestrianBox:
    frame_index: int
    track_id: str | None
    bbox_xyxy: tuple[float | None, float | None, float | None, float | None]
    bbox_raw_xyxy: tuple[str | None, str | None, str | None, str | None]
    outside: bool
    occluded_raw: str
    keyframe: bool
    attributes: dict[str, str]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_pedestrian_xml(path: Path) -> tuple[int, int, list[PedestrianBox]]:
    width: int | None = None
    height: int | None = None
    current_label: str | None = None
    boxes: list[PedestrianBox] = []
    try:
        for event, element in ET.iterparse(path, events=("start", "end")):
            tag = _local_name(element.tag)
            if event == "start" and tag == "track":
                current_label = element.get("label")
                continue
            if event != "end":
                continue
            if tag == "width" and width is None:
                try:
                    width = int(element.text or "")
                except ValueError:
                    pass
            elif tag == "height" and height is None:
                try:
                    height = int(element.text or "")
                except ValueError:
                    pass
            elif tag == "box" and current_label == "pedestrian":
                try:
                    frame_index = int(element.get("frame", ""))
                except ValueError as exc:
                    raise ValueError(f"{path}: pedestrian box has an invalid frame index.") from exc
                attributes = {
                    child.get("name", ""): (child.text or "").strip()
                    for child in element
                    if _local_name(child.tag) == "attribute" and child.get("name")
                }
                raw_coordinates = tuple(element.get(name) for name in _COORDINATES)
                parsed_coordinates: list[float | None] = []
                for value in raw_coordinates:
                    try:
                        number = float(value) if value is not None else math.nan
                    except ValueError:
                        number = math.nan
                    parsed_coordinates.append(number if math.isfinite(number) else None)
                boxes.append(
                    PedestrianBox(
                        frame_index=frame_index,
                        track_id=attributes.get("id") or None,
                        bbox_xyxy=tuple(parsed_coordinates),  # type: ignore[arg-type]
                        bbox_raw_xyxy=raw_coordinates,  # type: ignore[arg-type]
                        outside=element.get("outside", "0") == "1",
                        occluded_raw=element.get("occluded", "0"),
                        keyframe=element.get("keyframe", "0") == "1",
                        attributes=attributes,
                    )
                )
                element.clear()
            elif tag == "track":
                current_label = None
                element.clear()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Could not read pedestrian annotations from {path}: {exc}") from exc
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError(f"Spatial XML has no valid original image dimensions: {path}")
    return width, height, boxes


def validate_bbox(
    bbox_xyxy: tuple[float | None, float | None, float | None, float | None],
    image_width: int,
    image_height: int,
) -> str | None:
    x1, y1, x2, y2 = bbox_xyxy
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return "missing_or_non_finite_coordinate"
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return "non_finite_coordinate"
    if x1 >= x2 or y1 >= y2:
        return "non_positive_area"
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        return "outside_image_bounds"
    return None


def bbox_to_yolo(
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    reason = validate_bbox(bbox_xyxy, image_width, image_height)
    if reason is not None:
        raise ValueError(f"Invalid bounding box ({reason}): {bbox_xyxy}")
    x1, y1, x2, y2 = bbox_xyxy
    return (
        (x1 + x2) / (2 * image_width),
        (y1 + y2) / (2 * image_height),
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def select_frame_indices(
    boxes: list[PedestrianBox],
    frame_stride: int,
    image_width: int,
    image_height: int,
    preserve_keyframes: bool = False,
) -> tuple[dict[int, set[str]], dict[int, list[PedestrianBox]]]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")
    valid_by_frame: dict[int, list[PedestrianBox]] = defaultdict(list)
    valid_by_track: dict[str, list[PedestrianBox]] = defaultdict(list)
    for box in boxes:
        if box.outside or validate_bbox(box.bbox_xyxy, image_width, image_height):
            continue
        valid_by_frame[box.frame_index].append(box)
        if box.track_id:
            valid_by_track[box.track_id].append(box)

    reasons: dict[int, set[str]] = defaultdict(set)
    for frame_index in valid_by_frame:
        if frame_index % frame_stride == 0:
            reasons[frame_index].add("stride")
    for track_boxes in valid_by_track.values():
        ordered = sorted(track_boxes, key=lambda item: item.frame_index)
        reasons[ordered[0].frame_index].add("track_start")
        reasons[ordered[-1].frame_index].add("track_end")
        previous_state: tuple[Any, ...] | None = None
        for box in ordered:
            state = (
                box.attributes.get("action"),
                box.attributes.get("cross"),
                box.attributes.get("occlusion"),
                box.occluded_raw,
            )
            if previous_state is not None and state != previous_state:
                reasons[box.frame_index].add("label_or_visibility_transition")
            previous_state = state
            if preserve_keyframes and box.keyframe:
                reasons[box.frame_index].add("keyframe")
    return dict(reasons), dict(valid_by_frame)


def _load_preparation_config(path: Path) -> tuple[DatasetConfig, Path, int, bool, int, int]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read preparation config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Preparation config must be a JSON object.")
    dataset_config_name = raw.get("dataset_config", "pie_dataset.json")
    if not isinstance(dataset_config_name, str) or not dataset_config_name:
        raise ValueError("dataset_config must be a non-empty path string.")
    dataset_config = load_config((path.resolve().parent / dataset_config_name).resolve())
    output_name = raw.get("output_dir", "../artifacts/pie_detection_yolo")
    if not isinstance(output_name, str) or not output_name:
        raise ValueError("output_dir must be a non-empty path string.")
    output_dir = (path.resolve().parent / output_name).resolve()
    try:
        frame_stride = int(raw.get("frame_stride", 30))
        visual_sample_count = int(raw.get("visual_sample_count", 24))
        jpeg_quality = int(raw.get("jpeg_quality", 92))
    except (TypeError, ValueError) as exc:
        raise ValueError("Sampling, visual sample, and JPEG quality settings must be integers.") from exc
    preserve_keyframes = raw.get("preserve_keyframes", False)
    if not isinstance(preserve_keyframes, bool):
        raise ValueError("preserve_keyframes must be a boolean.")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")
    if visual_sample_count < 0:
        raise ValueError("visual_sample_count cannot be negative.")
    if not 0 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 0 and 100.")
    if output_dir == dataset_config.dataset_root or dataset_config.dataset_root in output_dir.parents:
        raise ValueError("Detection output must be outside the original dataset directory.")
    return dataset_config, output_dir, frame_stride, preserve_keyframes, visual_sample_count, jpeg_quality


def _split_mapping(config: DatasetConfig, set_ids: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split_name, assigned_sets in config.split_assignments.items():
        for set_id in assigned_sets:
            if set_id in mapping:
                raise ValueError(f"Recording set {set_id} is assigned to multiple splits.")
            mapping[set_id] = split_name
    missing = sorted(set_ids - mapping.keys())
    if missing:
        raise ValueError(f"Recording sets have no split assignment: {', '.join(missing)}")
    return mapping


def _csv_index(config: DatasetConfig, set_ids: set[str]) -> tuple[dict[tuple[str, str], set[int]], list[dict[str, str]]]:
    index: dict[tuple[str, str], set[int]] = {}
    issues: list[dict[str, str]] = []
    for set_id in sorted(set_ids):
        csv_path = config.dataset_root / config.spatial_dir / set_id / f"{set_id}_annotated_frames.csv"
        parse_issues = []
        parsed = parse_annotated_frames(csv_path, set_id, parse_issues)
        for video_id, frames in parsed.items():
            index[(set_id, video_id)] = set(frames.frame_indices)
        issues.extend(issue.to_dict() for issue in parse_issues)
    return index, issues


def _bbox_dict(box: PedestrianBox) -> dict[str, float | None]:
    return dict(zip(("x1", "y1", "x2", "y2"), box.bbox_xyxy))


def _box_status(box: PedestrianBox, image_width: int, image_height: int) -> str | None:
    if box.outside:
        return "outside"
    return validate_bbox(box.bbox_xyxy, image_width, image_height)


def _metadata_record(
    box: PedestrianBox,
    set_id: str,
    video_id: str,
    split: str,
    image_width: int,
    image_height: int,
    selected_frames: set[int],
) -> dict[str, Any]:
    invalid_reason = _box_status(box, image_width, image_height)
    included = invalid_reason is None and box.frame_index in selected_frames
    return {
        "set_id": set_id,
        "video_id": video_id,
        "split": split,
        "frame_index": box.frame_index,
        "track_id": box.track_id,
        "bbox_xyxy": _bbox_dict(box),
        "bbox_raw_xyxy": dict(zip(("xtl", "ytl", "xbr", "ybr"), box.bbox_raw_xyxy)),
        "class_id": 0 if included else None,
        "yolo_xywh": (
            bbox_to_yolo(
                tuple(float(value) for value in box.bbox_xyxy), image_width, image_height
            )
            if included
            else None
        ),
        "action": box.attributes.get("action"),
        "crossing_label": box.attributes.get("cross"),
        "occluded": box.occluded_raw == "1",
        "occluded_raw": box.occluded_raw,
        "occlusion": box.attributes.get("occlusion"),
        "keyframe": box.keyframe,
        "outside": box.outside,
        "attributes": box.attributes,
        "sampled_frame": box.frame_index in selected_frames,
        "label_included": included,
        "exclusion_reason": invalid_reason or (
            None if included else "frame_not_selected_by_sampling"
        ),
    }


def _yolo_label_line(box: PedestrianBox, width: int, height: int) -> str:
    bbox = tuple(float(value) for value in box.bbox_xyxy)
    x_center, y_center, box_width, box_height = bbox_to_yolo(bbox, width, height)
    return "0 " + " ".join(
        f"{value:.8f}" for value in (x_center, y_center, box_width, box_height)
    )


def _visual_roles(boxes: list[PedestrianBox]) -> set[str]:
    roles: set[str] = set()
    if len(boxes) > 1:
        roles.add("multiple_pedestrians")
    for box in boxes:
        action = (box.attributes.get("action") or "").lower()
        crossing = (box.attributes.get("cross") or "").lower()
        if action in {"walking", "standing"}:
            roles.add(action)
        if crossing == "crossing":
            roles.add("crossing")
        elif crossing in {"not-crossing", "not_crossing"}:
            roles.add("not_crossing")
        if box.occluded_raw == "1" or (box.attributes.get("occlusion") or "").lower() not in {
            "",
            "none",
            "0",
            "__undefined__",
        }:
            roles.add("occluded_or_partial")
    return roles


def _select_visual_frames(
    boxes_by_frame: dict[int, list[PedestrianBox]], count: int
) -> dict[int, set[str]]:
    if count == 0 or not boxes_by_frame:
        return {}
    roles_by_frame = {
        frame_index: _visual_roles(boxes)
        for frame_index, boxes in boxes_by_frame.items()
    }
    selected: dict[int, set[str]] = {}
    for role in (
        "walking",
        "standing",
        "crossing",
        "not_crossing",
        "occluded_or_partial",
        "multiple_pedestrians",
    ):
        for frame_index in sorted(roles_by_frame):
            if role in roles_by_frame[frame_index]:
                selected.setdefault(frame_index, set()).add(role)
                break
    remaining = [frame for frame in sorted(boxes_by_frame) if frame not in selected]
    target_count = min(count, len(boxes_by_frame))
    if len(selected) < target_count and remaining:
        needed = target_count - len(selected)
        if needed == 1:
            chosen = [remaining[len(remaining) // 2]]
        else:
            chosen = [
                remaining[round(index * (len(remaining) - 1) / (needed - 1))]
                for index in range(needed)
            ]
        for frame_index in chosen:
            selected.setdefault(frame_index, set()).update(roles_by_frame[frame_index])
    return dict(list(sorted(selected.items()))[:count])


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to decode frames. Install it in the active environment; "
            "this pipeline does not install dependencies."
        ) from exc
    return cv2


def _draw_visual_sample(frame: Any, boxes: list[PedestrianBox], header: str) -> Any:
    cv2 = _require_cv2()
    image = frame.copy()
    width = image.shape[1]
    cv2.rectangle(image, (0, 0), (width, 32), (20, 20, 20), -1)
    cv2.putText(image, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for box in boxes:
        x1, y1, x2, y2 = (int(round(float(value))) for value in box.bbox_xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 40), 2)
        labels = [f"ID={box.track_id or 'missing'}"]
        for name, label in (("action", "action"), ("cross", "cross")):
            value = box.attributes.get(name)
            if value and value not in {"__undefined__", "n/a"}:
                labels.append(f"{label}={value}")
        if box.occluded_raw == "1" or box.attributes.get("occlusion", "").lower() not in {"", "none", "0"}:
            labels.append(f"occluded={box.attributes.get('occlusion', box.occluded_raw)}")
        text = " | ".join(labels)
        text_y = max(48, y1 - 5)
        cv2.putText(image, text, (max(0, x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 220, 40), 1, cv2.LINE_AA)
    return image


def _write_image(cv2: Any, path: Path, image: Any, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"Could not write image: {path}")


def _prepare_video(
    config: DatasetConfig,
    output_dir: Path,
    set_id: str,
    video_id: str,
    split: str,
    csv_frames: set[int] | None,
    frame_stride: int,
    preserve_keyframes: bool,
    visual_sample_count: int,
    jpeg_quality: int,
    metadata_output: Any,
    summary: dict[str, Any],
) -> dict[str, Any]:
    annotation_path = (
        config.dataset_root / config.spatial_dir / set_id / f"{video_id}_annt.xml"
    )
    video_path = config.dataset_root / config.video_dir / set_id / f"{video_id}.mp4"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing spatial annotation XML: {annotation_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing source video: {video_path}")

    width, height, boxes = read_pedestrian_xml(annotation_path)
    visible_frames = {box.frame_index for box in boxes if not box.outside}
    valid_reasons, valid_by_frame = select_frame_indices(
        boxes, frame_stride, width, height, preserve_keyframes
    )
    selected_frames = set(valid_reasons)
    sampleable_frames = {
        frame_index: frame_boxes
        for frame_index, frame_boxes in valid_by_frame.items()
        if frame_index in selected_frames
    }
    visual_roles = _select_visual_frames(sampleable_frames, visual_sample_count)
    xml_only = sorted(visible_frames - (csv_frames or set()))
    csv_only = sorted((csv_frames or set()) - visible_frames) if csv_frames is not None else []
    cv2 = _require_cv2()

    label_count = 0
    capture = cv2.VideoCapture(str(video_path)) if selected_frames else None
    if capture is not None:
        try:
            if not capture.isOpened():
                raise OSError(f"Could not open source video: {video_path}")
            for frame_index in sorted(selected_frames):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success or frame is None:
                    raise OSError(f"Could not decode frame {frame_index} from {video_path}")
                actual_height, actual_width = frame.shape[:2]
                if (actual_width, actual_height) != (width, height):
                    raise ValueError(
                        f"XML/video dimensions differ for {set_id}/{video_id}: "
                        f"{width}x{height} versus {actual_width}x{actual_height}."
                    )
                relative = Path("images") / split / set_id / video_id / f"frame_{frame_index:06d}.jpg"
                image_path = output_dir / relative
                _write_image(cv2, image_path, frame, jpeg_quality)
                label_path = (
                    output_dir
                    / "labels"
                    / split
                    / set_id
                    / video_id
                    / f"frame_{frame_index:06d}.txt"
                )
                label_path.parent.mkdir(parents=True, exist_ok=True)
                valid_boxes = valid_by_frame[frame_index]
                label_path.write_text(
                    "\n".join(_yolo_label_line(box, width, height) for box in valid_boxes) + "\n",
                    encoding="utf-8",
                )
                label_count += len(valid_boxes)
                if frame_index in visual_roles:
                    rendered = _draw_visual_sample(
                        frame,
                        valid_boxes,
                        f"{set_id} | {video_id} | frame {frame_index}",
                    )
                    visual_path = (
                        output_dir
                        / "visual_samples"
                        / set_id
                        / video_id
                        / f"frame_{frame_index:06d}.jpg"
                    )
                    _write_image(cv2, visual_path, rendered, jpeg_quality)
        finally:
            capture.release()

    invalid_counts: Counter[str] = Counter()
    outside_count = 0
    for box in boxes:
        status = _box_status(box, width, height)
        if status == "outside":
            outside_count += 1
        elif status is not None:
            invalid_counts[status] += 1
        metadata_output.write(
            json.dumps(
                _metadata_record(
                    box,
                    set_id,
                    video_id,
                    split,
                    width,
                    height,
                    selected_frames,
                ),
                ensure_ascii=True,
                separators=(",", ":"),
            )
            + "\n"
        )

    comparison = {
        "set_id": set_id,
        "video_id": video_id,
        "split": split,
        "csv_present": csv_frames is not None,
        "xml_visible_pedestrian_frame_count": len(visible_frames),
        "csv_annotated_frame_count": len(csv_frames) if csv_frames is not None else None,
        "xml_only_frames": xml_only,
        "csv_only_frames": csv_only,
    }
    summary["outside_boxes_excluded"] += outside_count
    summary["invalid_boxes_excluded"] += sum(invalid_counts.values())
    summary["invalid_box_reasons"].update(invalid_counts)
    summary["visible_xml_pedestrian_frames"] += len(visible_frames)
    summary["sampled_frames"] += len(selected_frames)
    summary["stride_selected_frames"] += sum(
        "stride" in reasons for reasons in valid_reasons.values()
    )
    summary["important_frames"] += sum(
        bool(reasons - {"stride"}) for reasons in valid_reasons.values()
    )
    summary["keyframe_boxes"] += sum(box.keyframe and not box.outside for box in boxes)
    summary["annotation_boxes"] += len(boxes)
    summary["images"] += len(selected_frames)
    summary["labels"] += label_count
    summary["split_distribution"][split]["images"] += len(selected_frames)
    summary["split_distribution"][split]["labels"] += label_count
    summary["class_distribution"]["pedestrian"] += label_count
    return {
        **comparison,
        "annotation_boxes": len(boxes),
        "valid_visible_boxes": sum(len(frame_boxes) for frame_boxes in valid_by_frame.values()),
        "outside_boxes_excluded": outside_count,
        "invalid_boxes_excluded": sum(invalid_counts.values()),
        "invalid_box_reasons": dict(invalid_counts),
        "sampled_frames": len(selected_frames),
        "sampled_stride": frame_stride,
        "frame_selection_reasons": {
            str(frame_index): sorted(reasons)
            for frame_index, reasons in sorted(valid_reasons.items())
        },
        "xml_only_frame_count": len(xml_only),
        "csv_only_frame_count": len(csv_only),
        "label_count": label_count,
        "visual_samples": [
            {
                "frame_index": frame_index,
                "roles": sorted(roles),
                "path": (
                    Path("visual_samples")
                    / set_id
                    / video_id
                    / f"frame_{frame_index:06d}.jpg"
                ).as_posix(),
            }
            for frame_index, roles in sorted(visual_roles.items())
        ],
    }


def run_preparation(
    config_path: Path = _DEFAULT_CONFIG,
    frame_stride_override: int | None = None,
) -> dict[str, Any]:
    config, output_dir, frame_stride, preserve_keyframes, visual_sample_count, jpeg_quality = (
        _load_preparation_config(config_path)
    )
    if frame_stride_override is not None:
        if frame_stride_override < 1:
            raise ValueError("--frame-stride must be at least 1.")
        frame_stride = frame_stride_override

    video_root = config.dataset_root / config.video_dir
    videos = sorted(video_root.glob("set*/*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No PIE videos found under {video_root}")
    set_ids = {video.parent.name for video in videos}
    split_for_set = _split_mapping(config, set_ids)
    csv_index, csv_issues = _csv_index(config, set_ids)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "PIE",
        "class_names": ["pedestrian"],
        "class_distribution": {"pedestrian": 0},
        "images": 0,
        "labels": 0,
        "annotation_boxes": 0,
        "visible_xml_pedestrian_frames": 0,
        "sampled_frames": 0,
        "stride_selected_frames": 0,
        "important_frames": 0,
        "keyframe_boxes": 0,
        "outside_boxes_excluded": 0,
        "invalid_boxes_excluded": 0,
        "invalid_box_reasons": Counter(),
        "split_distribution": {
            name: {"sets": sorted(set_ids & set(config.split_assignments[name])), "images": 0, "labels": 0}
            for name in ("train", "validation", "test")
        },
        "sampling": {
            "frame_stride": frame_stride,
            "phase": "frame_index modulo stride within each video",
            "preserve_keyframes": preserve_keyframes,
            "preserved_frame_events": [
                "track start and end",
                "action, crossing, or occlusion-state transitions",
            ]
            + (["keyframes"] if preserve_keyframes else []),
            "keyframe_note": (
                "Keyframes are extracted in addition to the stride."
                if preserve_keyframes
                else "Keyframe flags are preserved in metadata; all keyframes are not forced into the sparse image sample."
            ),
        },
        "split_strategy": {
            "grouping_unit": "recording_set",
            "assignments": {
                name: sorted(set_ids & set(config.split_assignments[name]))
                for name in ("train", "validation", "test")
            },
            "limitations": [
                "Only six PIE recording sets are available, so validation and test have few independent groups.",
                "No footage from a recording set is split across train, validation, and test.",
                "PIE annotations target selected interactions and are not exhaustive scene labels.",
            ],
        },
        "csv_issues": csv_issues,
    }

    videos_summary: list[dict[str, Any]] = []
    crosschecks: list[dict[str, Any]] = []
    metadata_path = output_dir / "annotation_metadata.jsonl.gz"
    with gzip.open(metadata_path, "wt", encoding="utf-8", newline="") as metadata_output:
        for video_path in videos:
            set_id = video_path.parent.name
            video_id = video_path.stem
            entry = _prepare_video(
                config,
                output_dir,
                set_id,
                video_id,
                split_for_set[set_id],
                csv_index.get((set_id, video_id)),
                frame_stride,
                preserve_keyframes,
                visual_sample_count,
                jpeg_quality,
                metadata_output,
                summary,
            )
            videos_summary.append(entry)
            crosschecks.append(
                {
                    key: entry[key]
                    for key in (
                        "set_id",
                        "video_id",
                        "csv_present",
                        "xml_visible_pedestrian_frame_count",
                        "csv_annotated_frame_count",
                        "xml_only_frames",
                        "csv_only_frames",
                    )
                }
            )

    summary["invalid_box_reasons"] = dict(summary["invalid_box_reasons"])
    summary["csv_crosscheck"] = {
        "video_count": len(crosschecks),
        "videos_with_xml_only_frames": sum(bool(item["xml_only_frames"]) for item in crosschecks),
        "videos_with_csv_only_frames": sum(bool(item["csv_only_frames"]) for item in crosschecks),
        "xml_only_frame_count": sum(len(item["xml_only_frames"]) for item in crosschecks),
        "csv_only_frame_count": sum(len(item["csv_only_frames"]) for item in crosschecks),
        "csv_issues": csv_issues,
    }
    summary["videos"] = videos_summary
    (output_dir / "csv_crosscheck.json").write_text(
        json.dumps(crosschecks, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    split_manifest = {
        **summary["split_strategy"],
        "frame_sampling": summary["sampling"],
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (output_dir / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/validation\ntest: images/test\n"
        "nc: 1\nnames:\n  0: pedestrian\n",
        encoding="utf-8",
    )
    visual_manifest = [
        sample
        for entry in videos_summary
        for sample in entry["visual_samples"]
    ]
    summary["visual_sample_count"] = len(visual_manifest)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (output_dir / "visual_sample_manifest.json").write_text(
        json.dumps(visual_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir, summary)
    return {"output_dir": output_dir, "summary": summary}


def _write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PIE pedestrian detection preparation",
        "",
        "This is a sparse, reproducible YOLO-format preparation of visible PIE pedestrian boxes. "
        "It does not train a model, modify source files, add unannotated people as negatives, "
        "or convert crossing/action labels into detector classes.",
        "",
        "## Results",
        "",
        f"- Images: {summary['images']}",
        f"- Pedestrian labels: {summary['labels']}",
        f"- Source pedestrian annotation boxes: {summary['annotation_boxes']}",
        f"- Outside boxes excluded: {summary['outside_boxes_excluded']}",
        f"- Invalid boxes excluded: {summary['invalid_boxes_excluded']}",
        f"- Visible XML pedestrian frames: {summary['visible_xml_pedestrian_frames']}",
        f"- Sampled frames: {summary['sampled_frames']}",
        f"- Visual samples: {summary.get('visual_sample_count', 0)}",
        f"- CSV cross-check: {summary['csv_crosscheck']['xml_only_frame_count']} XML-only and "
        f"{summary['csv_crosscheck']['csv_only_frame_count']} CSV-only visible-pedestrian frames.",
        "",
        "## Sampling and splits",
        "",
        f"- Frame stride: {summary['sampling']['frame_stride']} (frame index modulo stride, per video).",
        "- All visible track starts/ends and action, crossing, or occlusion-state transitions are retained.",
        f"- {summary['sampling']['keyframe_note']}",
        "- Every pedestrian XML box, including outside, invalid, interpolated, sampled, and omitted boxes, "
        "is represented in `annotation_metadata.jsonl.gz` with an inclusion/exclusion status.",
        "- Samples use the existing recording-set assignments. Sets are never divided across splits.",
        "",
        "| Split | Sets | Images | Labels |",
        "|---|---:|---:|---:|",
    ]
    for split, values in summary["split_distribution"].items():
        lines.append(
            f"| {split} | {', '.join(values['sets']) or '(none)'} | "
            f"{values['images']} | {values['labels']} |"
        )
    lines.extend(
        [
            "",
            "## Annotation policy and limitations",
            "",
            "- YOLO labels have one class only: `pedestrian` (class ID 0).",
            "- `outside=1` and invalid/non-finite/non-positive/out-of-image boxes are not detector targets; "
            "they remain recorded in metadata and summary counts.",
            "- Occluded boxes are retained when their geometry is valid. Both the CVAT occlusion flag "
            "and PIE occlusion attribute are preserved.",
            "- `action` and `cross` remain metadata and are not mixed into detector labels.",
            "- CSV frame indices are a cross-check only. XML is the source of boxes; mismatches are listed "
            "without adding or correcting annotations.",
            "- PIE is not exhaustive: an image without a labeled pedestrian box is not created as a "
            "confirmed background example.",
            "- The sparse sample does not contain every CVAT keyframe by default. Keyframe flags and "
            "every annotation's sampling status are preserved so omitted frames are auditable; set "
            "`preserve_keyframes` to true to extract every valid keyframe frame as well.",
            "- The fixed validation/test sets contain few independent recording groups; report this "
            "limitation in later evaluation.",
            "",
            "## Files",
            "",
            "- `images/` and `labels/`: sampled frames and normalized YOLO labels, grouped by split/set/video.",
            "- `annotation_metadata.jsonl.gz`: source box and label metadata plus sampling/exclusion status.",
            "- `summary.json`: counts, sampling policy, split distribution, and per-video details.",
            "- `csv_crosscheck.json`: visible pedestrian XML/CSV frame-set comparison by video.",
            "- `split_manifest.json`: recording-set assignments and split rationale.",
            "- `visual_samples/` and `visual_sample_manifest.json`: rendered bounding-box examples and roles.",
            "- `data.yaml`: one-class YOLO dataset configuration.",
            "",
            "## Reproducing",
            "",
            "```powershell",
            "python -m smartped_ai.data.prepare_detection",
            "```",
            "",
        ]
    )
    (output_dir / "preparation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a sparse, set-split PIE pedestrian detection dataset in YOLO format."
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--frame-stride",
        type=int,
        help="Override the configured frame-index sampling stride.",
    )
    args = parser.parse_args(argv)
    try:
        result = run_preparation(args.config, args.frame_stride)
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"PIE detection preparation failed: {exc}", file=sys.stderr)
        return 2
    summary = result["summary"]
    print(
        f"Prepared {summary['images']} images and {summary['labels']} pedestrian labels "
        f"from {len(summary['videos'])} videos."
    )
    print(f"Output: {result['output_dir']}")
    print(f"Report: {result['output_dir'] / 'preparation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
