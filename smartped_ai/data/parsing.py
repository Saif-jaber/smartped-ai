from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from .models import SpatialSummary, ValidationIssue, increment_nested

_REQUIRED_LABEL_ATTRIBUTES = {
    "pedestrian": {"action", "cross"},
    "vehicle": {"type"},
}
_ALLOWED_LABEL_VALUES = {
    ("pedestrian", "action"): {"walking", "standing"},
    ("pedestrian", "cross"): {
        "not-crossing",
        "crossing",
        "crossing-irrelevant",
    },
}


@dataclass(frozen=True)
class AnnotatedFrames:
    declared_count: int
    frame_indices: list[int]


def _issue(
    issues: list[ValidationIssue],
    severity: str,
    code: str,
    message: str,
    path: Path,
    video_id: str | None = None,
    frame_index: int | None = None,
    track_id: str | None = None,
) -> None:
    issues.append(
        ValidationIssue(
            severity=severity,
            code=code,
            message=message,
            path=str(path),
            video_id=video_id,
            frame_index=frame_index,
            track_id=track_id,
        )
    )


def _json_line(output: IO[str], record: dict[str, Any]) -> None:
    output.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
    output.write("\n")


def _positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except ValueError:
        return None
    return result if result > 0 else None


def parse_annotated_frames(
    path: Path, set_id: str, issues: list[ValidationIssue]
) -> dict[str, AnnotatedFrames]:
    result: dict[str, AnnotatedFrames] = {}
    if not path.is_file():
        issues.append(
            ValidationIssue(
                "error", "missing_frame_csv", "Annotated-frame CSV is missing.", str(path)
            )
        )
        return result

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            for row_number, row in enumerate(csv.reader(source), start=1):
                if not row or all(not cell.strip() for cell in row):
                    continue
                video_id = Path(row[0].strip()).stem
                if not video_id or len(row) < 2:
                    _issue(
                        issues,
                        "error",
                        "malformed_frame_csv_row",
                        f"Row {row_number} must contain a video name and declared frame count.",
                        path,
                    )
                    continue
                try:
                    declared_count = int(row[1])
                except ValueError:
                    _issue(
                        issues,
                        "error",
                        "invalid_declared_frame_count",
                        f"Row {row_number} has a non-integer declared frame count.",
                        path,
                        video_id,
                    )
                    continue
                if declared_count < 0:
                    _issue(
                        issues,
                        "error",
                        "invalid_declared_frame_count",
                        f"Row {row_number} has a negative declared frame count.",
                        path,
                        video_id,
                    )

                if video_id in result:
                    _issue(
                        issues,
                        "error",
                        "duplicate_frame_csv_video",
                        f"Video {video_id} occurs more than once in the frame CSV.",
                        path,
                        video_id,
                    )
                    continue

                frames: list[int] = []
                seen: set[int] = set()
                for column, cell in enumerate(row[2:], start=3):
                    cell = cell.strip()
                    if not cell:
                        continue
                    try:
                        frame_index = int(cell)
                    except ValueError:
                        _issue(
                            issues,
                            "error",
                            "invalid_frame_csv_index",
                            f"Row {row_number}, column {column} is not an integer frame index.",
                            path,
                            video_id,
                        )
                        continue
                    if frame_index in seen:
                        _issue(
                            issues,
                            "error",
                            "duplicate_annotated_frame",
                            f"Frame {frame_index} is repeated in the frame CSV.",
                            path,
                            video_id,
                            frame_index,
                        )
                    else:
                        seen.add(frame_index)
                    frames.append(frame_index)

                if declared_count != len(frames):
                    _issue(
                        issues,
                        "error",
                        "frame_csv_count_mismatch",
                        f"Declared {declared_count} frame indices but found {len(frames)}.",
                        path,
                        video_id,
                    )
                result[video_id] = AnnotatedFrames(declared_count, frames)
    except (OSError, UnicodeError, csv.Error) as exc:
        _issue(issues, "error", "frame_csv_read_error", str(exc), path)

    for video_id in result:
        if not video_id:
            _issue(
                issues,
                "error",
                "empty_video_id_in_frame_csv",
                f"Frame CSV for {set_id} contains an empty video identifier.",
                path,
            )
    return result


def _parse_coordinate(
    value: str | None,
    name: str,
    issues: list[ValidationIssue],
    path: Path,
    video_id: str,
    frame_index: int | None,
    track_id: str | None,
) -> float | None:
    if value is None:
        _issue(
            issues,
            "error",
            "missing_bbox_coordinate",
            f"Bounding box is missing coordinate {name}.",
            path,
            video_id,
            frame_index,
            track_id,
        )
        return None
    try:
        result = float(value)
    except ValueError:
        result = math.nan
    if not math.isfinite(result):
        _issue(
            issues,
            "error",
            "invalid_bbox_coordinate",
            f"Bounding box coordinate {name} is not finite numeric data.",
            path,
            video_id,
            frame_index,
            track_id,
        )
        return None
    return result


def _attribute_values(box: ET.Element) -> dict[str, str]:
    return {
        attribute.get("name", ""): (attribute.text or "").strip()
        for attribute in box.findall("attribute")
        if attribute.get("name")
    }


def parse_spatial_xml(
    path: Path,
    set_id: str,
    video_id: str,
    output: IO[str],
    issues: list[ValidationIssue],
) -> SpatialSummary:
    visible_frames: set[int] = set()
    all_box_frames: set[int] = set()
    track_ids: dict[str, set[str]] = {}
    track_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    visible_box_counts: Counter[str] = Counter()
    attribute_counts: dict[str, dict[str, int]] = {}
    track_id_owners: dict[tuple[str, str], int] = {}
    task_size: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    current_label: str | None = None
    current_track_index = 0
    current_track_ids: set[str] = set()

    stack: list[ET.Element] = []
    try:
        for event, element in ET.iterparse(path, events=("start", "end")):
            if event == "start" and element.tag == "track":
                stack.append(element)
                current_label = element.get("label") or "unknown"
                current_track_index += 1
                current_track_ids = set()
                continue
            if event == "start":
                stack.append(element)
                continue

            if element.tag == "size" and task_size is None:
                if element.text:
                    try:
                        task_size = int(element.text.strip())
                    except ValueError:
                        task_size = None
            elif element.tag == "width" and image_width is None:
                image_width = _positive_int(element.text)
            elif element.tag == "height" and image_height is None:
                image_height = _positive_int(element.text)

            if element.tag == "box" and current_label is not None:
                label = current_label
                box_counts[label] += 1
                raw_frame = element.get("frame")
                try:
                    frame_index = int(raw_frame) if raw_frame is not None else None
                except ValueError:
                    frame_index = None
                if frame_index is None:
                    _issue(
                        issues,
                        "error",
                        "invalid_annotation_frame",
                        "Spatial annotation box has a missing or invalid frame index.",
                        path,
                        video_id,
                    )
                else:
                    all_box_frames.add(frame_index)

                values = _attribute_values(element)
                object_id = values.get("id") or None
                if object_id is None:
                    _issue(
                        issues,
                        "error",
                        "missing_object_id",
                        f"{label} annotation box has no ID attribute.",
                        path,
                        video_id,
                        frame_index,
                    )
                else:
                    track_ids.setdefault(label, set()).add(object_id)
                    current_track_ids.add(object_id)

                outside = element.get("outside", "0") == "1"
                if not outside:
                    visible_box_counts[label] += 1
                    if frame_index is not None:
                        visible_frames.add(frame_index)

                for attribute_name, attribute_value in values.items():
                    if attribute_name != "id":
                        increment_nested(attribute_counts, attribute_name, attribute_value)

                required_label_attributes = _REQUIRED_LABEL_ATTRIBUTES.get(label, set())
                for required_attribute in (
                    required_label_attributes if not outside else ()
                ):
                    if not values.get(required_attribute):
                        _issue(
                            issues,
                            "warning",
                            f"missing_{required_attribute}_label",
                            f"{label} annotation box has no {required_attribute} label.",
                            path,
                            video_id,
                            frame_index,
                            object_id,
                        )
                for (category, attribute_name), allowed in _ALLOWED_LABEL_VALUES.items():
                    if label == category and attribute_name in values and values[attribute_name] not in allowed:
                        _issue(
                            issues,
                            "warning",
                            f"unknown_{attribute_name}_label",
                            f"Unrecognized {attribute_name} label {values[attribute_name]!r}.",
                            path,
                            video_id,
                            frame_index,
                            object_id,
                        )

                coords = (
                    {
                        name: _parse_coordinate(
                            element.get(xml_name),
                            xml_name,
                            issues,
                            path,
                            video_id,
                            frame_index,
                            object_id,
                        )
                        for name, xml_name in (
                            ("x1", "xtl"),
                            ("y1", "ytl"),
                            ("x2", "xbr"),
                            ("y2", "ybr"),
                        )
                    }
                    if not outside
                    else {"x1": None, "y1": None, "x2": None, "y2": None}
                )
                if not outside and all(value is not None for value in coords.values()):
                    x1, y1, x2, y2 = (
                        coords["x1"],
                        coords["y1"],
                        coords["x2"],
                        coords["y2"],
                    )
                    if x1 >= x2 or y1 >= y2:
                        _issue(
                            issues,
                            "error",
                            "nonpositive_bbox",
                            "Bounding box must have positive width and height.",
                            path,
                            video_id,
                            frame_index,
                            object_id,
                        )
                    if (
                        image_width is not None
                        and image_height is not None
                        and (x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height)
                    ):
                        _issue(
                            issues,
                            "warning",
                            "bbox_outside_image",
                            f"Bounding box extends beyond {image_width}x{image_height} image bounds.",
                            path,
                            video_id,
                            frame_index,
                            object_id,
                        )

                _json_line(
                    output,
                    {
                        "set_id": set_id,
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "object_category": label,
                        "track_id": object_id,
                        "bbox_xyxy": coords,
                        "occluded": element.get("occluded"),
                        "outside": outside,
                        "keyframe": element.get("keyframe") == "1",
                        "attributes": values,
                    },
                )
                element.clear()
                stack.pop()
                continue

            if element.tag == "track" and current_label is not None:
                label = current_label
                track_counts[label] += 1
                if not current_track_ids:
                    _issue(
                        issues,
                        "error",
                        "missing_track_id",
                        f"{label} track contains no ID attribute.",
                        path,
                        video_id,
                    )
                elif len(current_track_ids) > 1:
                    _issue(
                        issues,
                        "error",
                        "inconsistent_track_id",
                        f"{label} track contains multiple IDs: {', '.join(sorted(current_track_ids))}.",
                        path,
                        video_id,
                    )
                for object_id in current_track_ids:
                    key = (label, object_id)
                    previous_track = track_id_owners.get(key)
                    if previous_track is not None:
                        _issue(
                            issues,
                            "error",
                            "duplicate_track_id",
                            f"ID {object_id} is used by multiple {label} tracks.",
                            path,
                            video_id,
                            track_id=object_id,
                        )
                    else:
                        track_id_owners[key] = current_track_index
                element.clear()
                stack.pop()
                current_label = None
                current_track_ids = set()
                continue

            stack.pop()
    except (ET.ParseError, OSError) as exc:
        _issue(issues, "error", "spatial_xml_read_error", str(exc), path, video_id)

    return SpatialSummary(
        task_size=task_size,
        image_width=image_width,
        image_height=image_height,
        visible_frames=visible_frames,
        all_box_frames=all_box_frames,
        track_ids=track_ids,
        track_counts=dict(track_counts),
        box_counts=dict(box_counts),
        visible_box_counts=dict(visible_box_counts),
        attribute_counts=attribute_counts,
    )


def _normalise_scalar(value: str) -> int | float | str:
    try:
        integer = int(value)
    except ValueError:
        try:
            number = float(value)
        except ValueError:
            return value
        return number if math.isfinite(number) else value
    return integer


def parse_pedestrian_attributes(
    path: Path,
    set_id: str,
    video_id: str,
    output: IO[str],
    issues: list[ValidationIssue],
) -> tuple[set[str], dict[str, int], list[float]]:
    ids: set[str] = set()
    crossing_counts: Counter[str] = Counter()
    intention_probabilities: list[float] = []
    seen: set[str] = set()
    stack: list[ET.Element] = []
    try:
        for event, element in ET.iterparse(path, events=("start", "end")):
            if event == "start":
                stack.append(element)
                continue
            if element.tag != "pedestrian":
                stack.pop()
                continue
            attributes = dict(element.attrib)
            track_id = attributes.get("id", "").strip()
            if not track_id:
                _issue(
                    issues,
                    "error",
                    "missing_attribute_track_id",
                    "Pedestrian attribute record has no ID.",
                    path,
                    video_id,
                )
            elif track_id in seen:
                _issue(
                    issues,
                    "error",
                    "duplicate_attribute_track_id",
                    f"Pedestrian attribute ID {track_id} occurs more than once.",
                    path,
                    video_id,
                    track_id=track_id,
                )
            else:
                ids.add(track_id)
                seen.add(track_id)

            crossing = attributes.get("crossing")
            if crossing is None:
                _issue(
                    issues,
                    "error",
                    "missing_crossing_outcome",
                    "Pedestrian attribute record has no crossing outcome.",
                    path,
                    video_id,
                    track_id=track_id or None,
                )
            else:
                crossing_counts[crossing] += 1
                if crossing not in {"-1", "0", "1"}:
                    _issue(
                        issues,
                        "warning",
                        "unknown_crossing_outcome",
                        f"Unrecognized crossing outcome {crossing!r}.",
                        path,
                        video_id,
                        track_id=track_id or None,
                    )

            raw_probability = attributes.get("intention_prob")
            probability: float | None = None
            if raw_probability is None:
                _issue(
                    issues,
                    "error",
                    "missing_intention_probability",
                    "Pedestrian attribute record has no intention_prob.",
                    path,
                    video_id,
                    track_id=track_id or None,
                )
            else:
                try:
                    probability = float(raw_probability)
                except ValueError:
                    probability = math.nan
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    _issue(
                        issues,
                        "error",
                        "invalid_intention_probability",
                        f"intention_prob must be finite and in [0, 1], got {raw_probability!r}.",
                        path,
                        video_id,
                        track_id=track_id or None,
                    )
                else:
                    intention_probabilities.append(probability)

            _json_line(
                output,
                {
                    "set_id": set_id,
                    "video_id": video_id,
                    "track_id": track_id or None,
                    "intention_probability": probability,
                    "crossing_outcome": _normalise_scalar(crossing)
                    if crossing is not None
                    else None,
                    "attributes": {
                        key: _normalise_scalar(value) for key, value in attributes.items()
                    },
                },
            )
            element.clear()
            stack.pop()
    except (ET.ParseError, OSError) as exc:
        _issue(issues, "error", "attribute_xml_read_error", str(exc), path, video_id)
    return ids, dict(crossing_counts), intention_probabilities


def parse_ego_telemetry(
    path: Path,
    set_id: str,
    video_id: str,
    output: IO[str],
    issues: list[ValidationIssue],
) -> tuple[set[int], dict[str, int]]:
    frame_ids: set[int] = set()
    duplicate_count = 0
    invalid_count = 0
    stack: list[ET.Element] = []
    try:
        for event, element in ET.iterparse(path, events=("start", "end")):
            if event == "start":
                stack.append(element)
                continue
            if element.tag != "frame":
                stack.pop()
                continue
            raw_id = element.get("id")
            try:
                frame_id = int(raw_id) if raw_id is not None else None
            except ValueError:
                frame_id = None
            if frame_id is None:
                invalid_count += 1
                _issue(
                    issues,
                    "error",
                    "invalid_ego_frame_id",
                    "Ego telemetry frame has a missing or invalid id.",
                    path,
                    video_id,
                )
            elif frame_id in frame_ids:
                duplicate_count += 1
                _issue(
                    issues,
                    "error",
                    "duplicate_ego_frame_id",
                    f"Ego telemetry frame ID {frame_id} occurs more than once.",
                    path,
                    video_id,
                    frame_id,
                )
            else:
                frame_ids.add(frame_id)

            fields = {
                key: _normalise_scalar(value)
                for key, value in element.attrib.items()
                if key != "id"
            }
            _json_line(
                output,
                {
                    "set_id": set_id,
                    "video_id": video_id,
                    "frame_index": frame_id,
                    "fields": fields,
                },
            )
            element.clear()
            stack.pop()
    except (ET.ParseError, OSError) as exc:
        _issue(issues, "error", "ego_xml_read_error", str(exc), path, video_id)
    return frame_ids, {
        "records": len(frame_ids) + duplicate_count + invalid_count,
        "unique_frames": len(frame_ids),
        "duplicate_ids": duplicate_count,
        "invalid_ids": invalid_count,
    }


def compare_frame_sets(
    xml_frames: set[int],
    csv_frames: set[int],
    path: Path,
    video_id: str,
    issues: list[ValidationIssue],
) -> dict[str, int]:
    xml_only = xml_frames - csv_frames
    csv_only = csv_frames - xml_frames
    if xml_only or csv_only:
        xml_only_sample = sorted(xml_only)[:20]
        csv_only_sample = sorted(csv_only)[:20]
        _issue(
            issues,
            "error",
            "annotation_csv_frame_mismatch",
            f"XML/CSV annotated frame sets differ: {len(xml_only)} XML-only and "
            f"{len(csv_only)} CSV-only indices. XML-only sample: {xml_only_sample}; "
            f"CSV-only sample: {csv_only_sample}.",
            path,
            video_id,
        )
    return {
        "xml_only": len(xml_only),
        "csv_only": len(csv_only),
    }


def frame_set_to_ranges(frames: list[int]) -> list[list[int]]:
    """Return compact inclusive ranges for sorted unique frame indices."""
    ordered = sorted(set(frames))
    if not ordered:
        return []
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        ranges.append([start, previous])
        start = previous = frame
    ranges.append([start, previous])
    return ranges

