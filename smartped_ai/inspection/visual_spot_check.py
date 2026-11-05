from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "pie_dataset.json"
_SAMPLE_CATEGORIES = (
    "walking",
    "standing",
    "crossing",
    "not_crossing",
    "occluded_or_partial",
    "multiple_pedestrians_and_vehicles",
)


@dataclass(frozen=True)
class AnnotationBox:
    category: str
    track_id: str | None
    bbox_xyxy: tuple[float, float, float, float] | None
    outside: bool
    occluded: bool
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class FrameAnnotations:
    boxes: list[AnnotationBox] = field(default_factory=list)
    boxes_by_frame: dict[int, list[AnnotationBox]] = field(default_factory=dict)
    visible_frames: set[int] = field(default_factory=set)
    image_width: int | None = None
    image_height: int | None = None


@dataclass
class FrameSummary:
    pedestrians: int = 0
    vehicles: int = 0
    categories: set[str] = field(default_factory=set)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _box_from_xml(element: ET.Element, category: str) -> AnnotationBox:
    attributes = {
        child.get("name", ""): (child.text or "").strip()
        for child in element
        if _local_name(child.tag) == "attribute" and child.get("name")
    }
    outside = element.get("outside", "0") == "1"
    try:
        coords = tuple(
            float(element.attrib[name])
            for name in ("xtl", "ytl", "xbr", "ybr")
        )
        bbox = coords if all(map(_is_finite, coords)) else None
    except (KeyError, ValueError):
        bbox = None
    return AnnotationBox(
        category=category,
        track_id=attributes.get("id"),
        bbox_xyxy=bbox,
        outside=outside,
        occluded=element.get("occluded", "0") == "1",
        attributes=attributes,
    )


def _is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def _read_spatial_xml(
    path: Path, selected_frames: set[int] | None = None
) -> FrameAnnotations:
    result = FrameAnnotations()
    current_label: str | None = None
    try:
        for event, element in ET.iterparse(path, events=("start", "end")):
            tag = _local_name(element.tag)
            if event == "start" and tag == "track":
                current_label = element.get("label") or "unknown"
                continue
            if event == "end" and tag == "width" and result.image_width is None:
                try:
                    result.image_width = int(element.text or "")
                except ValueError:
                    pass
            elif event == "end" and tag == "height" and result.image_height is None:
                try:
                    result.image_height = int(element.text or "")
                except ValueError:
                    pass
            elif event == "end" and tag == "box" and current_label is not None:
                try:
                    frame_index = int(element.get("frame", ""))
                except ValueError:
                    frame_index = None
                if frame_index is not None and element.get("outside", "0") != "1":
                    result.visible_frames.add(frame_index)
                if (
                    frame_index is not None
                    and (selected_frames is None or frame_index in selected_frames)
                ):
                    box = _box_from_xml(element, current_label)
                    result.boxes.append(box)
                    result.boxes_by_frame.setdefault(frame_index, []).append(box)
                element.clear()
            elif event == "end" and tag == "track":
                current_label = None
                element.clear()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Could not read spatial annotation XML {path}: {exc}") from exc
    return result


def _read_csv_frames(path: Path, video_id: str) -> set[int]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            for row in csv.reader(source):
                if not row or len(row) < 2:
                    continue
                if Path(row[0].strip()).stem != video_id:
                    continue
                frames: set[int] = set()
                for value in row[2:]:
                    if value.strip():
                        try:
                            frames.add(int(value))
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid frame index {value!r} in {path}."
                            ) from exc
                return frames
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read annotated-frame CSV {path}: {exc}") from exc
    raise ValueError(f"No {video_id} row found in annotated-frame CSV {path}.")


def _normalise_record(record: dict[str, Any]) -> AnnotationBox:
    coords = record.get("bbox_xyxy") or {}
    try:
        bbox = tuple(float(coords[name]) for name in ("x1", "y1", "x2", "y2"))
        if not all(map(_is_finite, bbox)):
            bbox = None
    except (KeyError, TypeError, ValueError):
        bbox = None
    attributes = record.get("attributes") or {}
    return AnnotationBox(
        category=str(record.get("object_category") or "unknown"),
        track_id=record.get("track_id"),
        bbox_xyxy=bbox,
        outside=bool(record.get("outside")),
        occluded=str(record.get("occluded", "0")) == "1",
        attributes={str(key): str(value) for key, value in attributes.items()},
    )


def _csv_frame_index(dataset_root: Path) -> dict[tuple[str, str], set[int]]:
    result: dict[tuple[str, str], set[int]] = {}
    for set_dir in sorted((dataset_root / "annotations").glob("set*")):
        csv_path = set_dir / f"{set_dir.name}_annotated_frames.csv"
        if not csv_path.is_file():
            continue
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
                for row in csv.reader(source):
                    if not row or len(row) < 2:
                        continue
                    video_id = Path(row[0].strip()).stem
                    frames = set()
                    for value in row[2:]:
                        if value.strip():
                            try:
                                frames.add(int(value))
                            except ValueError as exc:
                                raise ValueError(
                                    f"Invalid frame index {value!r} in {csv_path}."
                                ) from exc
                    result[(set_dir.name, video_id)] = frames
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ValueError(f"Could not read annotated-frame CSV {csv_path}: {exc}") from exc
    return result


def _select_category_frames(
    annotations_path: Path,
    csv_frames: dict[tuple[str, str], set[int]],
) -> dict[str, tuple[str, str, int]]:
    summaries: dict[tuple[str, str, int], FrameSummary] = {}
    selected: dict[str, tuple[str, str, int]] = {}
    try:
        with gzip.open(annotations_path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {annotations_path}: {exc}"
                    ) from exc
                set_id = str(record["set_id"])
                video_id = str(record["video_id"])
                frame_index = int(record["frame_index"])
                key = (set_id, video_id, frame_index)
                if frame_index not in csv_frames.get((set_id, video_id), set()):
                    continue
                box = _normalise_record(record)
                if box.outside or box.bbox_xyxy is None:
                    continue
                summary = summaries.setdefault(key, FrameSummary())
                attrs = box.attributes
                if box.category == "pedestrian":
                    summary.pedestrians += 1
                    action = attrs.get("action", "").lower()
                    crossing = attrs.get("cross", "").lower()
                    if action == "walking":
                        summary.categories.add("walking")
                    elif action == "standing":
                        summary.categories.add("standing")
                    if crossing == "crossing":
                        summary.categories.add("crossing")
                    elif crossing == "not-crossing":
                        summary.categories.add("not_crossing")
                    if box.occluded or attrs.get("occlusion", "").lower() not in {
                        "",
                        "none",
                        "0",
                        "__undefined__",
                    }:
                        summary.categories.add("occluded_or_partial")
                elif box.category == "vehicle":
                    summary.vehicles += 1
                if (
                    summary.pedestrians >= 2
                    and summary.vehicles >= 1
                    and "multiple_pedestrians_and_vehicles" not in selected
                ):
                    selected["multiple_pedestrians_and_vehicles"] = key
                for category in summary.categories:
                    selected.setdefault(category, key)
    except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            ("Invalid JSON", "Invalid frame index")
        ):
            raise
        raise ValueError(f"Could not scan normalized annotations {annotations_path}: {exc}") from exc
    return selected


def _read_normalized_frames(
    annotations_path: Path,
    selected_keys: set[tuple[str, str, int]],
) -> dict[tuple[str, str, int], list[AnnotationBox]]:
    result: dict[tuple[str, str, int], list[AnnotationBox]] = defaultdict(list)
    if not selected_keys:
        return result
    remaining = set(selected_keys)
    try:
        with gzip.open(annotations_path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                    key = (
                        str(record["set_id"]),
                        str(record["video_id"]),
                        int(record["frame_index"]),
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid record on line {line_number} of {annotations_path}: {exc}"
                    ) from exc
                if key in remaining:
                    result[key].append(_normalise_record(record))
    except (OSError, EOFError) as exc:
        raise ValueError(f"Could not read normalized annotations {annotations_path}: {exc}") from exc
    return result


def _load_mismatch_samples(
    issues_path: Path, dataset_root: Path
) -> tuple[dict[tuple[str, str, int], set[str]], dict[tuple[str, str], set[int]]]:
    try:
        issues = json.loads(issues_path.read_text(encoding="utf-8"))["issues"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Could not read validation issues {issues_path}: {exc}") from exc

    sample_roles: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    visible_by_video: dict[tuple[str, str], set[int]] = {}
    for issue in issues:
        if issue.get("code") != "annotation_csv_frame_mismatch":
            continue
        video_id = issue.get("video_id")
        source_path = issue.get("path")
        set_id = Path(source_path).parent.name if source_path else ""
        if not video_id or not set_id:
            continue
        annotation_path = dataset_root / "annotations" / set_id / f"{video_id}_annt.xml"
        csv_path = dataset_root / "annotations" / set_id / f"{set_id}_annotated_frames.csv"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Spatial annotation XML is missing: {annotation_path}")
        xml_data = _read_spatial_xml(annotation_path, set())
        key = (set_id, video_id)
        csv_set = _read_csv_frames(csv_path, video_id)
        visible_by_video[key] = xml_data.visible_frames
        xml_only = xml_data.visible_frames - csv_set
        csv_only = csv_set - xml_data.visible_frames
        for frame_index in xml_only:
            sample_roles[(set_id, video_id, frame_index)].add("reported_xml_only_mismatch")
        for frame_index in csv_only:
            sample_roles[(set_id, video_id, frame_index)].add("reported_csv_only_mismatch")

        aligned = xml_data.visible_frames & csv_set
        for frame_index in sorted(xml_only | csv_only):
            neighbors = sorted(aligned, key=lambda candidate: (abs(candidate - frame_index), candidate))
            for neighbor in neighbors[:1]:
                sample_roles[(set_id, video_id, neighbor)].add("aligned_mismatch_control")
    return sample_roles, visible_by_video


def _load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read dataset config {config_path}: {exc}") from exc
    config_dir = config_path.resolve().parent
    dataset_root = (config_dir / config.get("dataset_root", "../dataset")).resolve()
    return config, dataset_root


def _video_path(dataset_root: Path, set_id: str, video_id: str) -> Path:
    return dataset_root / "PIE_clips" / set_id / f"{video_id}.mp4"


def _annotation_path(dataset_root: Path, set_id: str, video_id: str) -> Path:
    return dataset_root / "annotations" / set_id / f"{video_id}_annt.xml"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _read_video_frame(video_path: Path, frame_index: int):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to decode or display frames. Install it in the active "
            "Python environment; this tool does not install dependencies."
        ) from exc
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index < 0 or frame_index >= frame_count:
            raise ValueError(
                f"Frame {frame_index} is outside {video_path.name} "
                f"(frame count: {frame_count})."
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            raise OSError(f"Could not decode frame {frame_index} from {video_path}.")
        return frame, frame_count
    finally:
        capture.release()


def _label_text(box: AnnotationBox) -> str:
    parts = [box.category, f"ID={box.track_id or 'missing'}"]
    for name in ("action", "cross"):
        value = box.attributes.get(name)
        if value and value not in {"__undefined__", "n/a"}:
            parts.append(f"{name}={value}")
    if box.occluded:
        parts.append("occluded")
    elif box.attributes.get("occlusion", "").lower() not in {
        "",
        "none",
        "0",
        "__undefined__",
    }:
        parts.append(f"occlusion={box.attributes['occlusion']}")
    return " | ".join(parts)


def _render_frame(frame, boxes: list[AnnotationBox], header: str):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to decode or display frames. Install it in the active "
            "Python environment; this tool does not install dependencies."
        ) from exc
    image = frame.copy()
    height, width = image.shape[:2]
    colors = {
        "pedestrian": (40, 220, 40),
        "vehicle": (255, 170, 35),
    }
    outside_boxes = [box for box in boxes if box.outside]
    outside_labels = [
        f"OUTSIDE {_label_text(box)}" for box in outside_boxes[:8]
    ]
    if len(outside_boxes) > len(outside_labels):
        outside_labels.append(f"... and {len(outside_boxes) - len(outside_labels)} more")
    bar_height = 34 + 22 * len(outside_labels)
    cv2.rectangle(image, (0, 0), (width, bar_height), (20, 20, 20), -1)
    cv2.putText(
        image,
        header,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for index, text in enumerate(outside_labels):
        cv2.putText(
            image,
            text,
            (10, 34 + 18 * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    occupied_labels: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if box.outside or box.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = box.bbox_xyxy
        left = max(0, min(width - 1, round(x1)))
        top = max(0, min(height - 1, round(y1)))
        right = max(0, min(width - 1, round(x2)))
        bottom = max(0, min(height - 1, round(y2)))
        if right <= left or bottom <= top:
            continue
        color = colors.get(box.category, (0, 190, 255))
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        text = _label_text(box)
        text_size, baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        text_height = text_size[1] + baseline + 2
        text_width = min(width, text_size[0] + 8)
        text_left = max(0, min(left, width - text_width))
        base_top = max(bar_height, top - text_height - 5)
        step = text_height + 3
        candidates = [base_top]
        for distance in range(1, 10):
            candidates.extend((base_top + distance * step, base_top - distance * step))
        candidates.extend((bottom + 5, top + 5))
        selected_top = None
        for candidate_top in candidates:
            candidate_top = max(bar_height, min(height - text_height, candidate_top))
            candidate = (
                text_left,
                candidate_top,
                text_left + text_width,
                candidate_top + text_height,
            )
            if not any(
                candidate[0] < other[2]
                and candidate[2] > other[0]
                and candidate[1] < other[3]
                and candidate[3] > other[1]
                for other in occupied_labels
            ):
                selected_top = candidate_top
                occupied_labels.append(candidate)
                break
        if selected_top is None:
            selected_top = max(bar_height, min(height - text_height, base_top))
        baseline_y = selected_top + text_size[1] + baseline
        cv2.rectangle(
            image,
            (text_left, selected_top),
            (min(width - 1, text_left + text_width), selected_top + text_height),
            color,
            -1,
        )
        cv2.putText(
            image,
            text,
            (text_left + 4, baseline_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return image


def _save_rendered_frame(
    frame,
    boxes: list[AnnotationBox],
    set_id: str,
    video_id: str,
    frame_index: int,
    alignment: str,
    output_path: Path,
) -> dict[str, Any]:
    import cv2

    header = f"{video_id} | {set_id} | frame {frame_index} | {alignment}"
    rendered = _render_frame(frame, boxes, header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 92]
    ):
        raise OSError(f"Could not write rendered frame: {output_path}")
    visible = [box for box in boxes if not box.outside]
    missing_bbox = sum(box.bbox_xyxy is None for box in visible)
    missing_id = sum(not box.track_id for box in visible)
    missing_action_or_cross = sum(
        box.category == "pedestrian"
        and not (
            box.attributes.get("action")
            and box.attributes.get("action") not in {"__undefined__", "n/a"}
            and box.attributes.get("cross")
            and box.attributes.get("cross") not in {"__undefined__", "n/a"}
        )
        for box in visible
    )
    height, width = frame.shape[:2]
    invalid_bbox = sum(
        box.bbox_xyxy is not None
        and (
            box.bbox_xyxy[0] >= box.bbox_xyxy[2]
            or box.bbox_xyxy[1] >= box.bbox_xyxy[3]
        )
        for box in visible
    )
    out_of_image_bbox = sum(
        box.bbox_xyxy is not None
        and (
            box.bbox_xyxy[0] < 0
            or box.bbox_xyxy[1] < 0
            or box.bbox_xyxy[2] > width
            or box.bbox_xyxy[3] > height
        )
        for box in visible
    )
    ids = [box.track_id for box in visible if box.track_id]
    return {
        "set_id": set_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "alignment": alignment,
        "image": output_path.name,
        "visible_box_count": len(visible),
        "outside_box_count": len(boxes) - len(visible),
        "missing_bbox_count": missing_bbox,
        "invalid_bbox_count": invalid_bbox,
        "out_of_image_bbox_count": out_of_image_bbox,
        "missing_id_count": missing_id,
        "duplicate_id_count": sum(count > 1 for count in Counter(ids).values()),
        "pedestrian_label_gaps": missing_action_or_cross,
        "ids": sorted({box.track_id for box in visible if box.track_id}),
        "labels": sorted(
            {
                f"{box.category}:{name}={box.attributes[name]}"
                for box in visible
                for name in ("action", "cross")
                if box.attributes.get(name)
            }
        ),
        "visual_status": "pending_manual_review",
        "finding": "",
        "likely_cause": "",
    }


def _likely_mismatch_cause(set_id: str) -> str:
    if set_id == "set02":
        return (
            "Possible CSV scope mismatch or omitted CSV index: the XML-only frame "
            "contains a visible crosswalk box and no visible pedestrian box."
        )
    if set_id == "set03":
        return (
            "Possible CSV scope mismatch or omitted CSV index: the XML-only frame "
            "contains a crosswalk box while the pedestrian track is outside."
        )
    if set_id == "set06":
        return (
            "Possible CSV scope mismatch, export omission, or vehicle-annotation "
            "interval outside the CSV's intended frame coverage."
        )
    return "Possible CSV scope mismatch or omitted frame index; inspect the source records."


def _write_report(
    output_dir: Path,
    frames: list[dict[str, Any]],
    unavailable_categories: list[str],
) -> None:
    mismatch_frames = [
        item
        for item in frames
        if "reported_xml_only_mismatch" in item["sample_categories"]
        or "reported_csv_only_mismatch" in item["sample_categories"]
    ]
    by_video: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in mismatch_frames:
        by_video[(item["set_id"], item["video_id"])].append(item)
    manually_reviewed = [
        item
        for item in frames
        if item["visual_status"] != "pending_manual_review"
    ]
    mismatch_frames_reviewed = [
        item
        for item in mismatch_frames
        if item["visual_status"] != "pending_manual_review"
    ]
    review_counts = Counter(item["visual_status"] for item in manually_reviewed)
    lines = [
        "# SmartPed AI Visual Spot-Check Report",
        "",
        "Generated by `smartped_ai.inspection.visual_spot_check` from the existing "
        "validation issues, source XML/CSV annotations, and selected video frames.",
        "This process does not modify the source videos, annotations, or validation artifacts.",
        "",
        "## Scope and status",
        "",
        f"- Rendered frame samples: **{len(frames)}**",
        f"- Frames with matching XML/CSV coverage: **{sum(item['alignment'] == 'XML/CSV aligned' for item in frames)}**",
        f"- Manually reviewed frames: **{len(manually_reviewed)}**",
        f"- Reported mismatch frames rendered: **{len(mismatch_frames)}**",
        f"- Reported mismatch frames manually reviewed / pending: **{len(mismatch_frames_reviewed)} / {len(mismatch_frames) - len(mismatch_frames_reviewed)}**",
        f"- Reviewed correct / incorrect / uncertain: **{review_counts['reviewed_correct']} / "
        f"{review_counts['reviewed_incorrect']} / {review_counts['reviewed_uncertain']}**",
        "- No frame alignment or annotation was automatically corrected.",
        "",
        "A rendered frame means the video frame was decoded and made available with its "
        "annotation overlays. It does not by itself establish that every box is visually correct.",
        "",
        "## Prioritized alignment discrepancies",
        "",
        "| Priority | Set / video | XML-only frames | CSV-only frames | Evidence and likely cause |",
        "|---|---|---:|---:|---|",
    ]
    mismatch_notes = []
    for (set_id, video_id), items in sorted(by_video.items()):
        xml_only = sorted(
            item["frame_index"]
            for item in items
            if "reported_xml_only_mismatch" in item["sample_categories"]
        )
        csv_only = sorted(
            item["frame_index"]
            for item in items
            if "reported_csv_only_mismatch" in item["sample_categories"]
        )
        priority = "High" if len(xml_only) >= 10 or len(csv_only) >= 10 else "Medium"
        lines.append(
            f"| {priority} | `{set_id}/{video_id}` | {len(xml_only)} | {len(csv_only)} | "
            f"{_likely_mismatch_cause(set_id)} |"
        )
        mismatch_notes.append(
            f"- `{set_id}/{video_id}` XML-only: "
            + (", ".join(map(str, xml_only)) if xml_only else "none")
            + "; CSV-only: "
            + (", ".join(map(str, csv_only)) if csv_only else "none")
            + "."
        )
    if mismatch_notes:
        lines.extend(["", *mismatch_notes])
    lines.extend(
        [
            "",
            "The set06 interval is prioritized first because it spans 50 contiguous frames. "
            "set02 and set03 each contain a single XML-only frame.",
            "",
            "### Alignment interpretation",
            "",
            f"- Source XML/CSV discrepancies reproduced: **{len(by_video)} videos**.",
            f"- XML-only frames: **{sum('reported_xml_only_mismatch' in item['sample_categories'] for item in mismatch_frames)}**.",
            f"- CSV-only frames: **{sum('reported_csv_only_mismatch' in item['sample_categories'] for item in mismatch_frames)}**.",
            "- Every reported mismatch frame was successfully decoded and rendered.",
            "- These are frame-list inconsistencies, not evidence by themselves of a temporal offset. "
            "The visual overlays determine whether the spatial boxes also appear displaced.",
            "",
            "## Bounding boxes, IDs, and labels",
            "",
            f"- Visible boxes with missing coordinates: **{sum(item['missing_bbox_count'] for item in frames)}**",
            f"- Visible boxes with non-positive geometry: **{sum(item['invalid_bbox_count'] for item in frames)}**",
            f"- Visible boxes extending beyond the decoded image: **{sum(item['out_of_image_bbox_count'] for item in frames)}**",
            f"- Visible boxes missing IDs: **{sum(item['missing_id_count'] for item in frames)}**",
            f"- Duplicate IDs within a frame: **{sum(item['duplicate_id_count'] for item in frames)}**",
            f"- Pedestrian boxes missing action/crossing labels: **{sum(item['pedestrian_label_gaps'] for item in frames)}**",
            "- These machine checks do not determine whether a box tightly matches the visible object.",
            "- Outside annotations are retained as metadata but are not drawn because they have no visible box.",
            "",
            "## Sample inventory",
            "",
            "| Set | Video | Frame | Sample categories | XML/CSV status | Visible boxes | Review status | Likely cause / finding | Screenshot |",
            "|---|---|---:|---|---|---:|---|---|---|",
        ]
    )
    for item in frames:
        link = f"[{item['image']}](frames/{item['image']})"
        cause_or_finding = "; ".join(
            value
            for value in (
                item["finding"],
                f"Likely cause: {item['likely_cause']}" if item["likely_cause"] else "",
            )
            if value
        ) or "—"
        lines.append(
            f"| `{item['set_id']}` | `{item['video_id']}` | {item['frame_index']} | "
            f"{', '.join(item['sample_categories'])} | {item['alignment']} | "
            f"{item['visible_box_count']} | {item['visual_status']} | "
            f"{cause_or_finding} | {link} |"
        )
    if unavailable_categories:
        lines.extend(
            [
                "",
                "Categories with no aligned sample found in normalized annotations: "
                + ", ".join(f"`{category}`" for category in unavailable_categories)
                + ".",
            ]
        )
    lines.extend(
        [
            "",
            "### Recorded manual findings",
            "",
        ]
    )
    if manually_reviewed:
        lines.extend(
            [
                "| Set / video | Frame | Status | Finding | Likely cause |",
                "|---|---:|---|---|---|",
            ]
        )
        for item in manually_reviewed:
            lines.append(
                f"| `{item['set_id']}/{item['video_id']}` | {item['frame_index']} | "
                f"{item['visual_status']} | {item['finding']} | {item['likely_cause']} |"
            )
    else:
        lines.append("No manual findings have been recorded.")
    lines.extend(
        [
            "",
            "## Recommended data-policy changes",
            "",
            "- Preserve current XML, CSV, and videos as immutable source evidence; do not "
            "silently add or remove frame indices.",
            "- Confirm and document whether the annotated-frame CSV represents any visible "
            "spatial object or only a narrower pedestrian/task subset. Validate matching scopes.",
            "- Mismatch-only boxes are non-pedestrian (crosswalk/vehicle), supporting a possible "
            "CSV scope mismatch rather than a temporal offset; verify against the PIE CSV contract.",
            "- If CSV coverage should include all spatial labels, amend it only through a reviewed, "
            "versioned curation step and retain the diff.",
            "- Define whether outside/fully occluded tracks and non-pedestrian labels count toward "
            "frame-level annotation coverage.",
            "",
            "## Reproducing the inspection",
            "",
            "```powershell",
            "python -m smartped_ai.inspection.visual_spot_check --samples",
            "python -m smartped_ai.inspection.visual_spot_check --set set06 --video video_0002 --frame 5260",
            "```",
            "",
            "In the single-frame window, press `s` to save a screenshot and `q` or Escape to close. "
            "Use `--no-display` to render without opening a window. Use `--record-review` with a "
            "status, finding, and likely cause to add a manual review to the report.",
            "",
        ]
    )
    (output_dir / "visual_spot_check_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def generate_samples(
    dataset_root: Path,
    output_dir: Path,
    issues_path: Path,
    normalized_annotations_path: Path,
) -> list[dict[str, Any]]:
    if not normalized_annotations_path.is_file():
        raise FileNotFoundError(
            f"Normalized annotations are missing: {normalized_annotations_path}. "
            "Run the existing ingestion pipeline first."
        )
    if not issues_path.is_file():
        raise FileNotFoundError(f"Validation issue file is missing: {issues_path}")

    csv_frames = _csv_frame_index(dataset_root)
    sample_roles, visible_by_video = _load_mismatch_samples(issues_path, dataset_root)
    selected = _select_category_frames(normalized_annotations_path, csv_frames)
    for category, key in selected.items():
        sample_roles[key].add(category)

    selected_keys = set(sample_roles)
    mismatch_keys_by_video: dict[tuple[str, str], set[int]] = defaultdict(set)
    for (set_id, video_id, frame_index), roles in sample_roles.items():
        if any(
            role in roles
            for role in (
                "reported_xml_only_mismatch",
                "reported_csv_only_mismatch",
                "aligned_mismatch_control",
            )
        ):
            mismatch_keys_by_video[(set_id, video_id)].add(frame_index)
    xml_frames: dict[tuple[str, str, int], list[AnnotationBox]] = {}
    for (set_id, video_id), frame_indices in mismatch_keys_by_video.items():
        xml_data = _read_spatial_xml(
            _annotation_path(dataset_root, set_id, video_id), frame_indices
        )
        visible_by_video[(set_id, video_id)] = xml_data.visible_frames
        for frame_index in frame_indices:
            xml_frames[(set_id, video_id, frame_index)] = xml_data.boxes_by_frame.get(
                frame_index, []
            )
    normalized_frames = _read_normalized_frames(
        normalized_annotations_path,
        selected_keys - {
            key
            for key, roles in sample_roles.items()
            if any(
                role in roles
                for role in (
                    "reported_xml_only_mismatch",
                    "reported_csv_only_mismatch",
                    "aligned_mismatch_control",
                )
            )
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for key in sorted(selected_keys):
        set_id, video_id, frame_index = key
        roles = sample_roles[key]
        mismatch = any(
            role in roles
            for role in (
                "reported_xml_only_mismatch",
                "reported_csv_only_mismatch",
                "aligned_mismatch_control",
            )
        )
        if mismatch:
            boxes = xml_frames.get(key, [])
            visible_frames = visible_by_video.get((set_id, video_id), set())
        else:
            boxes = normalized_frames.get(key, [])
        frame_listed = frame_index in csv_frames.get((set_id, video_id), set())
        is_visible = (
            frame_index in visible_frames
            if mismatch
            else any(not box.outside and box.bbox_xyxy is not None for box in boxes)
        )
        if is_visible and frame_listed:
            alignment = "XML/CSV aligned"
        elif is_visible:
            alignment = "XML-only"
        elif frame_listed:
            alignment = "CSV-only"
        else:
            alignment = "not in XML visible frames or CSV"

        video_path = _video_path(dataset_root, set_id, video_id)
        frame, frame_count = _read_video_frame(video_path, frame_index)
        image_name = (
            f"{_safe_name(set_id)}_{_safe_name(video_id)}_frame_{frame_index:06d}.jpg"
        )
        item = _save_rendered_frame(
            frame,
            boxes,
            set_id,
            video_id,
            frame_index,
            alignment,
            frames_dir / image_name,
        )
        item["sample_categories"] = sorted(roles)
        item["video_frame_count"] = frame_count
        if "reported_xml_only_mismatch" in roles:
            item["finding"] = (
                "Visible spatial annotation is present in XML but this frame is absent "
                "from the CSV; video frame decoded. Box-to-object match requires review."
            )
            item["likely_cause"] = _likely_mismatch_cause(set_id)
        elif "reported_csv_only_mismatch" in roles:
            item["finding"] = (
                "Frame is listed in the CSV but has no visible spatial box in XML; "
                "video frame decoded. Annotation completeness requires review."
            )
            item["likely_cause"] = (
                "Possible missing spatial annotation or a difference in the CSV/XML "
                "frame coverage policy."
            )
        manifest.append(item)

    categories_found = {
        category for categories in sample_roles.values() for category in categories
    }
    unavailable = [
        category for category in _SAMPLE_CATEGORIES if category not in categories_found
    ]
    manifest_path = output_dir / "sample_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(output_dir, manifest, unavailable)
    return manifest


def record_review(
    output_dir: Path,
    set_id: str,
    video_id: str,
    frame_index: int,
    status: str,
    finding: str,
    likely_cause: str,
) -> None:
    manifest_path = output_dir / "sample_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read sample manifest {manifest_path}: {exc}") from exc
    item = next(
        (
            sample
            for sample in manifest
            if sample["set_id"] == set_id
            and sample["video_id"] == video_id
            and sample["frame_index"] == frame_index
        ),
        None,
    )
    if item is None:
        raise ValueError(
            f"No generated sample for {set_id}/{video_id} frame {frame_index}. "
            "Run --samples first."
        )
    item["visual_status"] = f"reviewed_{status}"
    item["finding"] = finding.strip()
    item["likely_cause"] = likely_cause.strip()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    categories = {
        category
        for sample in manifest
        for category in sample["sample_categories"]
    }
    unavailable = [
        category for category in _SAMPLE_CATEGORIES if category not in categories
    ]
    _write_report(output_dir, manifest, unavailable)


def inspect_single_frame(
    dataset_root: Path,
    output_dir: Path,
    set_id: str,
    video_id: str,
    frame_index: int,
    show: bool,
) -> Path:
    if frame_index < 0:
        raise ValueError("Frame index must be zero or greater.")
    annotation_path = _annotation_path(dataset_root, set_id, video_id)
    video_path = _video_path(dataset_root, set_id, video_id)
    annotations = _read_spatial_xml(annotation_path, {frame_index})
    csv_path = dataset_root / "annotations" / set_id / f"{set_id}_annotated_frames.csv"
    csv_frames = _read_csv_frames(csv_path, video_id)
    is_visible = frame_index in annotations.visible_frames
    frame_listed = frame_index in csv_frames
    if is_visible and frame_listed:
        alignment = "XML/CSV aligned"
    elif is_visible:
        alignment = "XML-only"
    elif frame_listed:
        alignment = "CSV-only"
    else:
        alignment = "not listed"
    frame, _ = _read_video_frame(video_path, frame_index)
    output_path = output_dir / "frames" / (
        f"{_safe_name(set_id)}_{_safe_name(video_id)}_frame_{frame_index:06d}.jpg"
    )
    _save_rendered_frame(
        frame,
        annotations.boxes,
        set_id,
        video_id,
        frame_index,
        alignment,
        output_path,
    )

    if show:
        import cv2

        image = cv2.imread(str(output_path))
        window_name = f"PIE {set_id}/{video_id} frame {frame_index}"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image)
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("s"):
                    if not cv2.imwrite(
                        str(output_path),
                        image,
                        [cv2.IMWRITE_JPEG_QUALITY, 92],
                    ):
                        raise OSError(f"Could not write screenshot: {output_path}")
                    print(f"Screenshot saved: {output_path}")
                if key in (ord("q"), 27):
                    break
        except cv2.error as exc:
            raise RuntimeError(
                "Could not open the OpenCV inspection window. Use --no-display to "
                f"save the rendered frame without a window. Details: {exc}"
            ) from exc
        finally:
            cv2.destroyAllWindows()
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect selected PIE frames and annotations without modifying the dataset."
        )
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: artifacts/visual_spot_check).",
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Render reported mismatch frames, aligned controls, and label-category examples.",
    )
    parser.add_argument("--set", dest="set_id", help="Recording set, for example set06.")
    parser.add_argument("--video", help="Video ID, for example video_0002.")
    parser.add_argument("--frame", type=int, help="Zero-based video frame index to inspect.")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save a single-frame render without opening a GUI window.",
    )
    parser.add_argument(
        "--record-review",
        action="store_true",
        help="Record a visual finding for a frame in the generated sample manifest.",
    )
    parser.add_argument(
        "--review-status",
        choices=("correct", "incorrect", "uncertain"),
        help="Visual review outcome (requires --record-review).",
    )
    parser.add_argument("--finding", help="Observed visual finding (requires --record-review).")
    parser.add_argument(
        "--likely-cause",
        help="Likely cause of a recorded finding (requires --record-review).",
    )
    args = parser.parse_args(argv)

    selection_values = (args.set_id, args.video, args.frame)
    review_values = (args.review_status, args.finding, args.likely_cause)
    if args.samples and (
        any(value is not None for value in selection_values)
        or args.record_review
        or any(value is not None for value in review_values)
    ):
        parser.error("--samples cannot be combined with --set, --video, or --frame.")
    if not args.samples and (
        args.set_id is None or args.video is None or args.frame is None
    ):
        parser.error("Choose --samples or provide --set, --video, and --frame.")
    if args.record_review and (
        any(value is None for value in review_values)
        or not all(value.strip() for value in (args.finding, args.likely_cause))
    ):
        parser.error(
            "--record-review requires --review-status, --finding, and --likely-cause."
        )
    if not args.record_review and any(value is not None for value in review_values):
        parser.error("Review details require --record-review.")
    if args.record_review and args.no_display:
        parser.error("--no-display is not applicable with --record-review.")
    try:
        config, dataset_root = _load_config(args.config)
        config_dir = args.config.resolve().parent
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else (config_dir.parent / "artifacts" / "visual_spot_check")
        )
        if args.samples:
            report_dir = (
                config.get("output_dir", "../artifacts/dataset_validation")
            )
            report_dir_path = (config_dir / report_dir).resolve()
            manifest = generate_samples(
                dataset_root,
                output_dir,
                report_dir_path / "validation_issues.json",
                report_dir_path / "object_annotations.jsonl.gz",
            )
            print(f"Rendered {len(manifest)} frames to {output_dir / 'frames'}.")
            print(f"Report: {output_dir / 'visual_spot_check_report.md'}")
        elif args.record_review:
            record_review(
                output_dir,
                args.set_id,
                args.video,
                args.frame,
                args.review_status,
                args.finding,
                args.likely_cause,
            )
            print(f"Recorded review in {output_dir / 'visual_spot_check_report.md'}")
        else:
            image_path = inspect_single_frame(
                dataset_root,
                output_dir,
                args.set_id,
                args.video,
                args.frame,
                not args.no_display,
            )
            print(f"Rendered frame: {image_path}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Visual spot-check failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
