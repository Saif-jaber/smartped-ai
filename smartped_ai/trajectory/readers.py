from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..data.models import ValidationIssue
from ..data.prepare_detection import read_pedestrian_xml
from .models import TrajectoryConfig, TrackObservation

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
_DEFAULT_VIDEO_ID = "video"
_BBOX_KEYS = ("bbox_xyxy", "bbox")
_COORDINATE_KEYS = {
    "x1": ("x1", "bbox_x1", "xtl"),
    "y1": ("y1", "bbox_y1", "ytl"),
    "x2": ("x2", "bbox_x2", "xbr"),
    "y2": ("y2", "bbox_y2", "ybr"),
}
_XYWH_KEYS = {
    "x": ("x", "bbox_x"),
    "y": ("y", "bbox_y"),
    "w": ("w", "bbox_w", "width"),
    "h": ("h", "bbox_h", "height"),
}
_TRACK_ID_KEYS = ("track_id", "id", "pedestrian_id", "person_id")
_FRAME_KEYS = ("frame_index", "frame", "frame_id", "frame_number")
_TIMESTAMP_KEYS = ("timestamp_seconds", "timestamp", "time")
_SET_KEYS = ("set_id", "set", "recording_set")
_VIDEO_KEYS = ("video_id", "video")


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


def _first(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            value = mapping[name]
            if value is not None and str(value).strip() != "":
                return value
    return None


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        return int(number) if math.isfinite(number) and number.is_integer() else None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "occluded"}:
        return True
    if text in {"0", "false", "no", "n", "visible"}:
        return False
    return None


def _validate_bbox(
    values: Iterable[Any],
) -> tuple[tuple[float, float, float, float] | None, str | None]:
    coordinates: list[float] = []
    for value in values:
        number = _parse_float(value)
        if number is None:
            return None, "non_finite_or_missing_coordinate"
        coordinates.append(number)
    if len(coordinates) != 4:
        return None, "invalid_bbox_length"
    x1, y1, x2, y2 = coordinates
    if x1 >= x2 or y1 >= y2:
        return None, "non_positive_bbox"
    return (x1, y1, x2, y2), None


def _coerce_bbox_value(
    value: Any,
) -> tuple[tuple[float, float, float, float] | None, str | None]:
    if isinstance(value, Mapping):
        nested = {
            name: _first(value, names)
            for name, names in _COORDINATE_KEYS.items()
        }
        if any(item is None for item in nested.values()):
            return None, "missing_bbox_coordinate"
        return _validate_bbox(nested[name] for name in ("x1", "y1", "x2", "y2"))
    if isinstance(value, (list, tuple)):
        return _validate_bbox(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None, "missing_bbox"
        if text[0] in "[{(":
            try:
                return _coerce_bbox_value(json.loads(text))
            except json.JSONDecodeError:
                pass
        return _validate_bbox(
            item for item in re.split(r"[\s,;]+", text.strip("[](){}")) if item
        )
    return None, "invalid_bbox_value"


def _parse_bbox(
    mapping: Mapping[str, Any],
) -> tuple[tuple[float, float, float, float] | None, str | None]:
    for key in _BBOX_KEYS:
        if key in mapping and mapping[key] is not None and str(mapping[key]).strip() != "":
            return _coerce_bbox_value(mapping[key])

    coordinates = {
        name: _first(mapping, aliases) for name, aliases in _COORDINATE_KEYS.items()
    }
    if all(value is not None for value in coordinates.values()):
        return _validate_bbox(
            coordinates[name] for name in ("x1", "y1", "x2", "y2")
        )

    top_left = {name: _first(mapping, aliases) for name, aliases in _XYWH_KEYS.items()}
    if all(value is not None for value in top_left.values()):
        x = _parse_float(top_left["x"])
        y = _parse_float(top_left["y"])
        width = _parse_float(top_left["w"])
        height = _parse_float(top_left["h"])
        if None in {x, y, width, height}:
            return None, "non_finite_or_missing_coordinate"
        assert x is not None and y is not None
        assert width is not None and height is not None
        return _validate_bbox((x, y, x + width, y + height))

    return None, "missing_bbox"


def _observation_from_mapping(
    mapping: Mapping[str, Any],
    *,
    path: Path,
    config: TrajectoryConfig,
    default_set_id: str | None,
    default_video_id: str | None,
    issues: list[ValidationIssue],
) -> TrackObservation | None:
    track_id = _clean_optional_text(_first(mapping, _TRACK_ID_KEYS))
    frame_index = _parse_int(_first(mapping, _FRAME_KEYS))
    set_id = _clean_optional_text(_first(mapping, _SET_KEYS)) or default_set_id
    video_id = (
        _clean_optional_text(_first(mapping, _VIDEO_KEYS))
        or default_video_id
        or _DEFAULT_VIDEO_ID
    )
    if track_id is None:
        _issue(
            issues,
            "error",
            "missing_track_id",
            "Tracking record has no track ID and was skipped; no identity was created.",
            path,
            video_id,
            frame_index,
        )
        return None
    if frame_index is None:
        _issue(
            issues,
            "error",
            "invalid_frame_index",
            "Tracking record has a missing or invalid frame index and was skipped.",
            path,
            video_id,
            None,
            track_id,
        )
        return None
    bbox, bbox_error = _parse_bbox(mapping)
    if bbox is None:
        _issue(
            issues,
            "error",
            f"invalid_bbox_{bbox_error}",
            f"Tracking record bounding box is invalid ({bbox_error}) and was skipped.",
            path,
            video_id,
            frame_index,
            track_id,
        )
        return None

    raw_timestamp = _first(mapping, _TIMESTAMP_KEYS)
    timestamp: float | None = None
    if raw_timestamp is not None:
        timestamp = _parse_float(raw_timestamp)
        if timestamp is None or timestamp < 0:
            _issue(
                issues,
                "error",
                "invalid_timestamp",
                "Tracking record has an invalid timestamp and was skipped.",
                path,
                video_id,
                frame_index,
                track_id,
            )
            return None

    confidence = _parse_float(_first(mapping, ("confidence", "score", "conf")))
    raw_occluded = _first(mapping, ("occluded", "occlusion"))
    occluded = _parse_bool(raw_occluded)
    if raw_occluded is not None and occluded is None:
        _issue(
            issues,
            "warning",
            "invalid_occluded_value",
            "Unrecognized occluded value; it was stored as unknown.",
            path,
            video_id,
            frame_index,
            track_id,
        )

    try:
        return TrackObservation(
            track_id=track_id,
            frame_index=frame_index,
            bbox_xyxy=bbox,
            timestamp_seconds=timestamp,
            set_id=set_id,
            video_id=video_id,
            confidence=confidence,
            occluded=occluded,
            source=path.as_posix(),
        )
    except ValueError as exc:
        _issue(
            issues,
            "error",
            "invalid_tracking_record",
            f"Tracking record was skipped: {exc}",
            path,
            video_id,
            frame_index,
            track_id,
        )
        return None


def _read_csv(
    path: Path,
    config: TrajectoryConfig,
    issues: list[ValidationIssue],
    default_set_id: str | None,
    default_video_id: str | None,
) -> list[TrackObservation]:
    observations: list[TrackObservation] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise ValueError(f"Tracking CSV has no header: {path}")
            for row_number, row in enumerate(reader, start=2):
                if not row or not any(
                    value is not None and str(value).strip() for value in row.values()
                ):
                    continue
                observation = _observation_from_mapping(
                    row,
                    path=path,
                    config=config,
                    default_set_id=default_set_id,
                    default_video_id=default_video_id,
                    issues=issues,
                )
                if observation is None:
                    _issue(
                        issues,
                        "warning",
                        "skipped_tracking_csv_row",
                        f"CSV row {row_number} was skipped.",
                        path,
                    )
                else:
                    observations.append(observation)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read tracking CSV {path}: {exc}") from exc
    return observations


def _read_jsonl(
    path: Path,
    config: TrajectoryConfig,
    issues: list[ValidationIssue],
    default_set_id: str | None,
    default_video_id: str | None,
) -> list[TrackObservation]:
    observations: list[TrackObservation] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path}: {exc}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise ValueError(
                        f"Line {line_number} of {path} is not a JSON object."
                    )
                observation = _observation_from_mapping(
                    record,
                    path=path,
                    config=config,
                    default_set_id=default_set_id,
                    default_video_id=default_video_id,
                    issues=issues,
                )
                if observation is None:
                    _issue(
                        issues,
                        "warning",
                        "skipped_tracking_jsonl_record",
                        f"JSONL line {line_number} was skipped.",
                        path,
                    )
                else:
                    observations.append(observation)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read tracking JSONL {path}: {exc}") from exc
    return observations


def _infer_pie_identity(
    path: Path,
    default_set_id: str | None,
    default_video_id: str | None,
) -> tuple[str | None, str]:
    video_id = default_video_id
    if video_id is None:
        name = path.name
        video_id = name[: -len("_annt.xml")] if name.endswith("_annt.xml") else path.stem
    set_id = default_set_id
    if set_id is None and path.parent.name.startswith("set"):
        set_id = path.parent.name
    return set_id, video_id


def _read_pie_xml_path(
    path: Path,
    config: TrajectoryConfig,
    issues: list[ValidationIssue],
    default_set_id: str | None,
    default_video_id: str | None,
) -> list[TrackObservation]:
    set_id, video_id = _infer_pie_identity(path, default_set_id, default_video_id)
    try:
        _width, _height, boxes = read_pedestrian_xml(path)
    except ValueError as exc:
        raise ValueError(f"Could not read PIE pedestrian annotations: {exc}") from exc

    observations: list[TrackObservation] = []
    for box in boxes:
        if box.outside:
            continue
        if not box.track_id:
            _issue(
                issues,
                "error",
                "missing_track_id",
                "PIE pedestrian box has no track ID and was skipped.",
                path,
                video_id,
                box.frame_index,
            )
            continue
        bbox, bbox_error = _validate_bbox(box.bbox_xyxy)
        if bbox is None:
            _issue(
                issues,
                "warning",
                f"invalid_bbox_{bbox_error}",
                f"PIE pedestrian box is invalid ({bbox_error}) and was skipped.",
                path,
                video_id,
                box.frame_index,
                box.track_id,
            )
            continue
        observations.append(
            TrackObservation(
                track_id=box.track_id,
                frame_index=box.frame_index,
                bbox_xyxy=bbox,
                timestamp_seconds=box.frame_index / config.fps,
                set_id=set_id,
                video_id=video_id,
                occluded=box.occluded_raw == "1",
                source=path.as_posix(),
            )
        )
    return observations


def _read_pie_xml(
    source: Path,
    config: TrajectoryConfig,
    issues: list[ValidationIssue],
    default_set_id: str | None,
    default_video_id: str | None,
) -> list[TrackObservation]:
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.rglob("*_annt.xml"))
    else:
        raise FileNotFoundError(f"Tracking input does not exist: {source}")
    if not paths:
        raise FileNotFoundError(
            f"No PIE spatial annotation XML files (*_annt.xml) found under {source}"
        )

    observations: list[TrackObservation] = []
    for path in paths:
        observations.extend(
            _read_pie_xml_path(
                path,
                config,
                issues,
                default_set_id,
                default_video_id,
            )
        )
    return observations


def _video_fps(path: Path, fallback: float) -> float:
    try:
        import cv2
    except ImportError:
        return fallback
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
    finally:
        capture.release()
    return fps if math.isfinite(fps) and fps > 0 else fallback


def _read_video(
    source: Path,
    config: TrajectoryConfig,
    model_path: Path | None,
    tracker_config: Path | None,
    confidence: float,
    default_set_id: str | None,
    default_video_id: str | None,
    max_seconds: float | None,
) -> list[TrackObservation]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "The optional video adapter requires Ultralytics. Install it in the "
            "active environment or provide an existing JSONL/CSV tracking output."
        ) from exc

    if model_path is None or not model_path.is_file():
        raise FileNotFoundError(
            f"Detector weights were not found: {model_path}. Provide --model."
        )
    if tracker_config is not None and not tracker_config.is_file():
        raise FileNotFoundError(f"Tracker configuration was not found: {tracker_config}")

    fps = _video_fps(source, config.fps)
    max_frames = (
        math.ceil(max_seconds * fps) if max_seconds is not None else None
    )
    model = YOLO(str(model_path))
    track_options: dict[str, Any] = {
        "source": str(source),
        "stream": True,
        "persist": True,
        "conf": confidence,
        "classes": [0],
        "save": False,
        "show": False,
        "verbose": False,
    }
    if tracker_config is not None:
        track_options["tracker"] = str(tracker_config)

    video_id = default_video_id or source.stem
    observations: list[TrackObservation] = []
    for frame_index, result in enumerate(model.track(**track_options)):
        if max_frames is not None and frame_index >= max_frames:
            break
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            continue
        track_ids = boxes.id.cpu().numpy().tolist()
        coordinates = boxes.xyxy.cpu().numpy().tolist()
        confidences = (
            boxes.conf.cpu().numpy().tolist()
            if boxes.conf is not None
            else [None] * len(track_ids)
        )
        for track_id, bbox_values, box_confidence in zip(
            track_ids, coordinates, confidences
        ):
            bbox, bbox_error = _validate_bbox(bbox_values)
            if bbox is None:
                continue
            observations.append(
                TrackObservation(
                    track_id=str(int(track_id)),
                    frame_index=frame_index,
                    bbox_xyxy=bbox,
                    timestamp_seconds=frame_index / fps,
                    set_id=default_set_id,
                    video_id=video_id,
                    confidence=(
                        float(box_confidence)
                        if box_confidence is not None
                        else None
                    ),
                    source=source.as_posix(),
                )
            )
    return observations


def _ensure_unique_observations(observations: list[TrackObservation]) -> None:
    seen: dict[tuple[tuple[str, str, str], int], TrackObservation] = {}
    for observation in observations:
        key = (observation.identity, observation.frame_index)
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "Duplicate observation for "
                f"{observation.set_id or '-'} / {observation.video_id or '-'} / "
                f"track {observation.track_id} at frame {observation.frame_index}. "
                "Input records were not merged or re-associated."
            )
        seen[key] = observation


def _resolve_format(source: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    if source.is_dir():
        return "pie-xml"
    suffix = source.suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        return "video"
    if suffix == ".csv":
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".xml":
        return "pie-xml"
    raise ValueError(
        f"Cannot infer the tracking input format from {source.name}. "
        "Provide --input-format jsonl, csv, pie-xml, or video."
    )


def load_observations(
    source: Path,
    config: TrajectoryConfig,
    *,
    input_format: str = "auto",
    default_set_id: str | None = None,
    default_video_id: str | None = None,
    model_path: Path | None = None,
    tracker_config: Path | None = None,
    confidence: float = 0.25,
    max_seconds: float | None = None,
) -> tuple[list[TrackObservation], list[ValidationIssue], str]:
    config.validate()
    source = source.resolve()
    resolved_format = _resolve_format(source, input_format)
    if max_seconds is not None:
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be finite and greater than zero.")
        if resolved_format != "video":
            raise ValueError("max_seconds is only supported for video input.")
    issues: list[ValidationIssue] = []

    if resolved_format == "csv":
        if not source.is_file():
            raise FileNotFoundError(f"Tracking CSV does not exist: {source}")
        observations = _read_csv(
            source, config, issues, default_set_id, default_video_id
        )
    elif resolved_format == "jsonl":
        if not source.is_file():
            raise FileNotFoundError(f"Tracking JSONL does not exist: {source}")
        observations = _read_jsonl(
            source, config, issues, default_set_id, default_video_id
        )
    elif resolved_format == "pie-xml":
        observations = _read_pie_xml(
            source, config, issues, default_set_id, default_video_id
        )
    elif resolved_format == "video":
        if not source.is_file():
            raise FileNotFoundError(f"Video does not exist: {source}")
        observations = _read_video(
            source,
            config,
            model_path,
            tracker_config,
            confidence,
            default_set_id,
            default_video_id,
            max_seconds,
        )
    else:
        raise ValueError(f"Unsupported tracking input format: {resolved_format}")

    _ensure_unique_observations(observations)
    observations.sort(
        key=lambda observation: (
            observation.set_id or "",
            observation.video_id or "",
            observation.track_id,
            observation.frame_index,
            observation.timestamp_seconds
            if observation.timestamp_seconds is not None
            else -1.0,
        )
    )
    return observations, issues, resolved_format
