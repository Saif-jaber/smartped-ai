from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "interaction_analysis.json"
_OBSERVATION_FIELDS = (
    "set_id",
    "video_id",
    "frame_index",
    "timestamp_seconds",
    "timestamp_validation_status",
    "timestamp_delta_seconds",
    "pedestrian_track_id",
    "pedestrian_footpoint_x_px",
    "pedestrian_footpoint_y_px",
    "pedestrian_bbox_x1",
    "pedestrian_bbox_y1",
    "pedestrian_bbox_x2",
    "pedestrian_bbox_y2",
    "vehicle_track_id",
    "vehicle_class",
    "vehicle_bbox_x1",
    "vehicle_bbox_y1",
    "vehicle_bbox_x2",
    "vehicle_bbox_y2",
    "vehicle_reference_x_px",
    "vehicle_reference_y_px",
    "nearest_vehicle_point_x_px",
    "nearest_vehicle_point_y_px",
    "distance_pixels",
    "proximity_status",
    "pedestrian_displacement_x_px",
    "pedestrian_displacement_y_px",
    "vehicle_displacement_x_px",
    "vehicle_displacement_y_px",
    "relative_displacement_x_px",
    "relative_displacement_y_px",
    "relative_movement_direction_degrees",
    "separation_change_pixels",
    "movement_status",
    "movement_valid",
    "movement_invalid_reason",
    "frame_gap_from_previous_pair_observation",
    "time_gap_seconds_from_previous_pair_observation",
)


@dataclass(frozen=True)
class InteractionConfig:
    pedestrian_trajectory_path: Path
    vehicle_detection_path: Path
    vehicle_trajectory_path: Path
    video_path: Path
    output_dir: Path
    set_ids: tuple[str, ...]
    video_ids: tuple[str, ...]
    max_observation_gap_frames: int
    min_observations_for_movement: int
    min_observations_for_closest_approach: int
    pixel_distance_threshold: float
    timestamp_tolerance_seconds: float
    movement_delta_tolerance_pixels: float
    visual_sample_count: int

    def validate(self) -> None:
        for name, value in (
            ("max_observation_gap_frames", self.max_observation_gap_frames),
            ("min_observations_for_movement", self.min_observations_for_movement),
            (
                "min_observations_for_closest_approach",
                self.min_observations_for_closest_approach,
            ),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name, value in (
            ("pixel_distance_threshold", self.pixel_distance_threshold),
            ("timestamp_tolerance_seconds", self.timestamp_tolerance_seconds),
            (
                "movement_delta_tolerance_pixels",
                self.movement_delta_tolerance_pixels,
            ),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if (
            not isinstance(self.visual_sample_count, int)
            or isinstance(self.visual_sample_count, bool)
            or self.visual_sample_count < 0
        ):
            raise ValueError("visual_sample_count must be a non-negative integer.")


@dataclass(frozen=True)
class Observation:
    set_id: str
    video_id: str
    track_id: str
    frame_index: int
    timestamp_seconds: float | None
    timestamp_issue: str | None
    bbox_xyxy: tuple[float, float, float, float]
    class_name: str | None = None
    confidence: float | None = None

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.set_id, self.video_id, self.track_id

    @property
    def frame_identity(self) -> tuple[str, str, int]:
        return self.set_id, self.video_id, self.frame_index

    @property
    def entity_frame_identity(self) -> tuple[str, str, str, int]:
        return self.set_id, self.video_id, self.track_id, self.frame_index


def _resolve_path(config_path: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string.")
    path = Path(value)
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    return path.resolve()


def _text_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{name} must be an array of non-empty strings.")
    return tuple(item.strip() for item in value)


def load_config(path: Path = _DEFAULT_CONFIG) -> InteractionConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read interaction config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Interaction config must be a JSON object.")

    def integer(name: str) -> int:
        value = raw.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer.")
        return value

    def number(name: str) -> float:
        value = raw.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be numeric.")
        return float(value)

    config = InteractionConfig(
        pedestrian_trajectory_path=_resolve_path(
            path, raw.get("pedestrian_trajectory_path"), "pedestrian_trajectory_path"
        ),
        vehicle_detection_path=_resolve_path(
            path, raw.get("vehicle_detection_path"), "vehicle_detection_path"
        ),
        vehicle_trajectory_path=_resolve_path(
            path, raw.get("vehicle_trajectory_path"), "vehicle_trajectory_path"
        ),
        video_path=_resolve_path(path, raw.get("video_path"), "video_path"),
        output_dir=_resolve_path(path, raw.get("output_dir"), "output_dir"),
        set_ids=_text_list(raw.get("set_ids"), "set_ids"),
        video_ids=_text_list(raw.get("video_ids"), "video_ids"),
        max_observation_gap_frames=integer("max_observation_gap_frames"),
        min_observations_for_movement=integer("min_observations_for_movement"),
        min_observations_for_closest_approach=integer(
            "min_observations_for_closest_approach"
        ),
        pixel_distance_threshold=number("pixel_distance_threshold"),
        timestamp_tolerance_seconds=number("timestamp_tolerance_seconds"),
        movement_delta_tolerance_pixels=number("movement_delta_tolerance_pixels"),
        visual_sample_count=integer("visual_sample_count"),
    )
    config.validate()
    return config


def _read_csv(path: Path, required_columns: Iterable[str]) -> tuple[list[dict[str, str]], int]:
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
                    f"CSV {path} is missing required columns: {', '.join(sorted(missing))}"
                )
            rows = [
                {key: value or "" for key, value in row.items() if key is not None}
                for row in reader
                if row and any(value and value.strip() for value in row.values() if value)
            ]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read CSV {path}: {exc}") from exc
    return rows, len(rows)


def _parse_frame(value: str) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _parse_timestamp(value: str) -> tuple[float | None, str | None]:
    if not value.strip():
        return None, "missing_timestamp"
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None, "invalid_timestamp"
    if not math.isfinite(result) or result < 0:
        return None, "invalid_timestamp"
    return result, None


def _parse_bbox(
    row: dict[str, str], prefix: str = "bbox_"
) -> tuple[float, float, float, float] | None:
    try:
        bbox = tuple(
            float(row[f"{prefix}{coordinate}"])
            for coordinate in ("x1", "y1", "x2", "y2")
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        len(bbox) != 4
        or not all(math.isfinite(value) for value in bbox)
        or bbox[0] >= bbox[2]
        or bbox[1] >= bbox[3]
    ):
        return None
    return bbox  # type: ignore[return-value]


def _within_filter(
    set_id: str, video_id: str, config: InteractionConfig
) -> bool:
    return (not config.set_ids or set_id in config.set_ids) and (
        not config.video_ids or video_id in config.video_ids
    )


def _observation_from_row(
    row: dict[str, str],
    *,
    config: InteractionConfig,
    bbox_prefix: str = "bbox_",
    class_name: str | None = None,
    require_class_name: bool = False,
) -> tuple[Observation | None, str | None]:
    set_id = row.get("set_id", "").strip()
    video_id = row.get("video_id", "").strip()
    track_id = row.get("track_id", "").strip()
    frame_index = _parse_frame(row.get("frame_index", ""))
    if not set_id or not video_id:
        return None, "missing_recording_identity"
    if frame_index is None:
        return None, "invalid_frame_index"
    if not track_id:
        return None, "missing_track_id"
    if not _within_filter(set_id, video_id, config):
        return None, "filtered_out"
    resolved_class = class_name or row.get("class_name", "").strip() or None
    if require_class_name and resolved_class is None:
        return None, "missing_class_name"
    bbox = _parse_bbox(row, bbox_prefix)
    if bbox is None:
        return None, "invalid_bbox"
    timestamp, timestamp_issue = _parse_timestamp(
        row.get("timestamp_seconds", "")
    )
    confidence: float | None = None
    raw_confidence = row.get("confidence", "").strip()
    if raw_confidence:
        try:
            value = float(raw_confidence)
            confidence = value if math.isfinite(value) else None
        except ValueError:
            confidence = None
    return (
        Observation(
            set_id=set_id,
            video_id=video_id,
            track_id=track_id,
            frame_index=frame_index,
            timestamp_seconds=timestamp,
            timestamp_issue=timestamp_issue,
            bbox_xyxy=bbox,
            class_name=resolved_class,
            confidence=confidence,
        ),
        None,
    )


def _deduplicate(
    observations: Iterable[Observation],
    *,
    include_class: bool = False,
) -> tuple[list[Observation], int]:
    unique: dict[tuple[Any, ...], Observation] = {}
    duplicates = 0
    for observation in observations:
        key: tuple[Any, ...] = observation.entity_frame_identity
        if include_class:
            key += (observation.class_name,)
        previous = unique.get(key)
        if previous is None:
            unique[key] = observation
            continue
        if previous != observation:
            raise ValueError(
                "Conflicting duplicate observations for "
                f"{observation.set_id}/{observation.video_id}/"
                f"{observation.track_id}/frame {observation.frame_index}."
            )
        duplicates += 1
    return list(unique.values()), duplicates


def _load_pedestrians(
    config: InteractionConfig,
) -> tuple[list[Observation], dict[str, Any]]:
    rows, raw_count = _read_csv(
        config.pedestrian_trajectory_path,
        (
            "set_id",
            "video_id",
            "track_id",
            "frame_index",
            "timestamp_seconds",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
        ),
    )
    observations: list[Observation] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        observation, reason = _observation_from_row(
            row, config=config, bbox_prefix="bbox_"
        )
        if observation is not None:
            observations.append(observation)
        elif reason != "filtered_out":
            skipped[reason or "unknown"] += 1
    observations, duplicates = _deduplicate(observations)
    return observations, {
        "rows_read": raw_count,
        "selected_observations": len(observations),
        "unique_tracks": len({observation.identity for observation in observations}),
        "duplicates_removed": duplicates,
        "invalid_rows_skipped": dict(sorted(skipped.items())),
    }


def _load_vehicles(
    config: InteractionConfig,
) -> tuple[list[Observation], dict[str, Any]]:
    detection_rows, detection_raw_count = _read_csv(
        config.vehicle_detection_path,
        (
            "set_id",
            "video_id",
            "frame_index",
            "timestamp_seconds",
            "class_name",
            "track_id",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
        ),
    )
    trajectory_rows, trajectory_raw_count = _read_csv(
        config.vehicle_trajectory_path,
        (
            "set_id",
            "video_id",
            "frame_index",
            "timestamp_seconds",
            "class_name",
            "track_id",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
        ),
    )

    detection_observations: list[Observation] = []
    detection_skipped: Counter[str] = Counter()
    untracked_detections = 0
    for row in detection_rows:
        if not _within_filter(
            row.get("set_id", "").strip(),
            row.get("video_id", "").strip(),
            config,
        ):
            continue
        if not row.get("track_id", "").strip():
            untracked_detections += 1
            continue
        observation, reason = _observation_from_row(
            row,
            config=config,
            bbox_prefix="bbox_",
            class_name=row.get("class_name", "").strip() or None,
            require_class_name=True,
        )
        if observation is not None:
            detection_observations.append(observation)
        elif reason != "filtered_out":
            detection_skipped[reason or "unknown"] += 1
    detection_observations, detection_duplicates = _deduplicate(
        detection_observations, include_class=True
    )
    detection_by_key = {
        (
            item.set_id,
            item.video_id,
            item.class_name,
            item.track_id,
            item.frame_index,
        ): item
        for item in detection_observations
    }

    trajectory_observations: list[Observation] = []
    trajectory_skipped: Counter[str] = Counter()
    for row in trajectory_rows:
        observation, reason = _observation_from_row(
            row,
            config=config,
            bbox_prefix="bbox_",
            class_name=row.get("class_name", "").strip() or None,
            require_class_name=True,
        )
        if observation is not None:
            trajectory_observations.append(observation)
        elif reason != "filtered_out":
            trajectory_skipped[reason or "unknown"] += 1
    trajectory_observations, trajectory_duplicates = _deduplicate(
        trajectory_observations, include_class=True
    )

    matched = 0
    missing_detection = 0
    bbox_mismatch = 0
    timestamp_mismatch = 0
    vehicles: list[Observation] = []
    for trajectory in trajectory_observations:
        key = (
            trajectory.set_id,
            trajectory.video_id,
            trajectory.class_name,
            trajectory.track_id,
            trajectory.frame_index,
        )
        detection = detection_by_key.get(key)
        if detection is None:
            missing_detection += 1
            vehicles.append(trajectory)
            continue
        matched += 1
        if any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-4)
            for a, b in zip(trajectory.bbox_xyxy, detection.bbox_xyxy)
        ):
            bbox_mismatch += 1
        if (
            trajectory.timestamp_seconds is None
            or detection.timestamp_seconds is None
            or abs(
                trajectory.timestamp_seconds - detection.timestamp_seconds
            )
            > config.timestamp_tolerance_seconds
        ):
            timestamp_mismatch += 1
        vehicles.append(
            Observation(
                set_id=trajectory.set_id,
                video_id=trajectory.video_id,
                track_id=trajectory.track_id,
                frame_index=trajectory.frame_index,
                timestamp_seconds=trajectory.timestamp_seconds,
                timestamp_issue=trajectory.timestamp_issue,
                bbox_xyxy=trajectory.bbox_xyxy,
                class_name=trajectory.class_name,
                confidence=detection.confidence,
            )
        )

    unmatched_tracked_detections = len(detection_observations) - matched
    vehicle_deduplicated, final_duplicates = _deduplicate(
        vehicles, include_class=True
    )
    return vehicle_deduplicated, {
        "detection_rows_read": detection_raw_count,
        "trajectory_rows_read": trajectory_raw_count,
        "selected_tracked_detection_observations": len(detection_observations),
        "selected_trajectory_observations": len(trajectory_observations),
        "unique_tracks": len(
            {
                (item.set_id, item.video_id, item.class_name, item.track_id)
                for item in vehicle_deduplicated
            }
        ),
        "untracked_detections": untracked_detections,
        "tracked_detections_without_trajectory": unmatched_tracked_detections,
        "trajectory_observations_without_detection": missing_detection,
        "matched_detection_trajectory_observations": matched,
        "matched_bbox_mismatches": bbox_mismatch,
        "matched_timestamp_mismatches": timestamp_mismatch,
        "detection_duplicates_removed": detection_duplicates,
        "trajectory_duplicates_removed": trajectory_duplicates,
        "final_duplicates_removed": final_duplicates,
        "invalid_detection_rows_skipped": dict(sorted(detection_skipped.items())),
        "invalid_trajectory_rows_skipped": dict(sorted(trajectory_skipped.items())),
    }


def pedestrian_footpoint(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float]:
    x1, _, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2.0, y2


def vehicle_reference_point(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def nearest_point_on_box(
    point: tuple[float, float],
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float]:
    x, y = point
    x1, y1, x2, y2 = bbox_xyxy
    return min(max(x, x1), x2), min(max(y, y1), y2)


def pixel_box_distance(
    point: tuple[float, float],
    bbox_xyxy: tuple[float, float, float, float],
) -> float:
    nearest = nearest_point_on_box(point, bbox_xyxy)
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def _timestamp_status(
    pedestrian: Observation,
    vehicle: Observation,
    tolerance_seconds: float,
) -> tuple[str, float | None]:
    if pedestrian.timestamp_issue == "invalid_timestamp" or vehicle.timestamp_issue == "invalid_timestamp":
        return "invalid_timestamp", None
    if pedestrian.timestamp_seconds is None or vehicle.timestamp_seconds is None:
        return "missing_timestamp", None
    delta = abs(pedestrian.timestamp_seconds - vehicle.timestamp_seconds)
    if delta > tolerance_seconds:
        return "timestamp_mismatch", delta
    return "consistent", delta


def _direction_degrees(dx: float, dy: float) -> float | None:
    if dx == 0.0 and dy == 0.0:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _pair_key(
    pedestrian: Observation, vehicle: Observation
) -> tuple[str, str, str, str, str]:
    return (
        pedestrian.set_id,
        pedestrian.video_id,
        pedestrian.track_id,
        vehicle.class_name or "",
        vehicle.track_id,
    )


def _base_pair_row(
    pedestrian: Observation,
    vehicle: Observation,
    config: InteractionConfig,
) -> dict[str, Any]:
    footpoint = pedestrian_footpoint(pedestrian.bbox_xyxy)
    nearest = nearest_point_on_box(footpoint, vehicle.bbox_xyxy)
    distance = math.hypot(footpoint[0] - nearest[0], footpoint[1] - nearest[1])
    center = vehicle_reference_point(vehicle.bbox_xyxy)
    timestamp_status, timestamp_delta = _timestamp_status(
        pedestrian, vehicle, config.timestamp_tolerance_seconds
    )
    return {
        "set_id": pedestrian.set_id,
        "video_id": pedestrian.video_id,
        "frame_index": pedestrian.frame_index,
        "timestamp_seconds": (
            pedestrian.timestamp_seconds
            if timestamp_status == "consistent"
            else None
        ),
        "timestamp_validation_status": timestamp_status,
        "timestamp_delta_seconds": timestamp_delta,
        "pedestrian_track_id": pedestrian.track_id,
        "pedestrian_footpoint_x_px": footpoint[0],
        "pedestrian_footpoint_y_px": footpoint[1],
        "pedestrian_bbox_x1": pedestrian.bbox_xyxy[0],
        "pedestrian_bbox_y1": pedestrian.bbox_xyxy[1],
        "pedestrian_bbox_x2": pedestrian.bbox_xyxy[2],
        "pedestrian_bbox_y2": pedestrian.bbox_xyxy[3],
        "vehicle_track_id": vehicle.track_id,
        "vehicle_class": vehicle.class_name or "",
        "vehicle_bbox_x1": vehicle.bbox_xyxy[0],
        "vehicle_bbox_y1": vehicle.bbox_xyxy[1],
        "vehicle_bbox_x2": vehicle.bbox_xyxy[2],
        "vehicle_bbox_y2": vehicle.bbox_xyxy[3],
        "vehicle_reference_x_px": center[0],
        "vehicle_reference_y_px": center[1],
        "nearest_vehicle_point_x_px": nearest[0],
        "nearest_vehicle_point_y_px": nearest[1],
        "distance_pixels": distance,
        "proximity_status": (
            "nearby"
            if distance <= config.pixel_distance_threshold
            else "not_nearby"
        ),
    }


def _attach_pair_movement(
    row: dict[str, Any],
    previous: dict[str, Any] | None,
    config: InteractionConfig,
) -> None:
    row.update(
        {
            "pedestrian_displacement_x_px": None,
            "pedestrian_displacement_y_px": None,
            "vehicle_displacement_x_px": None,
            "vehicle_displacement_y_px": None,
            "relative_displacement_x_px": None,
            "relative_displacement_y_px": None,
            "relative_movement_direction_degrees": None,
            "separation_change_pixels": None,
            "movement_status": "indeterminate",
            "movement_valid": False,
            "movement_invalid_reason": "first_pair_observation",
            "frame_gap_from_previous_pair_observation": None,
            "time_gap_seconds_from_previous_pair_observation": None,
        }
    )
    if previous is None:
        return

    frame_gap = row["frame_index"] - previous["frame_index"]
    row["frame_gap_from_previous_pair_observation"] = frame_gap
    if frame_gap <= 0:
        row["movement_invalid_reason"] = "nonpositive_frame_gap"
        return
    if frame_gap > config.max_observation_gap_frames:
        row["movement_invalid_reason"] = "frame_gap_exceeded"
        return
    if (
        row["timestamp_validation_status"] != "consistent"
        or previous["timestamp_validation_status"] != "consistent"
    ):
        row["movement_invalid_reason"] = (
            "invalid_timestamp_alignment"
            if row["timestamp_validation_status"] == "invalid_timestamp"
            or previous["timestamp_validation_status"] == "invalid_timestamp"
            else (
                "timestamp_mismatch"
                if row["timestamp_validation_status"] == "timestamp_mismatch"
                or previous["timestamp_validation_status"] == "timestamp_mismatch"
                else "missing_timestamp"
            )
        )
        return
    time_gap = row["timestamp_seconds"] - previous["timestamp_seconds"]
    row["time_gap_seconds_from_previous_pair_observation"] = time_gap
    if not math.isfinite(time_gap) or time_gap <= 0:
        row["movement_invalid_reason"] = "nonpositive_time_gap"
        return

    ped_dx = (
        row["pedestrian_footpoint_x_px"]
        - previous["pedestrian_footpoint_x_px"]
    )
    ped_dy = (
        row["pedestrian_footpoint_y_px"]
        - previous["pedestrian_footpoint_y_px"]
    )
    vehicle_dx = row["vehicle_reference_x_px"] - previous["vehicle_reference_x_px"]
    vehicle_dy = row["vehicle_reference_y_px"] - previous["vehicle_reference_y_px"]
    relative_dx = vehicle_dx - ped_dx
    relative_dy = vehicle_dy - ped_dy
    separation_change = row["distance_pixels"] - previous["distance_pixels"]
    row.update(
        {
            "pedestrian_displacement_x_px": ped_dx,
            "pedestrian_displacement_y_px": ped_dy,
            "vehicle_displacement_x_px": vehicle_dx,
            "vehicle_displacement_y_px": vehicle_dy,
            "relative_displacement_x_px": relative_dx,
            "relative_displacement_y_px": relative_dy,
            "relative_movement_direction_degrees": _direction_degrees(
                relative_dx, relative_dy
            ),
            "separation_change_pixels": separation_change,
            "movement_status": (
                "approaching"
                if separation_change < -config.movement_delta_tolerance_pixels
                else (
                    "separating"
                    if separation_change > config.movement_delta_tolerance_pixels
                    else "indeterminate"
                )
            ),
            "movement_valid": True,
            "movement_invalid_reason": "",
        }
    )


def generate_interaction_rows(
    pedestrians: Iterable[Observation],
    vehicles: Iterable[Observation],
    config: InteractionConfig,
) -> list[dict[str, Any]]:
    config.validate()
    pedestrians, _ = _deduplicate(pedestrians)
    vehicles, _ = _deduplicate(vehicles, include_class=True)
    pedestrian_frames: dict[
        tuple[str, str, int], list[Observation]
    ] = defaultdict(list)
    vehicle_frames: dict[tuple[str, str, int], list[Observation]] = defaultdict(list)
    for pedestrian in pedestrians:
        pedestrian_frames[pedestrian.frame_identity].append(pedestrian)
    for vehicle in vehicles:
        vehicle_frames[vehicle.frame_identity].append(vehicle)

    pair_rows: dict[
        tuple[str, str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for frame_identity in sorted(set(pedestrian_frames) & set(vehicle_frames)):
        for pedestrian in pedestrian_frames[frame_identity]:
            for vehicle in vehicle_frames[frame_identity]:
                row = _base_pair_row(pedestrian, vehicle, config)
                pair_rows[_pair_key(pedestrian, vehicle)].append(row)

    rows: list[dict[str, Any]] = []
    for key in sorted(pair_rows):
        ordered = sorted(
            pair_rows[key],
            key=lambda row: (row["frame_index"], row["timestamp_seconds"] or -1),
        )
        previous: dict[str, Any] | None = None
        for row in ordered:
            _attach_pair_movement(row, previous, config)
            rows.append(row)
            previous = row
    return sorted(
        rows,
        key=lambda row: (
            row["set_id"],
            row["video_id"],
            row["frame_index"],
            row["pedestrian_track_id"],
            row["vehicle_class"],
            row["vehicle_track_id"],
        ),
    )


def _valid_segments(
    rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    segments: list[list[dict[str, Any]]] = [[rows[0]]]
    for row in rows[1:]:
        if row["movement_valid"]:
            segments[-1].append(row)
        else:
            segments.append([row])
    return segments


def summarize_interactions(
    rows: Iterable[dict[str, Any]],
    config: InteractionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
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

    summaries: list[dict[str, Any]] = []
    movement_valid_count = 0
    closest_valid_count = 0
    for key in sorted(grouped):
        observations = sorted(grouped[key], key=lambda row: row["frame_index"])
        if len(observations) < 2:
            continue
        distances = [float(row["distance_pixels"]) for row in observations]
        movement_edges = sum(bool(row["movement_valid"]) for row in observations)
        segments = _valid_segments(observations)
        movement_segment = max(
            segments,
            key=lambda segment: (len(segment), -segment[0]["frame_index"]),
        )
        movement_valid = (
            len(movement_segment) >= config.min_observations_for_movement
        )
        if movement_valid:
            movement_valid_count += 1
            delta = (
                movement_segment[-1]["distance_pixels"]
                - movement_segment[0]["distance_pixels"]
            )
            movement_status = (
                "approaching"
                if delta < -config.movement_delta_tolerance_pixels
                else (
                    "separating"
                    if delta > config.movement_delta_tolerance_pixels
                    else "indeterminate"
                )
            )
            movement_invalid_reason = ""
        else:
            movement_status = "indeterminate"
            movement_invalid_reason = (
                "insufficient_consecutive_valid_observations"
                if movement_edges
                else "no_valid_movement_edges"
            )

        closest_segment = max(
            segments, key=lambda segment: (len(segment), -segment[0]["frame_index"])
        )
        closest_valid = (
            len(closest_segment) >= config.min_observations_for_closest_approach
            and all(
                row["timestamp_validation_status"] == "consistent"
                for row in closest_segment
            )
        )
        if closest_valid:
            closest_valid_count += 1
            closest_row = min(
                closest_segment,
                key=lambda row: (row["distance_pixels"], row["frame_index"]),
            )
            closest_frame: int | None = closest_row["frame_index"]
            closest_time: float | None = closest_row["timestamp_seconds"]
            closest_invalid_reason = ""
        else:
            closest_frame = None
            closest_time = None
            closest_invalid_reason = (
                "insufficient_consecutive_valid_observations"
                if len(closest_segment)
                < config.min_observations_for_closest_approach
                else "invalid_timestamp_alignment"
            )
        summaries.append(
            {
                "set_id": key[0],
                "video_id": key[1],
                "pedestrian_track_id": key[2],
                "vehicle_class": key[3],
                "vehicle_track_id": key[4],
                "observation_count": len(observations),
                "first_frame_index": observations[0]["frame_index"],
                "last_frame_index": observations[-1]["frame_index"],
                "minimum_separation_pixels": min(distances),
                "median_separation_pixels": statistics.median(distances),
                "movement_valid_edge_count": movement_edges,
                "movement_status": movement_status,
                "movement_analysis_valid": movement_valid,
                "movement_invalid_reason": movement_invalid_reason,
                "closest_approach_type": "minimum_observed_separation",
                "closest_approach_frame_index": closest_frame,
                "closest_approach_timestamp_seconds": closest_time,
                "closest_approach_separation_pixels": (
                    min(
                        float(row["distance_pixels"])
                        for row in closest_segment
                    )
                    if closest_valid
                    else None
                ),
                "closest_approach_valid": closest_valid,
                "closest_approach_invalid_reason": closest_invalid_reason,
                "closest_approach_assumptions": (
                    "Observed minimum within the longest consecutive valid pair "
                    "segment; not a future projection or TTC."
                ),
            }
        )
    return summaries, {
        "persistent_pair_count": len(summaries),
        "transient_pair_count": sum(
            len(pair_rows) == 1 for pair_rows in grouped.values()
        ),
        "movement_analysis_valid_persistent_pairs": movement_valid_count,
        "movement_analysis_invalid_persistent_pairs": (
            len(summaries) - movement_valid_count
        ),
        "closest_approach_valid_persistent_pairs": closest_valid_count,
        "closest_approach_invalid_persistent_pairs": (
            len(summaries) - closest_valid_count
        ),
    }


def _track_gap_statistics(
    observations: Iterable[Observation],
    *,
    include_class: bool,
    max_gap: int,
) -> dict[str, int]:
    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for observation in observations:
        key: tuple[Any, ...] = observation.identity
        if include_class:
            key += (observation.class_name,)
        grouped[key].append(observation.frame_index)
    gap_count = 0
    missing_slots = 0
    over_limit = 0
    for frame_indices in grouped.values():
        ordered = sorted(set(frame_indices))
        for previous, current in zip(ordered, ordered[1:]):
            gap = current - previous
            if gap > 1:
                gap_count += 1
                missing_slots += gap - 1
                if gap > max_gap:
                    over_limit += 1
    return {
        "track_count": len(grouped),
        "observation_count": sum(len(set(frames)) for frames in grouped.values()),
        "gap_count_within_observed_track_spans": gap_count,
        "unobserved_frame_slots_within_track_spans": missing_slots,
        "gaps_exceeding_maximum": over_limit,
    }


def _metrics(
    pedestrians: list[Observation],
    vehicles: list[Observation],
    rows: list[dict[str, Any]],
    summary_metrics: dict[str, Any],
    pedestrian_inputs: dict[str, Any],
    vehicle_inputs: dict[str, Any],
    config: InteractionConfig,
) -> dict[str, Any]:
    frame_timestamps: dict[
        tuple[str, str, int], list[float | None]
    ] = defaultdict(list)
    for observation in pedestrians + vehicles:
        frame_timestamps[observation.frame_identity].append(
            observation.timestamp_seconds
        )
    common_frames = {
        observation.frame_identity for observation in pedestrians
    } & {observation.frame_identity for observation in vehicles}
    mismatch_frames = 0
    invalid_timestamp_frames = 0
    valid_timestamp_frames = 0
    maximum_timestamp_delta: float | None = None
    for frame_identity in common_frames:
        timestamps = frame_timestamps[frame_identity]
        if any(value is None for value in timestamps):
            invalid_timestamp_frames += 1
            continue
        valid = [value for value in timestamps if value is not None]
        delta = max(valid) - min(valid)
        maximum_timestamp_delta = (
            delta
            if maximum_timestamp_delta is None
            else max(maximum_timestamp_delta, delta)
        )
        if delta > config.timestamp_tolerance_seconds:
            mismatch_frames += 1
        else:
            valid_timestamp_frames += 1

    pair_counts: Counter[tuple[str, str, str, str, str]] = Counter(
        (
            row["set_id"],
            row["video_id"],
            row["pedestrian_track_id"],
            row["vehicle_class"],
            row["vehicle_track_id"],
        )
        for row in rows
    )
    movement_valid = sum(bool(row["movement_valid"]) for row in rows)
    invalid_movement_reasons = Counter(
        row["movement_invalid_reason"]
        for row in rows
        if not row["movement_valid"]
    )
    pair_gap_count = 0
    pair_missing_slots = 0
    pair_gaps_exceeding_limit = 0
    grouped_frames: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
    for row in rows:
        key = (
            row["set_id"],
            row["video_id"],
            row["pedestrian_track_id"],
            row["vehicle_class"],
            row["vehicle_track_id"],
        )
        grouped_frames[key].append(row["frame_index"])
    for frames in grouped_frames.values():
        ordered = sorted(set(frames))
        for previous, current in zip(ordered, ordered[1:]):
            gap = current - previous
            if gap > 1:
                pair_gap_count += 1
                pair_missing_slots += gap - 1
                if gap > config.max_observation_gap_frames:
                    pair_gaps_exceeding_limit += 1

    return {
        "experiment": {
            "set_ids": list(config.set_ids),
            "video_ids": list(config.video_ids),
            "measurements_are_image_space_only": True,
            "distance_unit": "pixels",
            "no_collision_or_risk_classification": True,
        },
        "inputs": {
            "pedestrian": pedestrian_inputs,
            "vehicle": vehicle_inputs,
        },
        "pedestrian_count": len(pedestrians),
        "vehicle_track_observation_count": len(vehicles),
        "unique_pedestrian_tracks": len({item.identity for item in pedestrians}),
        "unique_vehicle_tracks": len(
            {
                (item.set_id, item.video_id, item.class_name, item.track_id)
                for item in vehicles
            }
        ),
        "candidate_pair_observations": len(rows),
        "candidate_unique_track_pairs": len(pair_counts),
        "persistent_pair_count": summary_metrics["persistent_pair_count"],
        "transient_pair_count": summary_metrics["transient_pair_count"],
        "valid_movement_estimates": movement_valid,
        "invalid_movement_estimates": len(rows) - movement_valid,
        "invalid_movement_reasons": dict(sorted(invalid_movement_reasons.items())),
        "movement_analysis_valid_persistent_pairs": summary_metrics[
            "movement_analysis_valid_persistent_pairs"
        ],
        "movement_analysis_invalid_persistent_pairs": summary_metrics[
            "movement_analysis_invalid_persistent_pairs"
        ],
        "valid_closest_approach_estimates": summary_metrics[
            "closest_approach_valid_persistent_pairs"
        ],
        "invalid_closest_approach_estimates": summary_metrics[
            "closest_approach_invalid_persistent_pairs"
        ],
        "timestamp_validation": {
            "common_frames": len(common_frames),
            "consistent_frames": valid_timestamp_frames,
            "mismatched_frames": mismatch_frames,
            "frames_with_missing_or_invalid_timestamps": invalid_timestamp_frames,
            "maximum_timestamp_spread_seconds": maximum_timestamp_delta,
            "tolerance_seconds": config.timestamp_tolerance_seconds,
            "frame_index_is_primary_alignment_key": True,
        },
        "missing_observation_statistics": {
            "pedestrian": _track_gap_statistics(
                pedestrians,
                include_class=False,
                max_gap=config.max_observation_gap_frames,
            ),
            "vehicle": _track_gap_statistics(
                vehicles,
                include_class=True,
                max_gap=config.max_observation_gap_frames,
            ),
            "pair": {
                "gap_count_between_pair_observations": pair_gap_count,
                "unobserved_frame_slots_between_pair_observations": pair_missing_slots,
                "gaps_exceeding_maximum": pair_gaps_exceeding_limit,
                "interpretation": (
                    "Unobserved slots within an observed identity span; not proof "
                    "that either object was present or detected in those frames."
                ),
            },
        },
        "thresholds": {
            "max_observation_gap_frames": config.max_observation_gap_frames,
            "min_observations_for_movement": config.min_observations_for_movement,
            "min_observations_for_closest_approach": (
                config.min_observations_for_closest_approach
            ),
            "nearby_pixel_distance_threshold": config.pixel_distance_threshold,
            "movement_delta_tolerance_pixels": (
                config.movement_delta_tolerance_pixels
            ),
        },
        "interpretation": (
            "Image-space interaction candidates only. No metric distance, physical "
            "velocity, TTC, collision, or risk level is inferred."
        ),
    }


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _summary_fields() -> tuple[str, ...]:
    return (
        "set_id",
        "video_id",
        "pedestrian_track_id",
        "vehicle_class",
        "vehicle_track_id",
        "observation_count",
        "first_frame_index",
        "last_frame_index",
        "minimum_separation_pixels",
        "median_separation_pixels",
        "movement_valid_edge_count",
        "movement_status",
        "movement_analysis_valid",
        "movement_invalid_reason",
        "closest_approach_type",
        "closest_approach_frame_index",
        "closest_approach_timestamp_seconds",
        "closest_approach_separation_pixels",
        "closest_approach_valid",
        "closest_approach_invalid_reason",
        "closest_approach_assumptions",
    )


def _write_visual_samples(
    config: InteractionConfig,
    pedestrians: list[Observation],
    vehicles: list[Observation],
    rows: list[dict[str, Any]],
) -> list[str]:
    if config.visual_sample_count == 0 or not rows:
        return []
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Visual samples were requested but OpenCV is unavailable."
        ) from exc
    if not config.video_path.is_file():
        raise FileNotFoundError(f"Visual sample video does not exist: {config.video_path}")

    frames = sorted({row["frame_index"] for row in rows})
    recording_ids = {
        (row["set_id"], row["video_id"])
        for row in rows
    }
    if len(recording_ids) != 1:
        raise ValueError(
            "Visual samples require exactly one set/video identity because the "
            "configuration provides a single video_path."
        )
    sample_count = min(config.visual_sample_count, len(frames))
    selected_indices = (
        [0]
        if sample_count == 1
        else [
            round(index * (len(frames) - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]
    )
    selected_frames = [frames[index] for index in selected_indices]
    pedestrians_by_frame: dict[tuple[str, str, int], list[Observation]] = defaultdict(list)
    vehicles_by_frame: dict[tuple[str, str, int], list[Observation]] = defaultdict(list)
    for item in pedestrians:
        pedestrians_by_frame[item.frame_identity].append(item)
    for item in vehicles:
        vehicles_by_frame[item.frame_identity].append(item)
    rows_by_frame: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[(row["set_id"], row["video_id"], row["frame_index"])].append(row)

    capture = cv2.VideoCapture(str(config.video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open visual sample video: {config.video_path}")
        sample_dir = config.output_dir / "visual_samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        created: list[str] = []
        for frame_index in selected_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok or image is None:
                raise OSError(f"Could not decode video frame {frame_index}.")
            identity = next(
                (
                    key
                    for key in rows_by_frame
                    if key[2] == frame_index
                    and _within_filter(key[0], key[1], config)
                ),
                None,
            )
            if identity is None:
                continue
            for pedestrian in pedestrians_by_frame[identity]:
                x1, y1, x2, y2 = map(round, pedestrian.bbox_xyxy)
                cv2.rectangle(image, (x1, y1), (x2, y2), (60, 210, 60), 2)
                cv2.putText(
                    image,
                    f"ped {pedestrian.track_id}",
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (60, 210, 60),
                    2,
                )
                footpoint = pedestrian_footpoint(pedestrian.bbox_xyxy)
                cv2.circle(image, tuple(map(round, footpoint)), 5, (0, 255, 255), -1)
            for vehicle in vehicles_by_frame[identity]:
                x1, y1, x2, y2 = map(round, vehicle.bbox_xyxy)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 140, 0), 2)
                cv2.putText(
                    image,
                    f"{vehicle.class_name} {vehicle.track_id}",
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 140, 0),
                    2,
                )
            nearest_pair = min(
                rows_by_frame[identity],
                key=lambda row: (
                    row["distance_pixels"],
                    row["pedestrian_track_id"],
                    row["vehicle_class"],
                    row["vehicle_track_id"],
                ),
            )
            for row in (nearest_pair,):
                start = (
                    round(row["pedestrian_footpoint_x_px"]),
                    round(row["pedestrian_footpoint_y_px"]),
                )
                end = (
                    round(row["nearest_vehicle_point_x_px"]),
                    round(row["nearest_vehicle_point_y_px"]),
                )
                cv2.line(image, start, end, (0, 0, 255), 2)
                label = (
                    f"p{row['pedestrian_track_id']} / "
                    f"{row['vehicle_class']} {row['vehicle_track_id']} "
                    f"{row['distance_pixels']:.1f}px "
                    f"{row['movement_status']}"
                )
                cv2.putText(
                    image,
                    label,
                    (start[0] + 5, max(18, start[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
            output_path = sample_dir / f"interaction_frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(output_path), image):
                raise OSError(f"Could not write visual sample: {output_path}")
            created.append(output_path.relative_to(config.output_dir).as_posix())
    finally:
        capture.release()
    return created


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _build_report(
    metrics: dict[str, Any],
    summaries: list[dict[str, Any]],
    visual_paths: list[str],
) -> str:
    timestamp = metrics["timestamp_validation"]
    missing = metrics["missing_observation_statistics"]
    lines = [
        "# Pedestrian–Vehicle Interaction Analysis",
        "",
        "## Scope and result",
        "",
        "This experiment produces same-frame pedestrian–vehicle interaction "
        "candidates and image-space movement descriptors. It is not a validated "
        "collision predictor and does not assign risk levels.",
        "",
        f"- Recording filter: sets {', '.join(metrics['experiment']['set_ids']) or 'all'}; "
        f"videos {', '.join(metrics['experiment']['video_ids']) or 'all'}.",
        f"- Pedestrian observations: {metrics['pedestrian_count']:,} across "
        f"{metrics['unique_pedestrian_tracks']:,} tracks.",
        f"- Vehicle trajectory observations: {metrics['vehicle_track_observation_count']:,} across "
        f"{metrics['unique_vehicle_tracks']:,} class-aware tracks.",
        f"- Candidate pair observations: {metrics['candidate_pair_observations']:,}.",
        f"- Unique pedestrian–vehicle track pairs: {metrics['candidate_unique_track_pairs']:,}; "
        f"persistent pairs (at least two observations): {metrics['persistent_pair_count']:,}; "
        f"single-observation pairs: {metrics['transient_pair_count']:,}.",
        f"- Valid/invalid per-observation movement estimates: "
        f"{metrics['valid_movement_estimates']:,}/{metrics['invalid_movement_estimates']:,}.",
        f"- Valid/invalid persistent-pair movement summaries: "
        f"{metrics['movement_analysis_valid_persistent_pairs']:,}/"
        f"{metrics['movement_analysis_invalid_persistent_pairs']:,}.",
        f"- Valid/invalid observed closest-approach summaries: "
        f"{metrics['valid_closest_approach_estimates']:,}/"
        f"{metrics['invalid_closest_approach_estimates']:,}.",
        "",
        "## Method",
        "",
        "Observations are joined only on the exact `(set_id, video_id, frame_index)` "
        "key. Pedestrian and vehicle tracker IDs remain separate and are preserved; "
        "vehicle identities are class-aware. Every pedestrian and vehicle present "
        "in a shared frame forms one candidate pair, with no interpolation or "
        "distance-based pair pruning.",
        "",
        "The pedestrian footpoint is the bounding-box bottom-center "
        "`((x1+x2)/2, y2)`. The vehicle reference point for displacement is the "
        "bounding-box center. Per-frame separation is the Euclidean distance from "
        "the footpoint to the nearest point on the vehicle rectangle, obtained by "
        "clamping each footpoint coordinate to the box bounds: "
        "`q=(clamp(px,x1,x2), clamp(py,y1,y2))`, "
        "`d_px=sqrt((px-qx)^2+(py-qy)^2)`. The vehicle center is an image-space "
        "reference only, not a physical position.",
        "",
        "For valid successive pair observations, pedestrian displacement is the "
        "change in footpoint; vehicle displacement is the change in box center; "
        "relative displacement is `delta_vehicle_center - delta_pedestrian_footpoint`; "
        "its image-plane direction is `atan2(relative_dy, relative_dx)`. "
        "Approaching/separating is determined by `delta_d = d_current - d_previous`: "
        "negative beyond the configured tolerance means approaching, positive means "
        "separating, otherwise indeterminate. No movement is computed across "
        "nonpositive timestamps, timestamp misalignment, or frame gaps above the "
        "configured limit. Positions are never interpolated.",
        "",
        "The closest-approach output is the minimum **observed** separation inside "
        "the longest consecutive valid pair segment. It is not a projected future "
        "closest approach and is not TTC.",
        "",
        "## Alignment and data quality",
        "",
        f"- Shared frames: {timestamp['common_frames']:,}; timestamp-consistent: "
        f"{timestamp['consistent_frames']:,}; mismatched: {timestamp['mismatched_frames']:,}; "
        f"missing/invalid timestamps: {timestamp['frames_with_missing_or_invalid_timestamps']:,}.",
        f"- Maximum timestamp spread: "
        f"{_format_metric(timestamp['maximum_timestamp_spread_seconds'])} seconds "
        f"(tolerance {timestamp['tolerance_seconds']}).",
        f"- Vehicle detection-to-trajectory matched observations: "
        f"{metrics['inputs']['vehicle']['matched_detection_trajectory_observations']:,}; "
        f"tracked detections without trajectory rows: "
        f"{metrics['inputs']['vehicle']['tracked_detections_without_trajectory']:,}; "
        f"trajectory rows without detection rows: "
        f"{metrics['inputs']['vehicle']['trajectory_observations_without_detection']:,}.",
        f"- Missing frame slots within pedestrian track spans: "
        f"{missing['pedestrian']['unobserved_frame_slots_within_track_spans']:,}; "
        f"within vehicle track spans: "
        f"{missing['vehicle']['unobserved_frame_slots_within_track_spans']:,}.",
        f"- Pair-observation gaps: {missing['pair']['gap_count_between_pair_observations']:,}, "
        f"with {missing['pair']['unobserved_frame_slots_between_pair_observations']:,} "
        "unobserved frame slots. These are gaps in sampled outputs, not proof of "
        "object presence or detector misses.",
        "",
        "## Interpretation and limitations",
        "",
        f"The configured `nearby` indicator is a distance threshold of "
        f"{metrics['thresholds']['nearby_pixel_distance_threshold']} pixels. It is "
        "an experimental image-space filter, not a physical clearance or risk "
        "threshold. No Low/Medium/High categories, collision labels, meters, "
        "real-world speeds, or validated TTC values are produced.",
        "",
        "PIE is recorded from a moving ego-vehicle camera. Ego motion, perspective, "
        "parallax, changing scale, occlusion, detector jitter, and tracker ID "
        "switches can change apparent image-space distance and direction. Vehicle "
        "annotations are not exhaustive; unmatched detections are not interpreted "
        "as false positives. A 2D footpoint/box overlap is not proof of physical "
        "contact.",
        "",
        "The source trackers provide estimated IDs, not guaranteed persistent "
        "identities. Missing observations are counted within observed track spans "
        "only; no absent position is fabricated.",
        "",
        "Each sample shows all available pedestrian and vehicle boxes/track IDs "
        "and pedestrian footpoints, with a distance/status line for only the "
        "nearest image-space candidate pair in that frame. This presentation "
        "does not filter the saved interaction data.",
        "",
        "## Visual samples",
        "",
    ]
    if visual_paths:
        lines.extend(f"- [{path}]({path})" for path in visual_paths)
    else:
        lines.append("No visual samples were generated (no candidate pairs or disabled).")
    lines.extend(["", "## Persistent pair overview", ""])
    if summaries:
        lines.extend(
            [
                "| Set/video | Pedestrian track | Vehicle class/track | Observations | Min separation (px) | Median (px) | Movement | Closest observed frame |",
                "|---|---|---|---:|---:|---:|---|---:|",
            ]
        )
        for summary in summaries[:20]:
            lines.append(
                f"| {summary['set_id']}/{summary['video_id']} | "
                f"{summary['pedestrian_track_id']} | "
                f"{summary['vehicle_class']}/{summary['vehicle_track_id']} | "
                f"{summary['observation_count']} | "
                f"{summary['minimum_separation_pixels']:.2f} | "
                f"{summary['median_separation_pixels']:.2f} | "
                f"{summary['movement_status']} | "
                f"{_format_metric(summary['closest_approach_frame_index'])} |"
            )
        if len(summaries) > 20:
            lines.append(f"\nShowing 20 of {len(summaries)} persistent pairs.")
    else:
        lines.append("No persistent pedestrian–vehicle track pairs were observed.")
    lines.append("")
    return "\n".join(lines)


def run_analysis(config_path: Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    for key in (
        "pedestrian_trajectory_path",
        "vehicle_detection_path",
        "vehicle_trajectory_path",
    ):
        if not getattr(config, key).is_file():
            raise FileNotFoundError(
                f"Configured {key} does not exist: {getattr(config, key)}"
            )
    pedestrians, pedestrian_inputs = _load_pedestrians(config)
    vehicles, vehicle_inputs = _load_vehicles(config)
    rows = generate_interaction_rows(pedestrians, vehicles, config)
    summaries, summary_metrics = summarize_interactions(rows, config)
    metrics = _metrics(
        pedestrians,
        vehicles,
        rows,
        summary_metrics,
        pedestrian_inputs,
        vehicle_inputs,
        config,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        config.output_dir / "interaction_observations.csv",
        _OBSERVATION_FIELDS,
        rows,
    )
    _write_csv(
        config.output_dir / "interaction_summary.csv",
        _summary_fields(),
        summaries,
    )
    (config.output_dir / "interaction_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_snapshot = {
        key: (str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value)
        for key, value in config.__dict__.items()
    }
    (config.output_dir / "interaction_run_config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    visual_paths = _write_visual_samples(config, pedestrians, vehicles, rows)
    report = _build_report(metrics, summaries, visual_paths)
    (config.output_dir / "INTERACTION_ANALYSIS_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    return {
        "metrics": metrics,
        "summaries": summaries,
        "visual_samples": visual_paths,
        "report_path": config.output_dir / "INTERACTION_ANALYSIS_REPORT.md",
    }
