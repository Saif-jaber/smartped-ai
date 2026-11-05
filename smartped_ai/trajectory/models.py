from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

MOVEMENT_UNKNOWN = "unknown"
MOVEMENT_WALKING = "walking"
MOVEMENT_STANDING = "standing"

CSV_FIELDS = (
    "set_id",
    "video_id",
    "track_id",
    "observation_index",
    "frame_index",
    "timestamp_seconds",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "center_x",
    "center_y",
    "bbox_width",
    "bbox_height",
    "previous_frame_index",
    "frame_gap",
    "time_gap_seconds",
    "displacement_x_px",
    "displacement_y_px",
    "displacement_px",
    "velocity_x_px_s",
    "velocity_y_px_s",
    "speed_px_s",
    "raw_displacement_x_px",
    "raw_displacement_y_px",
    "raw_displacement_px",
    "raw_speed_px_s",
    "camera_motion_applied",
    "camera_motion_method",
    "camera_motion_dx_px",
    "camera_motion_dy_px",
    "camera_motion_local_samples",
    "camera_motion_fallback_steps",
    "camera_motion_invalid_reason",
    "direction_deg",
    "direction_label",
    "raw_movement_state",
    "smoothed_movement_state",
    "velocity_valid",
    "gap_exceeded",
    "velocity_invalid_reason",
    "confidence",
    "occluded",
    "source",
)


@dataclass(frozen=True)
class TrajectoryConfig:
    fps: float = 30.0
    max_gap_frames: int = 15
    standing_speed_threshold_px_s: float = 10.0
    smoothing_window_observations: int = 9
    smoothing_min_observations: int = 3

    def validate(self) -> None:
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be a finite value greater than zero.")
        if self.max_gap_frames < 1:
            raise ValueError("max_gap_frames must be at least 1.")
        if (
            not math.isfinite(self.standing_speed_threshold_px_s)
            or self.standing_speed_threshold_px_s < 0
        ):
            raise ValueError(
                "standing_speed_threshold_px_s must be finite and non-negative."
            )
        if self.smoothing_window_observations < 1:
            raise ValueError("smoothing_window_observations must be at least 1.")
        if self.smoothing_min_observations < 1:
            raise ValueError("smoothing_min_observations must be at least 1.")
        if self.smoothing_min_observations > self.smoothing_window_observations:
            raise ValueError(
                "smoothing_min_observations cannot exceed "
                "smoothing_window_observations."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackObservation:
    track_id: str
    frame_index: int
    bbox_xyxy: tuple[float, float, float, float]
    timestamp_seconds: float | None = None
    set_id: str | None = None
    video_id: str | None = None
    confidence: float | None = None
    occluded: bool | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("Track observation has no track_id.")
        if self.frame_index < 0:
            raise ValueError("Track observation frame_index cannot be negative.")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("Track observation bounding box must be finite.")
        if x1 >= x2 or y1 >= y2:
            raise ValueError(
                "Track observation bounding box must have positive width and height."
            )
        if self.timestamp_seconds is not None and (
            not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0
        ):
            raise ValueError(
                "Track observation timestamp must be finite and non-negative."
            )

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.set_id or "", self.video_id or "", self.track_id


@dataclass(frozen=True)
class TrajectoryFeature:
    set_id: str | None
    video_id: str | None
    track_id: str
    observation_index: int
    frame_index: int
    timestamp_seconds: float | None
    bbox_xyxy: tuple[float, float, float, float]
    center_x: float
    center_y: float
    bbox_width: float
    bbox_height: float
    previous_frame_index: int | None
    frame_gap: int | None
    time_gap_seconds: float | None
    displacement_x_px: float | None
    displacement_y_px: float | None
    displacement_px: float | None
    velocity_x_px_s: float | None
    velocity_y_px_s: float | None
    speed_px_s: float | None
    direction_deg: float | None
    direction_label: str | None
    raw_movement_state: str | None
    smoothed_movement_state: str
    velocity_valid: bool
    gap_exceeded: bool
    velocity_invalid_reason: str | None
    confidence: float | None
    occluded: bool | None
    source: str
    raw_displacement_x_px: float | None = None
    raw_displacement_y_px: float | None = None
    raw_displacement_px: float | None = None
    raw_speed_px_s: float | None = None
    camera_motion_applied: bool = False
    camera_motion_method: str | None = None
    camera_motion_dx_px: float | None = None
    camera_motion_dy_px: float | None = None
    camera_motion_local_samples: int = 0
    camera_motion_fallback_steps: int = 0
    camera_motion_invalid_reason: str | None = None

    def to_csv_row(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return {
            "set_id": self.set_id or "",
            "video_id": self.video_id or "",
            "track_id": self.track_id,
            "observation_index": self.observation_index,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
            "previous_frame_index": self.previous_frame_index,
            "frame_gap": self.frame_gap,
            "time_gap_seconds": self.time_gap_seconds,
            "displacement_x_px": self.displacement_x_px,
            "displacement_y_px": self.displacement_y_px,
            "displacement_px": self.displacement_px,
            "velocity_x_px_s": self.velocity_x_px_s,
            "velocity_y_px_s": self.velocity_y_px_s,
            "speed_px_s": self.speed_px_s,
            "raw_displacement_x_px": self.raw_displacement_x_px,
            "raw_displacement_y_px": self.raw_displacement_y_px,
            "raw_displacement_px": self.raw_displacement_px,
            "raw_speed_px_s": self.raw_speed_px_s,
            "camera_motion_applied": (
                "true" if self.camera_motion_applied else "false"
            ),
            "camera_motion_method": self.camera_motion_method or "",
            "camera_motion_dx_px": self.camera_motion_dx_px,
            "camera_motion_dy_px": self.camera_motion_dy_px,
            "camera_motion_local_samples": self.camera_motion_local_samples,
            "camera_motion_fallback_steps": self.camera_motion_fallback_steps,
            "camera_motion_invalid_reason": (
                self.camera_motion_invalid_reason or ""
            ),
            "direction_deg": self.direction_deg,
            "direction_label": self.direction_label or "",
            "raw_movement_state": self.raw_movement_state or "",
            "smoothed_movement_state": self.smoothed_movement_state,
            "velocity_valid": "true" if self.velocity_valid else "false",
            "gap_exceeded": "true" if self.gap_exceeded else "false",
            "velocity_invalid_reason": self.velocity_invalid_reason or "",
            "confidence": self.confidence,
            "occluded": (
                "true" if self.occluded else "false" if self.occluded is not None else ""
            ),
            "source": self.source,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "set_id": self.set_id,
            "video_id": self.video_id,
            "track_id": self.track_id,
            "observation_index": self.observation_index,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "bbox_xyxy": list(self.bbox_xyxy),
            "center_xy": [self.center_x, self.center_y],
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
            "previous_frame_index": self.previous_frame_index,
            "frame_gap": self.frame_gap,
            "time_gap_seconds": self.time_gap_seconds,
            "displacement_px": (
                {
                    "x": self.displacement_x_px,
                    "y": self.displacement_y_px,
                    "magnitude": self.displacement_px,
                }
                if self.displacement_px is not None
                else None
            ),
            "velocity_px_s": (
                {
                    "x": self.velocity_x_px_s,
                    "y": self.velocity_y_px_s,
                    "magnitude": self.speed_px_s,
                }
                if self.speed_px_s is not None
                else None
            ),
            "raw_displacement_px": (
                {
                    "x": self.raw_displacement_x_px,
                    "y": self.raw_displacement_y_px,
                    "magnitude": self.raw_displacement_px,
                }
                if self.raw_displacement_px is not None
                else None
            ),
            "raw_velocity_px_s": (
                {
                    "magnitude": self.raw_speed_px_s,
                }
                if self.raw_speed_px_s is not None
                else None
            ),
            "camera_motion": (
                {
                    "applied": self.camera_motion_applied,
                    "method": self.camera_motion_method,
                    "dx_px": self.camera_motion_dx_px,
                    "dy_px": self.camera_motion_dy_px,
                    "local_sample_count": self.camera_motion_local_samples,
                    "fallback_step_count": self.camera_motion_fallback_steps,
                    "invalid_reason": self.camera_motion_invalid_reason,
                }
                if self.camera_motion_applied
                or self.camera_motion_method is not None
                or self.camera_motion_invalid_reason is not None
                else None
            ),
            "direction_deg": self.direction_deg,
            "direction_label": self.direction_label,
            "raw_movement_state": self.raw_movement_state,
            "smoothed_movement_state": self.smoothed_movement_state,
            "velocity_valid": self.velocity_valid,
            "gap_exceeded": self.gap_exceeded,
            "velocity_invalid_reason": self.velocity_invalid_reason,
            "confidence": self.confidence,
            "occluded": self.occluded,
            "source": self.source,
        }


@dataclass(frozen=True)
class TrackSummary:
    set_id: str | None
    video_id: str | None
    track_id: str
    observation_count: int
    expected_observation_count: int
    missing_observation_count: int
    first_frame_index: int
    last_frame_index: int
    first_timestamp_seconds: float | None
    last_timestamp_seconds: float | None
    duration_seconds: float | None
    valid_velocity_count: int
    long_gap_count: int
    walking_count: int
    standing_count: int
    unknown_count: int
    mean_speed_px_s: float | None
    max_speed_px_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
