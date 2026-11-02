from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    path: str | None = None
    video_id: str | None = None
    frame_index: int | None = None
    track_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoMetadata:
    width: int | None
    height: int | None
    fps: float | None
    frame_count: int | None
    duration_seconds: float | None
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpatialSummary:
    task_size: int | None
    image_width: int | None
    image_height: int | None
    visible_frames: set[int]
    all_box_frames: set[int]
    track_ids: dict[str, set[str]]
    track_counts: dict[str, int]
    box_counts: dict[str, int]
    visible_box_counts: dict[str, int]
    attribute_counts: dict[str, dict[str, int]]


def increment_nested(
    counts: dict[str, dict[str, int]], key: str, value: str
) -> None:
    values = counts.setdefault(key, {})
    values[value] = values.get(value, 0) + 1
