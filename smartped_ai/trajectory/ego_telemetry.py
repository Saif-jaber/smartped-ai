from __future__ import annotations

import math
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EgoFrame:
    frame_index: int
    gps_speed_kmh: float | None
    obd_speed_kmh: float | None
    heading_deg: float | None
    gyro_x: float | None
    gyro_y: float | None
    gyro_z: float | None
    pitch_deg: float | None
    roll_deg: float | None
    yaw_deg: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "gps_speed_kmh": self.gps_speed_kmh,
            "obd_speed_kmh": self.obd_speed_kmh,
            "heading_deg": self.heading_deg,
            "gyro_x": self.gyro_x,
            "gyro_y": self.gyro_y,
            "gyro_z": self.gyro_z,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "yaw_deg": self.yaw_deg,
        }


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def read_ego_telemetry(
    path: Path,
    *,
    max_frame: int | None = None,
) -> list[EgoFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Ego telemetry XML does not exist: {path}")
    frames: list[EgoFrame] = []
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag != "frame":
            continue
        frame_index = int(_float_or_none(element.get("id")) or len(frames))
        if max_frame is not None and frame_index >= max_frame:
            element.clear()
            continue
        frames.append(
            EgoFrame(
                frame_index=frame_index,
                gps_speed_kmh=_float_or_none(element.get("GPS_speed")),
                obd_speed_kmh=_float_or_none(element.get("OBD_speed")),
                heading_deg=_float_or_none(element.get("heading_angle")),
                gyro_x=_float_or_none(element.get("gyroX")),
                gyro_y=_float_or_none(element.get("gyroY")),
                gyro_z=_float_or_none(element.get("gyroZ")),
                pitch_deg=_float_or_none(element.get("pitch")),
                roll_deg=_float_or_none(element.get("roll")),
                yaw_deg=_float_or_none(element.get("yaw")),
            )
        )
        element.clear()
    frames.sort(key=lambda frame: frame.frame_index)
    return frames


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_ego_telemetry(
    path: Path,
    *,
    max_frame: int | None = None,
) -> dict[str, Any]:
    frames = read_ego_telemetry(path, max_frame=max_frame)
    gps_speeds = [
        frame.gps_speed_kmh
        for frame in frames
        if frame.gps_speed_kmh is not None
    ]
    obd_speeds = [
        frame.obd_speed_kmh
        for frame in frames
        if frame.obd_speed_kmh is not None
    ]
    heading_changes = [
        abs(current.heading_deg - previous.heading_deg)
        for previous, current in zip(frames, frames[1:])
        if previous.heading_deg is not None and current.heading_deg is not None
    ]
    gps_deltas = [
        abs(current.gps_speed_kmh - previous.gps_speed_kmh)
        for previous, current in zip(frames, frames[1:])
        if previous.gps_speed_kmh is not None
        and current.gps_speed_kmh is not None
    ]
    return {
        "source": path.as_posix(),
        "frame_count": len(frames),
        "frame_range": (
            {
                "first": frames[0].frame_index,
                "last": frames[-1].frame_index,
            }
            if frames
            else None
        ),
        "gps_speed_kmh": _stats(gps_speeds),
        "obd_speed_kmh": _stats(obd_speeds),
        "mean_absolute_heading_change_deg_per_frame": (
            statistics.fmean(heading_changes) if heading_changes else None
        ),
        "mean_absolute_gps_speed_change_kmh_per_frame": (
            statistics.fmean(gps_deltas) if gps_deltas else None
        ),
        "usable_for_pixel_compensation": False,
        "pixel_compensation_reason": (
            "The telemetry XML contains ego speed and inertial measurements but no "
            "camera intrinsics, extrinsics, mounting geometry, or road-plane "
            "homography. It cannot be converted directly into pixel displacement."
        ),
    }
