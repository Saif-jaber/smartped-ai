from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from typing import TYPE_CHECKING, Iterable

from .models import (
    MOVEMENT_STANDING,
    MOVEMENT_UNKNOWN,
    MOVEMENT_WALKING,
    TrackObservation,
    TrackSummary,
    TrajectoryConfig,
    TrajectoryFeature,
)

if TYPE_CHECKING:
    from .camera_motion import CameraMotionSequence


def _effective_timestamp(
    observation: TrackObservation, fps: float
) -> float | None:
    if observation.timestamp_seconds is not None:
        return observation.timestamp_seconds
    if math.isfinite(fps) and fps > 0:
        return observation.frame_index / fps
    return None


def _direction(displacement_x: float, displacement_y: float) -> tuple[float | None, str]:
    if displacement_x == 0 and displacement_y == 0:
        return None, "none"
    degrees = math.degrees(math.atan2(displacement_y, displacement_x)) % 360.0
    if degrees < 22.5 or degrees >= 337.5:
        label = "right"
    elif degrees < 67.5:
        label = "down-right"
    elif degrees < 112.5:
        label = "down"
    elif degrees < 157.5:
        label = "down-left"
    elif degrees < 202.5:
        label = "left"
    elif degrees < 247.5:
        label = "up-left"
    elif degrees < 292.5:
        label = "up"
    else:
        label = "up-right"
    return degrees, label


def _smoothed_state(
    history: deque[str],
    raw_state: str,
    minimum_observations: int,
) -> str:
    if len(history) < minimum_observations:
        return raw_state
    counts = Counter(history)
    highest = max(counts.values())
    leaders = [state for state, count in counts.items() if count == highest]
    if raw_state in leaders:
        return raw_state
    return sorted(leaders)[0]


def _feature(
    observation: TrackObservation,
    *,
    observation_index: int,
    timestamp_seconds: float | None,
    previous_frame_index: int | None,
    frame_gap: int | None,
    time_gap_seconds: float | None,
    displacement_x: float | None,
    displacement_y: float | None,
    velocity_x: float | None,
    velocity_y: float | None,
    speed: float | None,
    raw_displacement_x: float | None,
    raw_displacement_y: float | None,
    raw_speed: float | None,
    camera_motion_applied: bool,
    camera_motion_method: str | None,
    camera_motion_dx: float | None,
    camera_motion_dy: float | None,
    camera_motion_local_samples: int,
    camera_motion_fallback_steps: int,
    camera_motion_invalid_reason: str | None,
    direction_deg: float | None,
    direction_label: str | None,
    raw_state: str | None,
    smoothed_state: str,
    velocity_valid: bool,
    gap_exceeded: bool,
    invalid_reason: str | None,
) -> TrajectoryFeature:
    center_x, center_y = observation.center_xy
    x1, y1, x2, y2 = observation.bbox_xyxy
    return TrajectoryFeature(
        set_id=observation.set_id,
        video_id=observation.video_id,
        track_id=observation.track_id,
        observation_index=observation_index,
        frame_index=observation.frame_index,
        timestamp_seconds=timestamp_seconds,
        bbox_xyxy=observation.bbox_xyxy,
        center_x=center_x,
        center_y=center_y,
        bbox_width=x2 - x1,
        bbox_height=y2 - y1,
        previous_frame_index=previous_frame_index,
        frame_gap=frame_gap,
        time_gap_seconds=time_gap_seconds,
        displacement_x_px=displacement_x,
        displacement_y_px=displacement_y,
        displacement_px=(
            math.hypot(displacement_x, displacement_y)
            if displacement_x is not None and displacement_y is not None
            else None
        ),
        velocity_x_px_s=velocity_x,
        velocity_y_px_s=velocity_y,
        speed_px_s=speed,
        direction_deg=direction_deg,
        direction_label=direction_label,
        raw_movement_state=raw_state,
        smoothed_movement_state=smoothed_state,
        velocity_valid=velocity_valid,
        gap_exceeded=gap_exceeded,
        velocity_invalid_reason=invalid_reason,
        confidence=observation.confidence,
        occluded=observation.occluded,
        source=observation.source,
        raw_displacement_x_px=raw_displacement_x,
        raw_displacement_y_px=raw_displacement_y,
        raw_displacement_px=(
            math.hypot(raw_displacement_x, raw_displacement_y)
            if raw_displacement_x is not None and raw_displacement_y is not None
            else None
        ),
        raw_speed_px_s=raw_speed,
        camera_motion_applied=camera_motion_applied,
        camera_motion_method=camera_motion_method,
        camera_motion_dx_px=camera_motion_dx,
        camera_motion_dy_px=camera_motion_dy,
        camera_motion_local_samples=camera_motion_local_samples,
        camera_motion_fallback_steps=camera_motion_fallback_steps,
        camera_motion_invalid_reason=camera_motion_invalid_reason,
    )


def _process_track(
    observations: list[TrackObservation],
    config: TrajectoryConfig,
    camera_motion: CameraMotionSequence | None = None,
) -> list[TrajectoryFeature]:
    ordered = sorted(
        observations,
        key=lambda observation: (
            observation.frame_index,
            observation.timestamp_seconds
            if observation.timestamp_seconds is not None
            else -1.0,
        ),
    )
    history: deque[str] = deque(
        maxlen=config.smoothing_window_observations
    )
    features: list[TrajectoryFeature] = []
    previous: TrackObservation | None = None
    previous_timestamp: float | None = None
    previous_center: tuple[float, float] | None = None

    for observation_index, observation in enumerate(ordered):
        timestamp = _effective_timestamp(observation, config.fps)
        center = observation.center_xy

        if previous is None or previous_center is None:
            features.append(
                _feature(
                    observation,
                    observation_index=observation_index,
                    timestamp_seconds=timestamp,
                    previous_frame_index=None,
                    frame_gap=None,
                    time_gap_seconds=None,
                    displacement_x=None,
                    displacement_y=None,
                    velocity_x=None,
                    velocity_y=None,
                    speed=None,
                    raw_displacement_x=None,
                    raw_displacement_y=None,
                    raw_speed=None,
                    camera_motion_applied=False,
                    camera_motion_method=None,
                    camera_motion_dx=None,
                    camera_motion_dy=None,
                    camera_motion_local_samples=0,
                    camera_motion_fallback_steps=0,
                    camera_motion_invalid_reason=None,
                    direction_deg=None,
                    direction_label=None,
                    raw_state=None,
                    smoothed_state=MOVEMENT_UNKNOWN,
                    velocity_valid=False,
                    gap_exceeded=False,
                    invalid_reason="first_observation",
                )
            )
            previous = observation
            previous_timestamp = timestamp
            previous_center = center
            continue

        frame_gap = observation.frame_index - previous.frame_index
        raw_displacement_x = center[0] - previous_center[0]
        raw_displacement_y = center[1] - previous_center[1]
        time_gap = (
            timestamp - previous_timestamp
            if timestamp is not None and previous_timestamp is not None
            else None
        )
        gap_exceeded = frame_gap > config.max_gap_frames

        displacement_x = raw_displacement_x
        displacement_y = raw_displacement_y
        camera_motion_applied = False
        camera_motion_method: str | None = None
        camera_motion_dx: float | None = None
        camera_motion_dy: float | None = None
        camera_motion_local_samples = 0
        camera_motion_fallback_steps = 0
        camera_motion_invalid_reason: str | None = None

        if camera_motion is not None:
            if frame_gap <= 0:
                camera_motion_invalid_reason = "nonpositive_frame_gap"
            elif gap_exceeded:
                camera_motion_invalid_reason = "frame_gap_exceeded"
            else:
                estimate = camera_motion.displacement_at(
                    previous_center,
                    previous.frame_index,
                    observation.frame_index,
                )
                camera_motion_local_samples = estimate.local_sample_count
                camera_motion_fallback_steps = estimate.fallback_step_count
                if estimate.available:
                    camera_motion_applied = True
                    camera_motion_method = estimate.method
                    camera_motion_dx = estimate.dx_px
                    camera_motion_dy = estimate.dy_px
                    displacement_x = raw_displacement_x - estimate.dx_px
                    displacement_y = raw_displacement_y - estimate.dy_px
                else:
                    camera_motion_invalid_reason = (
                        estimate.reason or "camera_motion_unavailable"
                    )

        invalid_reason: str | None = None
        if frame_gap <= 0:
            invalid_reason = "nonpositive_frame_gap"
        elif gap_exceeded:
            invalid_reason = "frame_gap_exceeded"
        elif time_gap is None:
            invalid_reason = "missing_timestamp"
        elif time_gap <= 0:
            invalid_reason = "nonpositive_time_gap"

        velocity_x: float | None = None
        velocity_y: float | None = None
        speed: float | None = None
        raw_speed: float | None = None
        raw_state: str | None = None
        direction_deg: float | None = None
        direction_label: str | None = None

        if invalid_reason is None:
            assert time_gap is not None
            raw_speed = (
                math.hypot(raw_displacement_x, raw_displacement_y) / time_gap
            )
            velocity_x = displacement_x / time_gap
            velocity_y = displacement_y / time_gap
            speed = math.hypot(velocity_x, velocity_y)
            raw_state = (
                MOVEMENT_WALKING
                if speed >= config.standing_speed_threshold_px_s
                else MOVEMENT_STANDING
            )
            history.append(raw_state)
            smoothed_state = _smoothed_state(
                history, raw_state, config.smoothing_min_observations
            )
        else:
            history.clear()
            smoothed_state = MOVEMENT_UNKNOWN

        if not gap_exceeded and frame_gap > 0 and (
            displacement_x != 0 or displacement_y != 0
        ):
            direction_deg, direction_label = _direction(
                displacement_x, displacement_y
            )

        features.append(
            _feature(
                observation,
                observation_index=observation_index,
                timestamp_seconds=timestamp,
                previous_frame_index=previous.frame_index,
                frame_gap=frame_gap,
                time_gap_seconds=time_gap,
                displacement_x=displacement_x,
                displacement_y=displacement_y,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                speed=speed,
                raw_displacement_x=raw_displacement_x,
                raw_displacement_y=raw_displacement_y,
                raw_speed=raw_speed,
                camera_motion_applied=camera_motion_applied,
                camera_motion_method=camera_motion_method,
                camera_motion_dx=camera_motion_dx,
                camera_motion_dy=camera_motion_dy,
                camera_motion_local_samples=camera_motion_local_samples,
                camera_motion_fallback_steps=camera_motion_fallback_steps,
                camera_motion_invalid_reason=camera_motion_invalid_reason,
                direction_deg=direction_deg,
                direction_label=direction_label,
                raw_state=raw_state,
                smoothed_state=smoothed_state,
                velocity_valid=invalid_reason is None,
                gap_exceeded=gap_exceeded,
                invalid_reason=invalid_reason,
            )
        )
        previous = observation
        previous_timestamp = timestamp
        previous_center = center

    return features


def extract_track_features(
    observations: Iterable[TrackObservation],
    config: TrajectoryConfig,
    *,
    camera_motion: CameraMotionSequence | None = None,
) -> list[TrajectoryFeature]:
    config.validate()
    grouped: dict[tuple[str, str, str], list[TrackObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.identity].append(observation)

    features: list[TrajectoryFeature] = []
    for identity in sorted(grouped):
        features.extend(
            _process_track(grouped[identity], config, camera_motion)
        )
    return features


def summarize_tracks(features: Iterable[TrajectoryFeature]) -> list[TrackSummary]:
    grouped: dict[tuple[str, str, str], list[TrajectoryFeature]] = defaultdict(list)
    for feature in features:
        grouped[(feature.set_id or "", feature.video_id or "", feature.track_id)].append(
            feature
        )

    summaries: list[TrackSummary] = []
    for identity in sorted(grouped):
        track_features = sorted(
            grouped[identity], key=lambda feature: feature.frame_index
        )
        speeds = [
            feature.speed_px_s
            for feature in track_features
            if feature.velocity_valid and feature.speed_px_s is not None
        ]
        state_counts = Counter(
            feature.smoothed_movement_state for feature in track_features
        )
        first_timestamp = track_features[0].timestamp_seconds
        last_timestamp = track_features[-1].timestamp_seconds
        expected_observations = (
            track_features[-1].frame_index - track_features[0].frame_index + 1
        )
        duration = (
            last_timestamp - first_timestamp
            if first_timestamp is not None
            and last_timestamp is not None
            and last_timestamp >= first_timestamp
            else None
        )
        summaries.append(
            TrackSummary(
                set_id=track_features[0].set_id,
                video_id=track_features[0].video_id,
                track_id=track_features[0].track_id,
                observation_count=len(track_features),
                expected_observation_count=expected_observations,
                missing_observation_count=max(
                    0, expected_observations - len(track_features)
                ),
                first_frame_index=track_features[0].frame_index,
                last_frame_index=track_features[-1].frame_index,
                first_timestamp_seconds=first_timestamp,
                last_timestamp_seconds=last_timestamp,
                duration_seconds=duration,
                valid_velocity_count=len(speeds),
                long_gap_count=sum(
                    feature.gap_exceeded for feature in track_features
                ),
                walking_count=state_counts[MOVEMENT_WALKING],
                standing_count=state_counts[MOVEMENT_STANDING],
                unknown_count=state_counts[MOVEMENT_UNKNOWN],
                mean_speed_px_s=(sum(speeds) / len(speeds) if speeds else None),
                max_speed_px_s=(max(speeds) if speeds else None),
            )
        )
    return summaries


def coverage_diagnostics(
    features: Iterable[TrajectoryFeature],
) -> dict[str, object]:
    grouped: dict[tuple[str, str, str], list[TrajectoryFeature]] = defaultdict(list)
    for feature in features:
        grouped[
            (feature.set_id or "", feature.video_id or "", feature.track_id)
        ].append(feature)

    gap_events = {
        "frame_gap_2_5": 0,
        "frame_gap_6_15": 0,
        "frame_gap_gt_15": 0,
    }
    missing_frames = {
        "frame_gap_2_5": 0,
        "frame_gap_6_15": 0,
        "frame_gap_gt_15": 0,
    }
    tracks_with_missing = 0
    tracks_with_long_gap = 0
    for identity in sorted(grouped):
        track_features = sorted(
            grouped[identity], key=lambda feature: feature.frame_index
        )
        track_missing = 0
        track_has_long_gap = False
        for previous, current in zip(track_features, track_features[1:]):
            frame_gap = current.frame_index - previous.frame_index
            if frame_gap <= 1:
                continue
            missing = frame_gap - 1
            if frame_gap <= 5:
                bucket = "frame_gap_2_5"
            elif frame_gap <= 15:
                bucket = "frame_gap_6_15"
            else:
                bucket = "frame_gap_gt_15"
                track_has_long_gap = True
            gap_events[bucket] += 1
            missing_frames[bucket] += missing
            track_missing += missing
        if track_missing:
            tracks_with_missing += 1
        if track_has_long_gap:
            tracks_with_long_gap += 1

    total_missing = sum(missing_frames.values())
    return {
        "gap_events": gap_events,
        "missing_frames": missing_frames,
        "total_gap_events": sum(gap_events.values()),
        "total_missing_frames": total_missing,
        "tracks_with_missing_observations": tracks_with_missing,
        "tracks_with_long_gaps": tracks_with_long_gap,
        "long_gap_missing_frame_share": (
            missing_frames["frame_gap_gt_15"] / total_missing
            if total_missing
            else None
        ),
    }
