from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .models import TrackObservation

CAMERA_MOTION_METHOD_LOCAL = "local_optical_flow"
CAMERA_MOTION_METHOD_GLOBAL = "global_affine_fallback"
CAMERA_MOTION_METHOD_MIXED = "mixed_local_and_global"
CAMERA_MOTION_METHOD_UNAVAILABLE = "unavailable"

@dataclass(frozen=True)
class CameraMotionConfig:
    max_width_px: int = 960
    feature_max_corners: int = 2500
    feature_quality_level: float = 0.01
    feature_min_distance_px: float = 10.0
    feature_block_size_px: int = 7
    lk_window_size_px: int = 21
    lk_max_level: int = 3
    lk_backward_error_px: float = 1.0
    local_radius_px: float = 160.0
    min_local_flow_samples: int = 8
    mask_padding_px: float = 6.0
    ransac_reprojection_threshold_px: float = 3.0
    min_global_inliers: int = 20

    def validate(self) -> None:
        if self.max_width_px < 64:
            raise ValueError("camera motion max_width_px must be at least 64.")
        if self.feature_max_corners < 8:
            raise ValueError("camera motion feature_max_corners must be at least 8.")
        if not 0 < self.feature_quality_level <= 1:
            raise ValueError(
                "camera motion feature_quality_level must be in (0, 1]."
            )
        if self.feature_min_distance_px < 1:
            raise ValueError(
                "camera motion feature_min_distance_px must be at least 1."
            )
        if self.feature_block_size_px < 3:
            raise ValueError("camera motion feature_block_size_px must be at least 3.")
        if self.lk_window_size_px < 3:
            raise ValueError("camera motion lk_window_size_px must be at least 3.")
        if self.lk_max_level < 0:
            raise ValueError("camera motion lk_max_level cannot be negative.")
        if self.lk_backward_error_px <= 0:
            raise ValueError("camera motion lk_backward_error_px must be positive.")
        if self.local_radius_px <= 0:
            raise ValueError("camera motion local_radius_px must be positive.")
        if self.min_local_flow_samples < 1:
            raise ValueError(
                "camera motion min_local_flow_samples must be at least 1."
            )
        if self.mask_padding_px < 0:
            raise ValueError("camera motion mask_padding_px cannot be negative.")
        if self.ransac_reprojection_threshold_px <= 0:
            raise ValueError(
                "camera motion ransac_reprojection_threshold_px must be positive."
            )
        if self.min_global_inliers < 3:
            raise ValueError("camera motion min_global_inliers must be at least 3.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_width_px": self.max_width_px,
            "feature_max_corners": self.feature_max_corners,
            "feature_quality_level": self.feature_quality_level,
            "feature_min_distance_px": self.feature_min_distance_px,
            "feature_block_size_px": self.feature_block_size_px,
            "lk_window_size_px": self.lk_window_size_px,
            "lk_max_level": self.lk_max_level,
            "lk_backward_error_px": self.lk_backward_error_px,
            "local_radius_px": self.local_radius_px,
            "min_local_flow_samples": self.min_local_flow_samples,
            "mask_padding_px": self.mask_padding_px,
            "ransac_reprojection_threshold_px": (
                self.ransac_reprojection_threshold_px
            ),
            "min_global_inliers": self.min_global_inliers,
        }


@dataclass(frozen=True)
class CameraMotionEstimate:
    dx_px: float
    dy_px: float
    method: str
    local_sample_count: int
    fallback_step_count: int
    available: bool
    reason: str | None = None

    @property
    def applied(self) -> bool:
        return self.available

    def to_dict(self) -> dict[str, Any]:
        return {
            "dx_px": self.dx_px,
            "dy_px": self.dy_px,
            "method": self.method,
            "local_sample_count": self.local_sample_count,
            "fallback_step_count": self.fallback_step_count,
            "available": self.available,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _PairMotion:
    source_points: Any
    flow: Any
    global_affine: Any | None
    global_inlier_count: int
    match_count: int

    def sample_local(
        self,
        point: tuple[float, float],
        *,
        radius_px: float,
        min_samples: int,
    ) -> tuple[float, float] | None:
        import numpy as np

        if len(self.source_points) < min_samples:
            return None
        distances = np.linalg.norm(
            self.source_points - np.asarray(point, dtype=float), axis=1
        )
        selected = distances <= radius_px
        if int(selected.sum()) < min_samples:
            return None
        local = np.median(self.flow[selected], axis=0)
        return float(local[0]), float(local[1])

    def map_point(self, point: tuple[float, float]) -> tuple[float, float] | None:
        if self.global_affine is None:
            return None
        matrix = self.global_affine
        return (
            float(matrix[0, 0] * point[0] + matrix[0, 1] * point[1] + matrix[0, 2]),
            float(matrix[1, 0] * point[0] + matrix[1, 1] * point[1] + matrix[1, 2]),
        )


@dataclass(frozen=True)
class CameraMotionSequence:
    video_path: Path
    config: CameraMotionConfig
    pairwise: dict[int, _PairMotion]
    frames_read: int
    source_width: int
    source_height: int
    working_width: int
    working_height: int

    @property
    def scale(self) -> float:
        if self.source_width <= 0:
            return 1.0
        return self.working_width / self.source_width

    def displacement_at(
        self,
        point: tuple[float, float],
        frame_from: int,
        frame_to: int,
    ) -> CameraMotionEstimate:
        if frame_to <= frame_from:
            raise ValueError("frame_to must be greater than frame_from.")
        current = point
        local_samples = 0
        fallback_steps = 0
        for frame_index in range(frame_from, frame_to):
            pair = self.pairwise.get(frame_index)
            if pair is None:
                return CameraMotionEstimate(
                    0.0,
                    0.0,
                    CAMERA_MOTION_METHOD_UNAVAILABLE,
                    local_samples,
                    fallback_steps,
                    False,
                    "missing_pair_motion",
                )
            local = pair.sample_local(
                current,
                radius_px=self.config.local_radius_px,
                min_samples=self.config.min_local_flow_samples,
            )
            if local is not None:
                delta_x, delta_y = local
                local_samples += 1
            else:
                mapped = pair.map_point(current)
                if mapped is None:
                    return CameraMotionEstimate(
                        0.0,
                        0.0,
                        CAMERA_MOTION_METHOD_UNAVAILABLE,
                        local_samples,
                        fallback_steps,
                        False,
                        "insufficient_local_and_global_motion",
                    )
                delta_x = mapped[0] - current[0]
                delta_y = mapped[1] - current[1]
                fallback_steps += 1
            current = (current[0] + delta_x, current[1] + delta_y)

        delta_x = current[0] - point[0]
        delta_y = current[1] - point[1]
        if fallback_steps == 0:
            method = CAMERA_MOTION_METHOD_LOCAL
        elif local_samples == 0:
            method = CAMERA_MOTION_METHOD_GLOBAL
        else:
            method = CAMERA_MOTION_METHOD_MIXED
        return CameraMotionEstimate(
            delta_x,
            delta_y,
            method,
            local_samples,
            fallback_steps,
            True,
        )

    def with_local_radius(self, radius_px: float) -> CameraMotionSequence:
        if radius_px <= 0:
            raise ValueError("local radius must be positive.")
        return replace(
            self,
            config=replace(self.config, local_radius_px=float(radius_px)),
        )

    def quality_summary(
        self,
        observations: Iterable[TrackObservation] | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        pairs = list(self.pairwise.values())
        match_counts = [pair.match_count for pair in pairs]
        inlier_counts = [
            pair.global_inlier_count
            for pair in pairs
            if pair.global_affine is not None
        ]
        summary: dict[str, Any] = {
            "video": self.video_path.as_posix(),
            "frames_read": self.frames_read,
            "frame_pairs": len(pairs),
            "source_width": self.source_width,
            "source_height": self.source_height,
            "working_width": self.working_width,
            "working_height": self.working_height,
            "working_scale": self.scale,
            "pair_coverage_ratio": (
                len(pairs) / max(0, self.frames_read - 1)
                if self.frames_read > 1
                else None
            ),
            "mean_flow_point_count": (
                float(np.mean(match_counts)) if match_counts else None
            ),
            "min_flow_point_count": min(match_counts) if match_counts else None,
            "max_flow_point_count": max(match_counts) if match_counts else None,
            "pairs_with_global_affine": len(inlier_counts),
            "mean_global_inliers": (
                float(np.mean(inlier_counts)) if inlier_counts else None
            ),
        }
        if observations is not None:
            methods: dict[str, int] = {}
            applied = 0
            unavailable = 0
            local_only = 0
            global_fallback = 0
            mixed = 0
            for observation in observations:
                if observation.frame_index <= 0:
                    continue
                estimate = self.displacement_at(
                    observation.center_xy,
                    observation.frame_index - 1,
                    observation.frame_index,
                )
                if estimate.available:
                    applied += 1
                    methods[estimate.method] = methods.get(estimate.method, 0) + 1
                    if estimate.method == CAMERA_MOTION_METHOD_LOCAL:
                        local_only += 1
                    elif estimate.method == CAMERA_MOTION_METHOD_GLOBAL:
                        global_fallback += 1
                    elif estimate.method == CAMERA_MOTION_METHOD_MIXED:
                        mixed += 1
                else:
                    unavailable += 1
            total = applied + unavailable
            summary["observation_compensation"] = {
                "observations_evaluated": total,
                "applied": applied,
                "unavailable": unavailable,
                "applied_ratio": applied / total if total else None,
                "local_only": local_only,
                "global_fallback_only": global_fallback,
                "mixed": mixed,
                "methods": methods,
            }
        return summary


def _foreground_mask(
    shape: tuple[int, int],
    boxes: Iterable[tuple[float, float, float, float]],
    *,
    scale: float,
    padding_px: float,
) -> Any:
    import cv2
    import numpy as np

    mask = np.full(shape, 255, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        left = max(0, int(math.floor((x1 - padding_px) * scale)))
        top = max(0, int(math.floor((y1 - padding_px) * scale)))
        right = int(math.ceil((x2 + padding_px) * scale))
        bottom = int(math.ceil((y2 + padding_px) * scale))
        cv2.rectangle(mask, (left, top), (right, bottom), 0, -1)
    return mask


def _detect_features(
    gray: Any,
    boxes: Iterable[tuple[float, float, float, float]],
    config: CameraMotionConfig,
    *,
    scale: float,
) -> Any | None:
    import cv2

    mask = _foreground_mask(
        gray.shape,
        boxes,
        scale=scale,
        padding_px=config.mask_padding_px,
    )
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=config.feature_max_corners,
        qualityLevel=config.feature_quality_level,
        minDistance=config.feature_min_distance_px,
        blockSize=config.feature_block_size_px,
        mask=mask,
    )
    return points


def estimate_pair_motion(
    previous_gray: Any,
    current_gray: Any,
    previous_boxes: Iterable[tuple[float, float, float, float]],
    config: CameraMotionConfig,
    *,
    scale: float = 1.0,
) -> _PairMotion | None:
    import cv2
    import numpy as np

    config.validate()
    previous_points = _detect_features(
        previous_gray, previous_boxes, config, scale=scale
    )
    if previous_points is None or len(previous_points) < 3:
        return None

    next_points, status, _error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        winSize=(config.lk_window_size_px, config.lk_window_size_px),
        maxLevel=config.lk_max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    back_points, back_status, _back_error = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        next_points,
        None,
        winSize=(config.lk_window_size_px, config.lk_window_size_px),
        maxLevel=config.lk_max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    backward_error = np.linalg.norm(
        back_points.reshape(-1, 2) - previous_points.reshape(-1, 2), axis=1
    )
    valid = (
        (status.reshape(-1) == 1)
        & (back_status.reshape(-1) == 1)
        & (backward_error <= config.lk_backward_error_px * scale)
    )
    if int(valid.sum()) < 3:
        return None

    source_scaled = previous_points.reshape(-1, 2)[valid]
    target_scaled = next_points.reshape(-1, 2)[valid]
    source_points = source_scaled / scale
    target_points = target_scaled / scale
    flow = target_points - source_points

    global_affine = None
    global_inliers = 0
    if len(source_scaled) >= config.min_global_inliers:
        matrix, mask = cv2.estimateAffinePartial2D(
            source_scaled,
            target_scaled,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.ransac_reprojection_threshold_px,
            maxIters=2000,
            confidence=0.995,
        )
        if matrix is not None:
            global_inliers = int(mask.sum()) if mask is not None else 0
            if global_inliers >= config.min_global_inliers:
                matrix = matrix.copy()
                matrix[0, 2] /= scale
                matrix[1, 2] /= scale
                global_affine = matrix

    return _PairMotion(
        source_points=source_points,
        flow=flow,
        global_affine=global_affine,
        global_inlier_count=global_inliers,
        match_count=len(source_points),
    )


def estimate_camera_motion(
    video_path: Path,
    observations: Iterable[TrackObservation],
    config: CameraMotionConfig,
    *,
    max_frame: int | None = None,
) -> CameraMotionSequence:
    import cv2
    import numpy as np

    config.validate()
    observations = list(observations)
    boxes_by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    for observation in observations:
        boxes_by_frame.setdefault(observation.frame_index, []).append(
            observation.bbox_xyxy
        )
    if max_frame is None:
        max_frame = max(
            (observation.frame_index for observation in observations), default=-1
        ) + 1
    if max_frame <= 1:
        raise ValueError("Camera motion requires at least two frames.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video for camera motion: {video_path}")
    try:
        ok, first_frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read the first video frame: {video_path}")
        source_height, source_width = first_frame.shape[:2]
        scale = min(1.0, config.max_width_px / source_width)
        working_width = max(1, int(round(source_width * scale)))
        working_height = max(1, int(round(source_height * scale)))

        def gray_of(frame: Any) -> Any:
            if working_width != source_width or working_height != source_height:
                frame = cv2.resize(
                    frame,
                    (working_width, working_height),
                    interpolation=cv2.INTER_AREA,
                )
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        previous_gray = gray_of(first_frame)
        previous_points = _detect_features(
            previous_gray, boxes_by_frame.get(0, []), config, scale=scale
        )
        pairwise: dict[int, _PairMotion] = {}
        frames_read = 1
        frame_index = 1
        while frame_index < max_frame:
            ok, frame = capture.read()
            if not ok:
                break
            frames_read += 1
            current_gray = gray_of(frame)
            if previous_points is not None and len(previous_points) >= 3:
                next_points, status, _error = cv2.calcOpticalFlowPyrLK(
                    previous_gray,
                    current_gray,
                    previous_points,
                    None,
                    winSize=(config.lk_window_size_px, config.lk_window_size_px),
                    maxLevel=config.lk_max_level,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        30,
                        0.01,
                    ),
                )
                back_points, back_status, _back_error = cv2.calcOpticalFlowPyrLK(
                    current_gray,
                    previous_gray,
                    next_points,
                    None,
                    winSize=(config.lk_window_size_px, config.lk_window_size_px),
                    maxLevel=config.lk_max_level,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        30,
                        0.01,
                    ),
                )
                backward_error = np.linalg.norm(
                    back_points.reshape(-1, 2) - previous_points.reshape(-1, 2),
                    axis=1,
                )
                valid = (
                    (status.reshape(-1) == 1)
                    & (back_status.reshape(-1) == 1)
                    & (backward_error <= config.lk_backward_error_px * scale)
                )
                if int(valid.sum()) >= 3:
                    source_scaled = previous_points.reshape(-1, 2)[valid]
                    target_scaled = next_points.reshape(-1, 2)[valid]
                    source_points = source_scaled / scale
                    target_points = target_scaled / scale
                    flow = target_points - source_points
                    global_affine = None
                    global_inliers = 0
                    if len(source_scaled) >= config.min_global_inliers:
                        matrix, mask = cv2.estimateAffinePartial2D(
                            source_scaled,
                            target_scaled,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=(
                                config.ransac_reprojection_threshold_px
                            ),
                            maxIters=2000,
                            confidence=0.995,
                        )
                        if matrix is not None:
                            global_inliers = (
                                int(mask.sum()) if mask is not None else 0
                            )
                            if global_inliers >= config.min_global_inliers:
                                matrix = matrix.copy()
                                matrix[0, 2] /= scale
                                matrix[1, 2] /= scale
                                global_affine = matrix
                    pairwise[frame_index - 1] = _PairMotion(
                        source_points=source_points,
                        flow=flow,
                        global_affine=global_affine,
                        global_inlier_count=global_inliers,
                        match_count=len(source_points),
                    )
            previous_gray = current_gray
            previous_points = _detect_features(
                current_gray,
                boxes_by_frame.get(frame_index, []),
                config,
                scale=scale,
            )
            frame_index += 1
    finally:
        capture.release()

    return CameraMotionSequence(
        video_path=video_path,
        config=config,
        pairwise=pairwise,
        frames_read=frames_read,
        source_width=source_width,
        source_height=source_height,
        working_width=working_width,
        working_height=working_height,
    )
