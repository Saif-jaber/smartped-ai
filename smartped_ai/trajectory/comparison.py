from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..data.prepare_detection import read_pedestrian_xml
from .calibration import classification_metrics
from .models import (
    MOVEMENT_STANDING,
    MOVEMENT_WALKING,
    TrajectoryFeature,
)


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _valid_bbox(
    values: Iterable[float | None],
) -> tuple[float, float, float, float] | None:
    coordinates = list(values)
    if len(coordinates) != 4 or any(value is None for value in coordinates):
        return None
    x1, y1, x2, y2 = (float(value) for value in coordinates if value is not None)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def _annotation_identity(path: Path) -> tuple[str | None, str]:
    set_id = path.parent.name if path.parent.name.startswith("set") else None
    video_id = (
        path.name[: -len("_annt.xml")]
        if path.name.endswith("_annt.xml")
        else path.stem
    )
    return set_id, video_id


def _resolve_annotation_paths(
    source: Path,
    features: list[TrajectoryFeature],
) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"PIE annotation source does not exist: {source}")
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"PIE annotation source is not a file or directory: {source}")

    paths = sorted(source.rglob("*_annt.xml"))
    if not paths:
        raise FileNotFoundError(
            f"No PIE spatial annotation XML files (*_annt.xml) found under {source}"
        )

    predicted = {
        (feature.set_id, feature.video_id)
        for feature in features
        if feature.video_id
    }
    matched: list[Path] = []
    for path in paths:
        set_id, video_id = _annotation_identity(path)
        if any(
            video_id == predicted_video
            and (
                not predicted_set
                or not set_id
                or predicted_set == set_id
            )
            for predicted_set, predicted_video in predicted
        ):
            matched.append(path)
    if matched:
        return matched
    if len(paths) == 1:
        return paths
    raise ValueError(
        "Could not select a PIE annotation file matching the predicted video. "
        "Pass the specific *_annt.xml file with --annotations."
    )


def _load_ground_truth(
    paths: list[Path],
) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        _width, _height, boxes = read_pedestrian_xml(path)
        for box in boxes:
            if box.outside or not box.track_id:
                continue
            bbox = _valid_bbox(box.bbox_xyxy)
            if bbox is None:
                continue
            by_frame[box.frame_index].append(
                {
                    "track_id": box.track_id,
                    "bbox_xyxy": bbox,
                    "action": (box.attributes.get("action") or "").strip().lower(),
                }
            )
    return by_frame


def _match_frame(
    ground_truth: list[dict[str, Any]],
    predictions: list[TrajectoryFeature],
    iou_threshold: float,
) -> list[tuple[dict[str, Any], TrajectoryFeature, float]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_box in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            overlap = _iou(gt_box["bbox_xyxy"], prediction.bbox_xyxy)
            if overlap >= iou_threshold:
                candidates.append((overlap, gt_index, prediction_index))
    candidates.sort(reverse=True)

    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[dict[str, Any], TrajectoryFeature, float]] = []
    for overlap, gt_index, prediction_index in candidates:
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        matches.append(
            (
                ground_truth[gt_index],
                predictions[prediction_index],
                overlap,
            )
        )
    return matches


def compare_predictions_to_pie(
    features: list[TrajectoryFeature],
    annotation_source: Path,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1.")
    if not features:
        raise ValueError("Cannot compare an empty prediction set with PIE annotations.")

    paths = _resolve_annotation_paths(annotation_source, features)
    ground_truth_by_frame = _load_ground_truth(paths)
    predictions_by_frame: dict[int, list[TrajectoryFeature]] = defaultdict(list)
    for feature in features:
        predictions_by_frame[feature.frame_index].append(feature)
    all_prediction_tracks = len(
        {
            (feature.set_id or "", feature.video_id or "", feature.track_id)
            for feature in features
        }
    )

    predicted_frames = sorted(predictions_by_frame)
    annotation_frames = sorted(ground_truth_by_frame)
    overlap_start = max(predicted_frames[0], annotation_frames[0])
    overlap_end = min(predicted_frames[-1], annotation_frames[-1])
    if overlap_start > overlap_end:
        raise ValueError(
            "Predicted frame range does not overlap the PIE annotation frame range."
        )

    evaluated_frames = [
        frame_index
        for frame_index in annotation_frames
        if overlap_start <= frame_index <= overlap_end
    ]
    gt_tracks: set[str] = set()
    predicted_tracks: set[str] = set()
    gt_tracks_matched: set[str] = set()
    predicted_tracks_matched: set[tuple[str, str, str]] = set()
    gt_boxes = 0
    predicted_boxes_on_gt_frames = 0
    matched_boxes = 0
    matched_iou_total = 0.0
    frames_with_match = 0
    frames_with_gt_without_predictions = 0
    gt_action_counts: Counter[str] = Counter()
    predicted_matched_state_counts: Counter[str] = Counter()
    comparable_actions = 0
    agreeing_actions = 0
    action_pairs: list[tuple[str, str]] = []

    for frame_index in evaluated_frames:
        ground_truth = ground_truth_by_frame[frame_index]
        predictions = predictions_by_frame.get(frame_index, [])
        gt_boxes += len(ground_truth)
        predicted_boxes_on_gt_frames += len(predictions)
        gt_tracks.update(box["track_id"] for box in ground_truth)
        predicted_tracks.update(
            (feature.set_id or "", feature.video_id or "", feature.track_id)
            for feature in predictions
        )
        if not predictions:
            frames_with_gt_without_predictions += 1
            continue

        matches = _match_frame(ground_truth, predictions, iou_threshold)
        if matches:
            frames_with_match += 1
        matched_boxes += len(matches)
        matched_iou_total += sum(overlap for _gt, _prediction, overlap in matches)
        for ground_truth_box, prediction, _overlap in matches:
            gt_tracks_matched.add(ground_truth_box["track_id"])
            predicted_tracks_matched.add(
                (
                    prediction.set_id or "",
                    prediction.video_id or "",
                    prediction.track_id,
                )
            )
            action = ground_truth_box["action"]
            state = prediction.smoothed_movement_state
            if action in {MOVEMENT_WALKING, MOVEMENT_STANDING}:
                gt_action_counts[action] += 1
                predicted_matched_state_counts[state] += 1
                if state in {MOVEMENT_WALKING, MOVEMENT_STANDING}:
                    comparable_actions += 1
                    action_pairs.append((action, state))
                    if state == action:
                        agreeing_actions += 1

    unmatched_gt_boxes = gt_boxes - matched_boxes
    unmatched_predictions_on_gt_frames = predicted_boxes_on_gt_frames - matched_boxes
    predictions_outside_gt_frames = len(features) - predicted_boxes_on_gt_frames
    return {
        "schema_version": 1,
        "comparison_type": "predictions_vs_pie_ground_truth",
        "ground_truth": {
            "source_files": [path.as_posix() for path in paths],
            "dataset": "PIE spatial XML",
            "labels": "visible pedestrian boxes and action attributes",
            "is_exhaustive_scene_ground_truth": False,
            "note": (
                "PIE annotations cover selected tracked pedestrians and are not "
                "exhaustive scene objects."
            ),
        },
        "predictions": {
            "source": "YOLO + BoT-SORT tracker output",
            "is_ground_truth": False,
            "identity_space": (
                "Tracker-local IDs; they are not PIE track IDs and are not "
                "expected to match by value."
            ),
        },
        "method": {
            "frame_alignment": "integer video frame index",
            "box_matching": (
                "one-to-one greedy matching per frame by descending IoU at or above "
                "the configured threshold"
            ),
            "iou_threshold": iou_threshold,
            "predicted_state": "smoothed_movement_state",
            "ground_truth_action": "PIE pedestrian action attribute",
        },
        "frame_ranges": {
            "prediction": {
                "first": predicted_frames[0],
                "last": predicted_frames[-1],
                "frame_count": len(predicted_frames),
                "observation_count": len(features),
            },
            "annotation": {
                "first": annotation_frames[0],
                "last": annotation_frames[-1],
                "frame_count": len(annotation_frames),
                "visible_box_count": sum(
                    len(boxes) for boxes in ground_truth_by_frame.values()
                ),
                "track_count": len(
                    {
                        box["track_id"]
                        for boxes in ground_truth_by_frame.values()
                        for box in boxes
                    }
                ),
            },
            "overlap": {
                "first": overlap_start,
                "last": overlap_end,
                "annotated_frame_count": len(evaluated_frames),
            },
        },
        "counts": {
            "frames_with_annotated_pedestrians": len(evaluated_frames),
            "frames_with_at_least_one_match": frames_with_match,
            "frames_with_annotated_pedestrians_without_predictions": (
                frames_with_gt_without_predictions
            ),
            "ground_truth_boxes_on_evaluated_frames": gt_boxes,
            "prediction_boxes_on_evaluated_frames": predicted_boxes_on_gt_frames,
            "matched_boxes": matched_boxes,
            "unmatched_ground_truth_boxes": unmatched_gt_boxes,
            "unmatched_prediction_boxes_on_evaluated_frames": (
                unmatched_predictions_on_gt_frames
            ),
            "prediction_observations_outside_annotated_frames": (
                predictions_outside_gt_frames
            ),
            "ground_truth_box_match_rate": _ratio(matched_boxes, gt_boxes),
            "prediction_box_match_rate_on_annotated_frames": _ratio(
                matched_boxes, predicted_boxes_on_gt_frames
            ),
            "frame_match_rate_on_annotated_frames": _ratio(
                frames_with_match, len(evaluated_frames)
            ),
            "mean_iou_of_matched_boxes": (
                matched_iou_total / matched_boxes if matched_boxes else None
            ),
            "all_prediction_tracks": all_prediction_tracks,
            "ground_truth_tracks_on_evaluated_frames": len(gt_tracks),
            "predicted_tracks_on_evaluated_frames": len(predicted_tracks),
            "ground_truth_tracks_with_a_match_on_evaluated_frames": len(
                gt_tracks_matched
            ),
            "predicted_tracks_with_a_match_on_evaluated_frames": len(
                predicted_tracks_matched
            ),
        },
        "walking_standing_action_comparison": {
            "comparable_matched_observations": comparable_actions,
            "agreeing_observations": agreeing_actions,
            "agreement_ratio": _ratio(agreeing_actions, comparable_actions),
            "ground_truth_action_counts": dict(gt_action_counts),
            "predicted_state_counts": dict(predicted_matched_state_counts),
            "metrics": classification_metrics(
                [truth for truth, _prediction in action_pairs],
                [prediction for _truth, prediction in action_pairs],
            ),
            "note": (
                "Only matched boxes with PIE action walking/standing and a known "
                "predicted state are compared. PIE action labels and velocity-derived "
                "states are different measurement sources."
            ),
        },
        "notes": [
            "Predicted tracks are YOLO + BoT-SORT outputs, not ground truth.",
            "PIE spatial annotations are not exhaustive scene ground truth; an unmatched prediction may be an unannotated person.",
            "An unmatched annotated box can reflect a detector miss, occlusion, low confidence, or tracker/frame mismatch.",
            "Tracker IDs and PIE track IDs use different identity spaces; comparisons are frame-local box matches, not identity metrics such as MOTA or IDF1.",
            "Predictions outside frames with visible PIE pedestrian annotations are excluded from box matching.",
        ],
    }
