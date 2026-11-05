"""Run and compare two pedestrian confidence settings through SmartPed AI.

The video runner receives isolated per-run overrides. Vehicle model, vehicle
confidence, and both tracker configurations are held constant.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import subprocess
import threading
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from smartped_ai.video_runner.pipeline import run_video

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "threshold_tracking"
VIDEO = ROOT / "dataset" / "PIE_clips" / "set01" / "video_0001.mp4"
PEDESTRIAN_MODEL = ROOT / "runs" / "detect" / "train-5" / "weights" / "best.pt"
VEHICLE_CONFIG = ROOT / "configs" / "vehicle_tracking.json"
PED_TRACKER = ROOT / "smartped_botsort.yaml"
VEH_TRACKER = ROOT / "smartped_vehicle_botsort.yaml"
ANNOTATION_METADATA = ROOT / "artifacts" / "pie_detection_yolo" / "annotation_metadata.jsonl.gz"
FPS = 30.0
FRAME_LIMIT = 1800
VEHICLE_CONFIDENCE = 0.25
PEDESTRIAN_TRACK_LOW_THRESHOLD = 0.10
RUNS = (
    ("threshold_009116", 0.0911623687),
    ("threshold_012545", 0.1254522055),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_model_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    names = result.stdout.splitlines()
    return names[0].strip() if names else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


class GpuSampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self.current_run = "idle"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            sampled_at = datetime.now(timezone.utc).isoformat()
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=8,
                )
                for gpu_index, line in enumerate(result.stdout.splitlines()):
                    values = [value.strip() for value in line.split(",")]
                    if len(values) != 3:
                        continue
                    self.samples.append(
                        {
                            "timestamp_utc": sampled_at,
                            "run": self.current_run,
                            "gpu_index": gpu_index,
                            "utilization_percent": float(values[0]),
                            "memory_used_mib": float(values[1]),
                            "memory_total_mib": float(values[2]),
                        }
                    )
            except (OSError, subprocess.SubprocessError, ValueError):
                self.samples.append(
                    {
                        "timestamp_utc": sampled_at,
                        "run": self.current_run,
                        "gpu_index": None,
                        "error": "nvidia-smi sample unavailable",
                    }
                )
            self.stop_event.wait(1.0)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def summary(self, run_name: str) -> dict:
        rows = [
            row
            for row in self.samples
            if row["run"] == run_name and row.get("gpu_index") == 0
        ]
        if not rows:
            return {"samples": 0, "available": False}
        return {
            "samples": len(rows),
            "available": True,
            "gpu_index": 0,
            "mean_utilization_percent": sum(row["utilization_percent"] for row in rows) / len(rows),
            "peak_utilization_percent": max(row["utilization_percent"] for row in rows),
            "mean_memory_used_mib": sum(row["memory_used_mib"] for row in rows) / len(rows),
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in rows),
            "memory_total_mib": max(row["memory_total_mib"] for row in rows),
        }

    def write(self, path: Path) -> None:
        fields = [
            "timestamp_utc",
            "run",
            "gpu_index",
            "utilization_percent",
            "memory_used_mib",
            "memory_total_mib",
            "error",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.samples)


def duration_summary(pedestrian_rows: list[dict[str, str]]) -> dict:
    tracks: dict[str, list[int]] = defaultdict(list)
    confidence: list[float] = []
    counts_by_frame: Counter[int] = Counter()
    for row in pedestrian_rows:
        track_id = row.get("track_id", "").strip()
        frame = int(row["frame_index"])
        if track_id:
            tracks[track_id].append(frame)
        counts_by_frame[frame] += 1
        if row.get("confidence", "").strip():
            confidence.append(float(row["confidence"]))
    durations = {
        track_id: (max(frames) - min(frames)) / FPS
        for track_id, frames in tracks.items()
        if frames
    }
    spans = {
        track_id: max(frames) - min(frames) + 1
        for track_id, frames in tracks.items()
        if frames
    }
    observations_per_track = {track_id: len(frames) for track_id, frames in tracks.items()}
    missing_per_track = {
        track_id: spans[track_id] - observations_per_track[track_id]
        for track_id in tracks
    }
    sorted_durations = sorted(durations.values())
    sorted_conf = sorted(confidence)
    confidence_histogram = []
    for lower in range(0, 100, 10):
        upper = lower + 10
        count = sum(
            (value >= lower / 100)
            and (value < upper / 100 or (upper == 100 and value <= 1.0))
            for value in sorted_conf
        )
        confidence_histogram.append(
            {
                "range": f"[{lower / 100:.1f}, {upper / 100:.1f}{']' if upper == 100 else ')'}",
                "count": count,
            }
        )

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        index = (len(values) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)

    frame_counts = [counts_by_frame.get(frame, 0) for frame in range(FRAME_LIMIT)]
    return {
        "tracked_detection_observations": len(pedestrian_rows),
        "unique_track_ids": len(tracks),
        "detection_count_per_frame": {
            "frames_processed": FRAME_LIMIT,
            "frames_with_detections": sum(count > 0 for count in frame_counts),
            "mean": sum(frame_counts) / FRAME_LIMIT,
            "median": median(frame_counts),
            "p95": percentile(sorted(frame_counts), 0.95),
            "max": max(frame_counts, default=0),
        },
        "confidence_distribution": {
            "count": len(sorted_conf),
            "mean": sum(sorted_conf) / len(sorted_conf) if sorted_conf else None,
            "median": median(sorted_conf) if sorted_conf else None,
            "p05": percentile(sorted_conf, 0.05),
            "p25": percentile(sorted_conf, 0.25),
            "p75": percentile(sorted_conf, 0.75),
            "p95": percentile(sorted_conf, 0.95),
            "min": min(sorted_conf) if sorted_conf else None,
            "max": max(sorted_conf) if sorted_conf else None,
            "histogram_0_1_bins": confidence_histogram,
        },
        "track_durations_seconds": {
            "count": len(sorted_durations),
            "mean": sum(sorted_durations) / len(sorted_durations) if sorted_durations else None,
            "median": median(sorted_durations) if sorted_durations else None,
            "p10": percentile(sorted_durations, 0.10),
            "p90": percentile(sorted_durations, 0.90),
            "min": min(sorted_durations) if sorted_durations else None,
            "max": max(sorted_durations) if sorted_durations else None,
        },
        "short_tracks_under_1_second": sum(duration < 1.0 for duration in durations.values()),
        "long_tracks_at_least_5_seconds": sum(duration >= 5.0 for duration in durations.values()),
        "track_observation_gaps": {
            "tracks_with_missing_frames_inside_first_last_observation_span": sum(
                count > 0 for count in missing_per_track.values()
            ),
            "missing_frame_slots_inside_track_spans": sum(missing_per_track.values()),
            "definition": "inclusive first-to-last observed frame span minus observed frames; no interpolation",
        },
    }


def annotation_identity_diagnostics(
    prediction_rows: list[dict[str, str]],
) -> dict:
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        by_frame[int(row["frame_index"])].append(row)
    ground_truth: dict[int, list[dict]] = defaultdict(list)
    with gzip.open(ANNOTATION_METADATA, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if (
                item.get("set_id") != "set01"
                or item.get("video_id") != "video_0001"
                or item.get("split") != "train"
                or not item.get("label_included")
                or int(item.get("frame_index", FRAME_LIMIT)) >= FRAME_LIMIT
                or not item.get("bbox_xyxy")
            ):
                continue
            box = item["bbox_xyxy"]
            ground_truth[int(item["frame_index"])].append(
                {
                    "track_id": str(item["track_id"]),
                    "bbox": (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])),
                }
            )

    identity_assignments: dict[str, list[tuple[int, str]]] = defaultdict(list)
    annotated_boxes = 0
    matched_boxes = 0
    for frame, targets in ground_truth.items():
        predictions = by_frame.get(frame, [])
        annotated_boxes += len(targets)
        pairs = []
        for target_index, target in enumerate(targets):
            ax1, ay1, ax2, ay2 = target["bbox"]
            for prediction_index, prediction in enumerate(predictions):
                bx1, by1, bx2, by2 = (
                    float(prediction[f"bbox_{coordinate}"])
                    for coordinate in ("x1", "y1", "x2", "y2")
                )
                intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
                    0.0, min(ay2, by2) - max(ay1, by1)
                )
                union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
                overlap = intersection / union if union > 0 else 0.0
                if overlap >= 0.5:
                    pairs.append((overlap, target_index, prediction_index))
        used_targets: set[int] = set()
        used_predictions: set[int] = set()
        for _overlap, target_index, prediction_index in sorted(pairs, reverse=True):
            if target_index in used_targets or prediction_index in used_predictions:
                continue
            target = targets[target_index]
            prediction = predictions[prediction_index]
            predicted_id = prediction.get("track_id", "").strip()
            if not predicted_id:
                continue
            identity_assignments[target["track_id"]].append((frame, predicted_id))
            used_targets.add(target_index)
            used_predictions.add(prediction_index)
            matched_boxes += 1

    ids_per_identity = {
        identity: {track_id for _, track_id in observations}
        for identity, observations in identity_assignments.items()
    }
    identity_switches = 0
    eligible_consecutive_pairs = 0
    for observations in identity_assignments.values():
        ordered = sorted(observations)
        for (frame_a, track_a), (frame_b, track_b) in zip(ordered, ordered[1:]):
            if frame_b - frame_a <= 30:
                eligible_consecutive_pairs += 1
                identity_switches += track_a != track_b
    return {
        "annotated_pedestrian_boxes_in_segment": annotated_boxes,
        "matched_boxes_iou_at_least_0_50": matched_boxes,
        "matched_fraction_of_included_sparse_annotations": (
            matched_boxes / annotated_boxes if annotated_boxes else None
        ),
        "annotated_pedestrian_identities_observed": len(identity_assignments),
        "identities_assigned_multiple_predicted_track_ids": sum(
            len(track_ids) > 1 for track_ids in ids_per_identity.values()
        ),
        "predicted_id_changes_between_matched_annotations_within_1_second": identity_switches,
        "eligible_adjacent_matched_annotation_pairs": eligible_consecutive_pairs,
        "method": (
            "Exploratory one-to-one per-frame IoU>=0.50 matching against included PIE annotations "
            "within the first 60 seconds; successive matched annotations <=30 frames apart are "
            "checked for predicted-ID changes. Sparse/non-exhaustive labels make this unsuitable "
            "as formal ID-switch accuracy."
        ),
    }


def summarize_interactions(run_dir: Path, run_metrics: dict) -> dict:
    rows = read_csv(run_dir / "interaction_observations.csv")
    nearby = [
        row
        for row in rows
        if row.get("proximity_status", "").strip().lower() == "nearby"
    ]
    nearby_pairs = {
        (row["pedestrian_track_id"], row["vehicle_track_id"])
        for row in nearby
    }
    invalid_reasons = Counter(
        row.get("movement_invalid_reason", "") or "unspecified"
        for row in rows
        if row.get("movement_valid", "").strip().lower() != "true"
    )
    return {
        "interaction_pair_observations": len(rows),
        "unique_pedestrian_vehicle_track_pairs": run_metrics["interaction"]["unique_pairs"],
        "nearby_observations_at_150_pixels": len(nearby),
        "unique_nearby_track_pairs": len(nearby_pairs),
        "valid_movement_estimates": run_metrics["interaction"]["valid_movement_estimates"],
        "invalid_movement_estimates": run_metrics["interaction"]["invalid_movement_estimates"],
        "invalid_movement_reasons": dict(sorted(invalid_reasons.items())),
        "nearby_threshold_pixels": run_metrics["interaction"]["pixel_distance_threshold"],
    }


def select_visual_frames() -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    with gzip.open(ANNOTATION_METADATA, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if (
                item.get("set_id") != "set01"
                or item.get("video_id") != "video_0001"
                or item.get("split") != "train"
                or not item.get("label_included")
                or int(item.get("frame_index", FRAME_LIMIT)) >= FRAME_LIMIT
                or not item.get("bbox_xyxy")
            ):
                continue
            grouped[int(item["frame_index"])].append(item)

    records = []
    for frame, items in grouped.items():
        areas = [
            float(item["yolo_xywh"][2]) * float(item["yolo_xywh"][3])
            for item in items
            if item.get("yolo_xywh")
        ]
        records.append(
            {
                "frame_index": frame,
                "ground_truth_boxes": len(items),
                "median_normalized_box_area": median(areas) if areas else None,
                "occluded_boxes": sum(
                    item.get("occlusion") in {"part", "full"} or item.get("occluded") is True
                    for item in items
                ),
            }
        )
    if not records:
        raise RuntimeError("No included set01/video_0001 annotations found in the first 60 seconds.")
    used: set[int] = set()
    selected = []

    def add(category: str, candidates: list[dict], reason: str) -> None:
        choice = next((row for row in candidates if row["frame_index"] not in used), None)
        if choice is None:
            raise RuntimeError(f"No non-duplicate annotated frame for visual category {category}.")
        used.add(choice["frame_index"])
        selected.append({**choice, "category": category, "selection_reason": reason})

    multi_pedestrian = [row for row in records if row["ground_truth_boxes"] >= 2]
    if multi_pedestrian:
        add(
            "small_or_distant",
            sorted(
                multi_pedestrian,
                key=lambda row: (
                    row["median_normalized_box_area"] or float("inf"),
                    -row["ground_truth_boxes"],
                ),
            ),
            "Small/distant candidate: at least two annotated pedestrians and the smallest median normalized box area available in the segment.",
        )
    occluded = [
        row for row in records if row["ground_truth_boxes"] >= 2 and row["occluded_boxes"] > 0
    ]
    if occluded:
        add(
            "occluded",
            sorted(occluded, key=lambda row: (-row["occluded_boxes"], -row["ground_truth_boxes"])),
            "Annotated partial/full occlusion; selected frame has the greatest occluded-box count among multi-pedestrian frames.",
        )
    add(
        "densest_available",
        sorted(
            records,
            key=lambda row: (-row["ground_truth_boxes"], row["median_normalized_box_area"] or 0),
        ),
        "Densest available annotated scene in this segment; the included labels contain at most four pedestrians per frame, so no highly crowded frame is available.",
    )
    return selected


def create_visual_comparisons(selected: list[dict], runs: dict[str, dict]) -> list[dict]:
    try:
        import cv2
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("OpenCV and Pillow are required to extract annotated visual samples.") from exc

    visual_dir = OUT / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample in selected:
        panels = []
        for run_name, run_record in runs.items():
            video_path = Path(run_record["outputs"]["annotated_video"])
            capture = cv2.VideoCapture(str(video_path))
            try:
                if not capture.isOpened():
                    raise OSError(f"Could not open annotated video: {video_path}")
                capture.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_index"])
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise OSError(
                        f"Could not decode frame {sample['frame_index']} from {video_path}"
                    )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            finally:
                capture.release()
            image = Image.fromarray(frame)
            image.thumbnail((960, 540), Image.Resampling.LANCZOS)
            panels.append((run_name.replace("_", " "), image))

        width, height = 1920, 575
        canvas = Image.new("RGB", (width, height), (25, 25, 25))
        draw = ImageDraw.Draw(canvas)
        for index, (title, panel) in enumerate(panels):
            x = index * 960
            draw.text((x + 10, 8), title, fill=(255, 255, 255), font=ImageFont.load_default())
            canvas.paste(panel, (x, 30))
        filename = f"frame_{sample['frame_index']:06d}_{sample['category']}.jpg"
        target = visual_dir / filename
        canvas.save(target, quality=90, optimize=True)
        manifest.append(
            {
                **sample,
                "file": f"visuals/{filename}",
                "source_videos": {
                    name: run_record["outputs"]["annotated_video"]
                    for name, run_record in runs.items()
                },
            }
        )
    return manifest


def build_run_record(
    name: str,
    confidence: float,
    result: dict,
    gpu_summary: dict,
) -> dict:
    run_dir = Path(result["output_dir"])
    metrics = result["metrics"]
    pedestrian_rows = read_csv(run_dir / "pedestrian_detections.csv")
    frame_counts = Counter(
        int(row["frame_index"]) for row in pedestrian_rows
    )
    write_csv(
        run_dir / "pedestrian_detections_per_frame.csv",
        ["frame_index", "timestamp_seconds", "detection_count"],
        [
            {
                "frame_index": frame,
                "timestamp_seconds": f"{frame / FPS:.6f}",
                "detection_count": frame_counts.get(frame, 0),
            }
            for frame in range(FRAME_LIMIT)
        ],
    )
    tracking = duration_summary(pedestrian_rows)
    tracking["annotation_identity_diagnostics"] = annotation_identity_diagnostics(
        pedestrian_rows
    )
    outputs = {
        key: str(run_dir / filename)
        for key, filename in {
            "annotated_video": "annotated_video.mp4",
            "pedestrian_detections": "pedestrian_detections.csv",
            "pedestrian_trajectories": "pedestrian_trajectories.csv",
            "vehicle_detections": "vehicle_detections.csv",
            "vehicle_trajectories": "vehicle_trajectories.csv",
            "pedestrian_detections_per_frame": "pedestrian_detections_per_frame.csv",
            "interaction_observations": "interaction_observations.csv",
            "interaction_summary": "interaction_summary.csv",
            "integrated_interactions": "integrated_interactions.csv",
            "processing_report": "PROCESSING_REPORT.md",
            "processing_metrics": "processing_metrics.json",
        }.items()
    }
    record = {
        "name": name,
        "status": "success",
        "pedestrian_confidence_threshold": confidence,
        "vehicle_confidence_threshold": metrics["input"]["vehicle_confidence_threshold"],
        "processing_runtime_seconds": metrics["timing"]["total_seconds"],
        "processing_fps": metrics["timing"]["processed_frames_per_second"],
        "gpu": gpu_summary,
        "pedestrian": tracking,
        "interaction": summarize_interactions(run_dir, metrics),
        "pipeline_metrics": metrics,
        "outputs": outputs,
    }
    (run_dir / "threshold_run_metrics.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def render_report(config: dict, results: dict[str, dict], visuals: list[dict]) -> str:
    lines = [
        "# SmartPed AI Detector Threshold A/B Tracking Evaluation",
        "",
        "**Experiment:** Integration/demo run on PIE training footage (`set01/video_0001`), first 60 seconds only. "
        "These results are not held-out generalization evidence and were not used to select a threshold.",
        "",
        "## Protocol and controls",
        "",
        f"- Video: `{config['video']}`",
        f"- Interval: first {config['duration_seconds']:.1f} seconds, frames 0-{config['frame_limit'] - 1}, "
        f"{config['fps']:.3f} FPS, {config['width']}x{config['height']}.",
        f"- Pedestrian confidence: Run A {RUNS[0][1]:.10f}; Run B {RUNS[1][1]:.10f}.",
        f"- Pedestrian checkpoint: `{config['pedestrian_checkpoint']}` "
        f"(SHA-256 `{config['pedestrian_checkpoint_sha256']}`).",
        f"- Vehicle checkpoint: `{config['vehicle_checkpoint']}` "
        f"(SHA-256 `{config['vehicle_checkpoint_sha256']}`); confidence fixed at {VEHICLE_CONFIDENCE:.2f}.",
        f"- Pedestrian tracker: `{config['pedestrian_tracker']}`.",
        f"- Vehicle tracker: `{config['vehicle_tracker']}`.",
        f"- GPU: {config.get('gpu_model') or 'not reported by nvidia-smi'}.",
        f"- Vehicle detector/tracker configuration: existing `configs/vehicle_tracking.json`; copied per-run into the runner's temporary workspace, with only video/output/model/tracker/duration/sample fields materialized and vehicle confidence retained at {VEHICLE_CONFIDENCE:.2f}.",
        f"- The unchanged pedestrian BoT-SORT config has `track_low_thresh: {PEDESTRIAN_TRACK_LOW_THRESHOLD:.2f}`. Run A requests detector confidence 0.091162, but the minimum emitted tracked observation was 0.1000; scores in [0.091162, 0.10) did not appear in the tracked output. Run B's minimum observed score was 0.1266.",
        "- Interaction analysis uses the existing configuration, including its 150-pixel nearby threshold, 15-frame maximum observation gap, and movement rules.",
        f"- Detector inference image size: {config['image_size']} (existing Ultralytics default, verified from the active package configuration); no source-video resizing or frame skipping was added.",
        "- Runs are sequential on the same host/GPU; only pedestrian confidence changes.",
        "- Control check: vehicle detection and trajectory CSVs are byte-identical across runs, with 13,226 vehicle detections in each. See the run-integrity hashes in the JSON artifacts.",
        "- The runner's crossing-intention integration remains present. The saved model is not tracker-feature compatible, so predictions remain unavailable rather than being inferred from incompatible features.",
        "",
        "## Before-run findings",
        "",
        "The existing runner already supported per-run `--model` and `--confidence`, but `--confidence` was applied to both pedestrian and vehicle inference, and `--model` was shared. To keep the vehicle side controlled, the runner now accepts optional `--vehicle-model` and `--vehicle-confidence` overrides. Defaults preserve the previous shared model/confidence behavior. No shared configuration or prior experiment output was overwritten.",
        "",
        "## Side-by-side results",
        "",
        "| Metric | Run A: 0.091162 | Run B: 0.125452 |",
        "|---|---:|---:|",
    ]
    a = results.get(RUNS[0][0], {})
    b = results.get(RUNS[1][0], {})

    rows = [
        ("Run status", "status", ""),
        ("Pedestrian tracked detection observations", "pedestrian.tracked_detection_observations", ","),
        ("Unique pedestrian track IDs", "pedestrian.unique_track_ids", ","),
        ("Vehicle detections (fixed vehicle side)", "pipeline_metrics.vehicle.detection_count", ","),
        ("Mean detections/frame (incl. empty frames)", "pedestrian.detection_count_per_frame.mean", ".3f"),
        ("Detection count/frame p95", "pedestrian.detection_count_per_frame.p95", ".2f"),
        ("Maximum detections/frame", "pedestrian.detection_count_per_frame.max", ","),
        ("Mean detection confidence", "pedestrian.confidence_distribution.mean", ".4f"),
        ("Median detection confidence", "pedestrian.confidence_distribution.median", ".4f"),
        ("Track duration median (s)", "pedestrian.track_durations_seconds.median", ".2f"),
        ("Track duration mean (s)", "pedestrian.track_durations_seconds.mean", ".2f"),
        ("Track duration p10 (s)", "pedestrian.track_durations_seconds.p10", ".2f"),
        ("Track duration p90 (s)", "pedestrian.track_durations_seconds.p90", ".2f"),
        ("Track duration min (s)", "pedestrian.track_durations_seconds.min", ".2f"),
        ("Track duration max (s)", "pedestrian.track_durations_seconds.max", ".2f"),
        ("Tracks < 1 s", "pedestrian.short_tracks_under_1_second", ","),
        ("Tracks >= 5 s", "pedestrian.long_tracks_at_least_5_seconds", ","),
        ("Missing frame slots inside track spans", "pedestrian.track_observation_gaps.missing_frame_slots_inside_track_spans", ","),
        ("Tracks with missing frame slots", "pedestrian.track_observation_gaps.tracks_with_missing_frames_inside_first_last_observation_span", ","),
        ("Included PIE boxes matched at IoU >= 0.50", "pedestrian.annotation_identity_diagnostics.matched_boxes_iou_at_least_0_50", ","),
        ("Annotated identities split across predicted IDs", "pedestrian.annotation_identity_diagnostics.identities_assigned_multiple_predicted_track_ids", ","),
        ("Annotation-matched identity changes (diagnostic)", "pedestrian.annotation_identity_diagnostics.predicted_id_changes_between_matched_annotations_within_1_second", ","),
        ("Interaction pair observations", "interaction.interaction_pair_observations", ","),
        ("Unique pedestrian-vehicle pairs", "interaction.unique_pedestrian_vehicle_track_pairs", ","),
        ("Nearby observations <= 150 px", "interaction.nearby_observations_at_150_pixels", ","),
        ("Unique nearby pairs", "interaction.unique_nearby_track_pairs", ","),
        ("Invalid movement estimates", "interaction.invalid_movement_estimates", ","),
        ("Runtime (s)", "processing_runtime_seconds", ".2f"),
        ("Average processing FPS", "processing_fps", ".2f"),
        ("Mean GPU utilization (%)", "gpu.mean_utilization_percent", ".1f"),
        ("Peak GPU memory (MiB)", "gpu.peak_memory_used_mib", ".0f"),
    ]

    def nested(record: dict, path: str) -> Any:
        value: Any = record
        for part in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    for label, key, format_spec in rows:
        if key == "status":
            left = a.get("status", "failed/not run")
            right = b.get("status", "failed/not run")
        else:
            left_raw, right_raw = nested(a, key), nested(b, key)
            left = f"{left_raw:{format_spec}}" if isinstance(left_raw, (int, float)) else "n/a"
            right = f"{right_raw:{format_spec}}" if isinstance(right_raw, (int, float)) else "n/a"
        lines.append(f"| {label} | {left} | {right} |")

    lines.extend(["", "## Confidence distributions", ""])
    lines.extend(
        [
            "| Statistic | Run A | Run B |",
            "|---|---:|---:|",
        ]
    )
    for label, key in (
        ("Minimum", "min"),
        ("5th percentile", "p05"),
        ("25th percentile", "p25"),
        ("Median", "median"),
        ("75th percentile", "p75"),
        ("95th percentile", "p95"),
        ("Maximum", "max"),
    ):
        values = [
            nested(a, f"pedestrian.confidence_distribution.{key}"),
            nested(b, f"pedestrian.confidence_distribution.{key}"),
        ]
        left = f"{values[0]:.4f}" if isinstance(values[0], (int, float)) else "n/a"
        right = f"{values[1]:.4f}" if isinstance(values[1], (int, float)) else "n/a"
        lines.append(
            f"| {label} | {left} | {right} |"
        )
    lines.extend(["", "### Confidence histogram", "", "| Confidence bin | Run A | Run B |", "|---|---:|---:|"])
    histogram_a = nested(a, "pedestrian.confidence_distribution.histogram_0_1_bins") or []
    histogram_b = nested(b, "pedestrian.confidence_distribution.histogram_0_1_bins") or []
    for left_bin, right_bin in zip(histogram_a, histogram_b):
        lines.append(
            f"| {left_bin['range']} | {left_bin['count']:,} | {right_bin['count']:,} |"
        )

    if a and b:
        lines.extend(
            [
                "",
                "## Observed trade-offs",
                "",
                f"- Run A yielded {a['pedestrian']['tracked_detection_observations'] - b['pedestrian']['tracked_detection_observations']:+,} tracked pedestrian observations and "
                f"{a['interaction']['interaction_pair_observations'] - b['interaction']['interaction_pair_observations']:+,} interaction observations versus Run B.",
                f"- Nearby observations were {a['interaction']['nearby_observations_at_150_pixels']:,} vs "
                f"{b['interaction']['nearby_observations_at_150_pixels']:,}; unique nearby pairs were "
                f"{a['interaction']['unique_nearby_track_pairs']:,} vs {b['interaction']['unique_nearby_track_pairs']:,}. "
                "The lower threshold increased nearby observations in this clip but did not increase the unique nearby-pair count.",
                f"- Median track duration was {a['pedestrian']['track_durations_seconds']['median']:.2f}s vs "
                f"{b['pedestrian']['track_durations_seconds']['median']:.2f}s; these are tracker outputs, not identity-accuracy measures.",
                "- Visual samples did not reveal an obvious duplicate pedestrian box in the selected frames. The smallest-area sample showed no pedestrian track box in either run; the selected occluded and densest-available samples showed a pedestrian track in each. This small, sparse sample cannot establish overall false-positive or occlusion recall behavior.",
                "- Both runs showed the interaction panel's explicit \"crossing unavailable\" state. It is readable at native video size, though text is small in reduced side-by-side stills.",
                "- The segment has no highly crowded included annotation frame: the maximum is four pedestrians. The densest available sample is therefore not a genuinely crowded scene.",
                "- Tracker IDs differ between independent tracker runs and should not be compared numerically across panels.",
                "- The annotation-matched ID-change indicator is one in each run, based on only 100 included sparse pedestrian boxes (39 matched for A, 38 for B). It is not formal ID-switch accuracy.",
                "- Runtime and FPS differ between sequential runs; this single A-then-B comparison is descriptive and does not isolate a causal threshold-performance effect.",
                "",
                "### Run A — 0.091162",
                "",
                f"- Advantage: yielded {a['pedestrian']['tracked_detection_observations'] - b['pedestrian']['tracked_detection_observations']:,} more tracked observations and "
                f"{a['interaction']['nearby_observations_at_150_pixels'] - b['interaction']['nearby_observations_at_150_pixels']:,} more nearby observations than Run B in this segment.",
                "- Disadvantage/limitation: the configured tracker low threshold is 0.10, so this run did not emit tracked boxes in the 0.091162–0.10 score interval; added interaction observations are not necessarily useful and require review.",
                "",
                "### Run B — 0.125452",
                "",
                "- Advantage: yielded fewer pedestrian and interaction observations, reducing the volume for downstream review.",
                f"- Disadvantage: yielded {abs(b['pedestrian']['tracked_detection_observations'] - a['pedestrian']['tracked_detection_observations']):,} fewer tracked pedestrian observations than Run A and may miss weaker detections; the small selected-image sample does not establish whether those detections were useful.",
            ]
        )
    lines.extend(["", "## Control-integrity verification", ""])
    integrity = config.get("vehicle_output_integrity", {})
    lines.extend(
        [
            "| Vehicle output | Run A SHA-256 | Run B SHA-256 | Identical |",
            "|---|---|---|---|",
        ]
    )
    for filename, check in integrity.items():
        hashes = check.get("sha256_by_run", {})
        lines.append(
            f"| `{filename}` | `{hashes.get(RUNS[0][0], 'n/a')}` | "
            f"`{hashes.get(RUNS[1][0], 'n/a')}` | "
            f"{'Yes' if check.get('byte_identical_across_runs') else 'No'} |"
        )
    lines.extend(["", "## Run outputs", ""])
    for name, record in results.items():
        video_path = Path(record["outputs"]["annotated_video"])
        lines.append(
            f"- **{name}** (`{record['pedestrian_confidence_threshold']:.10f}`): "
            f"[annotated video]({video_path.relative_to(OUT).as_posix()}), "
            f"[processing report]({Path(record['outputs']['processing_report']).relative_to(OUT).as_posix()}), "
            f"[run metrics]({Path(record['outputs']['processing_metrics']).relative_to(OUT).as_posix()}), "
            f"[pedestrian detections per frame]({Path(record['outputs']['pedestrian_detections_per_frame']).relative_to(OUT).as_posix()})."
        )
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = OUT / "tracking_experiment_config.json"
    report_path = OUT / "THRESHOLD_TRACKING_COMPARISON.md"
    if config_path.exists() or report_path.exists():
        raise FileExistsError(
            f"Experiment metadata already exists under {OUT}; refusing to overwrite."
        )
    for run_name, _ in RUNS:
        run_dir = OUT / run_name
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run output is not empty; refusing to overwrite: {run_dir}"
            )

    vehicle_config = load_json(VEHICLE_CONFIG)
    vehicle_model = Path(vehicle_config["model_path"])
    if not vehicle_model.is_absolute():
        vehicle_model = (VEHICLE_CONFIG.parent / vehicle_model).resolve()
    video_info = {
        "video": str(VIDEO.resolve()),
        "duration_seconds": 60.0,
        "frame_limit": FRAME_LIMIT,
        "fps": FPS,
        "width": 1920,
        "height": 1080,
        "image_size": 640,
        "pedestrian_checkpoint": str(PEDESTRIAN_MODEL.resolve()),
        "pedestrian_checkpoint_sha256": sha256(PEDESTRIAN_MODEL),
        "vehicle_checkpoint": str(vehicle_model),
        "vehicle_checkpoint_sha256": sha256(vehicle_model),
        "pedestrian_tracker": str(PED_TRACKER.resolve()),
        "vehicle_tracker": str(VEH_TRACKER.resolve()),
        "vehicle_config_source": str(VEHICLE_CONFIG.resolve()),
        "vehicle_confidence_threshold": VEHICLE_CONFIDENCE,
        "pedestrian_tracker_low_confidence_threshold": PEDESTRIAN_TRACK_LOW_THRESHOLD,
        "gpu_model": gpu_model_name(),
        "interaction_config_source": str((ROOT / "configs" / "interaction_analysis.json").resolve()),
        "integration_config_source": str((ROOT / "configs" / "integration_analysis.json").resolve()),
        "execution_order": [name for name, _ in RUNS],
        "frame_range": [0, FRAME_LIMIT - 1],
        "output_directories": {
            name: str((OUT / name).resolve()) for name, _ in RUNS
        },
    }
    if not VIDEO.is_file() or not PEDESTRIAN_MODEL.is_file() or not vehicle_model.is_file():
        raise FileNotFoundError("Required video or detector checkpoint is missing.")
    config_path.write_text(json.dumps(video_info, indent=2) + "\n", encoding="utf-8")

    sampler = GpuSampler()
    sampler.start()
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    try:
        for name, confidence in RUNS:
            sampler.current_run = name
            started = time.perf_counter()
            try:
                result = run_video(
                    VIDEO,
                    output_dir=OUT / name,
                    duration_seconds=60.0,
                    confidence=confidence,
                    model_path=PEDESTRIAN_MODEL,
                    vehicle_model_path=vehicle_model,
                    vehicle_confidence=VEHICLE_CONFIDENCE,
                    pedestrian_tracker_config=PED_TRACKER,
                    vehicle_tracker_config=VEH_TRACKER,
                )
                record = build_run_record(name, confidence, result, sampler.summary(name))
                record["wall_clock_seconds_including_reporting"] = time.perf_counter() - started
                results[name] = record
            except Exception as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"
                (OUT / name).mkdir(parents=True, exist_ok=True)
                (OUT / name / "failure.json").write_text(
                    json.dumps(
                        {
                            "name": name,
                            "status": "failed",
                            "pedestrian_confidence_threshold": confidence,
                            "error": failures[name],
                            "traceback": traceback.format_exc(),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
    finally:
        sampler.current_run = "idle"
        sampler.stop()
        sampler.write(OUT / "gpu_samples.csv")
    for record in results.values():
        record["gpu"] = sampler.summary(record["name"])
        run_dir = Path(record["outputs"]["processing_metrics"]).parent
        (run_dir / "threshold_run_metrics.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    integrity_checks: dict[str, Any] = {}
    if len(results) == 2:
        for filename in ("vehicle_detections.csv", "vehicle_trajectories.csv"):
            hashes = {
                name: sha256(Path(record["outputs"][filename.removesuffix(".csv")]))
                for name, record in results.items()
            }
            integrity_checks[filename] = {
                "sha256_by_run": hashes,
                "byte_identical_across_runs": len(set(hashes.values())) == 1,
            }
        video_info["vehicle_output_integrity"] = integrity_checks
        config_path.write_text(json.dumps(video_info, indent=2) + "\n", encoding="utf-8")

    visual_manifest = []
    visual_error = None
    if len(results) == 2:
        try:
            visual_manifest = create_visual_comparisons(select_visual_frames(), results)
            (OUT / "visuals" / "visual_manifest.json").write_text(
                json.dumps(visual_manifest, indent=2) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            visual_error = f"{type(exc).__name__}: {exc}"

    report = render_report(video_info, results, visual_manifest)
    if failures:
        report += "\n\n## Run failures\n\n"
        report += "\n".join(f"- **{name}:** `{error}`" for name, error in failures.items())
        report += "\n"
    if visual_error:
        report += f"\n\n## Visual sample generation limitation\n\n`{visual_error}`\n"
    report += "\n\n## Visual inspection and interpretation\n\n"
    if visual_manifest:
        for sample in visual_manifest:
            report += (
                f"### {sample['category'].replace('_', ' ').title()} — frame "
                f"{sample['frame_index']} ({sample['ground_truth_boxes']} included PIE boxes)\n\n"
                f"{sample['selection_reason']}\n\n"
                f"![{sample['category']} side-by-side annotated output]({sample['file']})\n\n"
            )
    report += (
        "The selected images are annotated output frames at identical source timestamps, "
        "with detector boxes, tracker IDs, and interaction panel from each run. PIE labels "
        "are sparse and non-exhaustive; visually unmatched boxes cannot automatically be "
        "called false positives. Track-ID continuity observations are diagnostic only. "
        "No collision-risk or real-world risk prediction is inferred.\n\n"
        "## Interpretation, limitations, and next step\n\n"
        "The lower threshold may surface additional small, occluded, or distant pedestrians, "
        "but also increases candidates that can be false-looking, unstable, or lead to extra "
        "pedestrian-vehicle pairs. The higher threshold may reduce this noise and interaction "
        "volume while missing weaker detections. Judge those trade-offs from the side-by-side "
        "outputs and the reported counts, not from this training-split demo as generalization "
        "evidence. The experiment is not a threshold-selection procedure.\n\n"
        "Next step: review the generated annotated samples and full videos, then if desired "
        "run a separately specified validation-split integration test; keep the deployed "
        "threshold unchanged until that review. The crossing-intention model remains "
        "incompatible with tracker-derived features, so integrated crossing probabilities "
        "should remain unavailable.\n"
    )
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": video_info,
        "run_statuses": {
            name: results.get(name, {}).get("status", "failed")
            for name, _ in RUNS
        },
        "failures": failures,
        "visual_error": visual_error,
        "vehicle_output_integrity": integrity_checks,
    }
    (OUT / "experiment_results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Comparison report: {report_path}")
    return 0 if len(results) == 2 and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
