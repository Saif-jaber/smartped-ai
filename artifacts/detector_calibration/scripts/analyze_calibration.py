"""Refine set06 confidence calibration from the saved inference predictions.

This analysis does not run inference or modify checkpoints. It writes refined
metrics, validation-only visual samples, and the calibration report under
artifacts/detector_calibration/.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import box_iou

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "detector_calibration"
METRICS = OUT / "metrics"
VISUALS = OUT / "visuals"
DATASET = ROOT / "artifacts" / "pie_detection_yolo"
FINE_KEY = "finetuned_train5_best"
PRE_KEY = "pretrained_yolo11s"
BASELINE_CONF = 0.25
MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.50


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_matches(rec: dict) -> np.ndarray:
    """Return each stored prediction's IoU=0.50 greedy-match status."""
    predictions = rec["pred"]
    ground_truth = rec["gt"]
    if not predictions or not ground_truth:
        return np.zeros(len(predictions), dtype=bool)

    pred_boxes = torch.tensor([box[:4] for box in predictions], dtype=torch.float32)
    true_boxes = torch.tensor([box[:4] for box in ground_truth], dtype=torch.float32)
    pred_classes = torch.zeros(len(predictions), dtype=torch.float32)
    true_classes = torch.zeros(len(ground_truth), dtype=torch.float32)
    iou = box_iou(true_boxes, pred_boxes)
    matcher = type(
        "Matcher",
        (),
        {"iouv": torch.linspace(0.50, 0.95, 10)},
    )()
    correct = DetectionValidator.match_predictions(
        matcher, pred_classes, true_classes, iou
    )
    return correct[:, 0].cpu().numpy()


def score_metrics(scores: np.ndarray, true_positives: np.ndarray, threshold: float) -> dict:
    kept = scores >= threshold
    predicted = int(kept.sum())
    tp = int(true_positives[kept].sum())
    fp = predicted - tp
    fn = TOTAL_GT - tp
    precision = tp / predicted if predicted else 0.0
    recall = tp / TOTAL_GT
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_boxes": predicted,
    }


def confidence_candidates(scores: np.ndarray, true_positives: np.ndarray) -> list[dict]:
    in_range = scores >= MIN_THRESHOLD
    scores = scores[in_range]
    true_positives = true_positives[in_range]
    if not scores.size:
        return []

    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tp_counts = np.bincount(inverse, weights=true_positives.astype(np.int64), minlength=len(values))
    rows = []
    cumulative_pred = 0
    cumulative_tp = 0
    for index in range(len(values) - 1, -1, -1):
        cumulative_pred += int(counts[index])
        cumulative_tp += int(tp_counts[index])
        threshold = float(values[index])
        if threshold > MAX_THRESHOLD:
            continue
        precision = cumulative_tp / cumulative_pred
        rows.append(
            {
                "threshold": threshold,
                "tp": cumulative_tp,
                "fp": cumulative_pred - cumulative_tp,
                "fn": TOTAL_GT - cumulative_tp,
                "predicted_boxes": cumulative_pred,
                "precision": precision,
                "recall": cumulative_tp / TOTAL_GT,
                "f1": 2 * cumulative_tp / (cumulative_pred + TOTAL_GT),
            }
        )
    for endpoint in (MIN_THRESHOLD, MAX_THRESHOLD):
        if not any(row["threshold"] == endpoint for row in rows):
            rows.append(score_metrics(scores, true_positives, endpoint))
    rows.sort(key=lambda row: row["threshold"])
    return rows


def group_visual_metadata(prediction_images: list[dict]) -> dict[str, dict]:
    metadata_path = DATASET / "annotation_metadata.jsonl.gz"
    grouped: dict[str, dict] = defaultdict(
        lambda: {"areas": [], "boxes": [], "occluded": 0}
    )
    with gzip.open(metadata_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if (
                item.get("split") != "validation"
                or item.get("set_id") != "set06"
                or not item.get("label_included")
            ):
                continue
            image = f"validation/set06/{item['video_id']}/frame_{int(item['frame_index']):06d}.jpg"
            box = item.get("yolo_xywh")
            if not box:
                continue
            target = grouped[image]
            target["boxes"].append(box)
            target["areas"].append(float(box[2]) * float(box[3]))
            target["occluded"] += int(
                item.get("occlusion") in {"part", "full"} or item.get("occluded") is True
            )

    known_images = {rec["image"] for rec in prediction_images}
    return {image: record for image, record in grouped.items() if image in known_images}


def choose_visual_samples(prediction_images: list[dict]) -> list[dict]:
    grouped = group_visual_metadata(prediction_images)
    if not grouped:
        raise RuntimeError("No set06 visual metadata matched the saved predictions.")

    areas = np.asarray([area for row in grouped.values() for area in row["areas"]])
    small_limit = float(np.percentile(areas, 25))
    rows = []
    for image, record in grouped.items():
        rows.append(
            {
                "image": image,
                "n_boxes": len(record["boxes"]),
                "median_area": float(np.median(record["areas"])),
                "occluded": int(record["occluded"]),
            }
        )

    categories = {
        "small_or_distant": sorted(
            (row for row in rows if row["n_boxes"] >= 2 and row["median_area"] <= small_limit),
            key=lambda row: (row["median_area"], -row["n_boxes"], row["image"]),
        ),
        "partially_occluded": sorted(
            (row for row in rows if row["n_boxes"] >= 2 and row["occluded"] > 0),
            key=lambda row: (-row["occluded"], -row["n_boxes"], row["image"]),
        ),
        "crowded": sorted(
            (row for row in rows if row["n_boxes"] >= 9),
            key=lambda row: (-row["n_boxes"], -row["median_area"], row["image"]),
        ),
    }

    reasons = {
        "small_or_distant": f"At least two GT boxes; median normalized box area <= set06 25th percentile ({small_limit:.6g}).",
        "partially_occluded": "At least two GT boxes and >=1 PIE partial/full occlusion annotation.",
        "crowded": "At least nine GT pedestrian boxes in the frame.",
    }
    used: set[str] = set()
    selected = []
    for category, candidates in categories.items():
        candidate = next((row for row in candidates if row["image"] not in used), None)
        if candidate is None:
            raise RuntimeError(f"No unused set06 candidate found for {category}.")
        used.add(candidate["image"])
        selected.append({**candidate, "category": category, "selection_reason": reasons[category]})
    return selected


def draw_panel(image: Image.Image, title: str, boxes: list, color: tuple[int, int, int]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width = max(3, canvas.width // 480)
    font = ImageFont.load_default()
    for box in boxes:
        x1, y1, x2, y2 = (float(value) for value in box[:4])
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        if len(box) > 4:
            label = f"{float(box[4]):.2f}"
            draw.text((x1, max(0, y1 - 14)), label, fill=color, font=font)
    return canvas


def render_visuals(
    selected_samples: list[dict],
    fine_by_image: dict[str, dict],
    pre_by_image: dict[str, dict],
    selected_threshold: float,
) -> list[dict]:
    VISUALS.mkdir(parents=True, exist_ok=True)
    outputs = []
    panel_size = (1280, 720)
    header_height = 30
    for sample in selected_samples:
        image_key = sample["image"]
        fine = fine_by_image[image_key]
        pre = pre_by_image[image_key]
        source = Path(fine["source_image"])
        if not source.is_file():
            raise FileNotFoundError(f"Validation image is missing: {source}")
        original = Image.open(source).convert("RGB")
        fine_selected = [box for box in fine["pred"] if float(box[4]) >= selected_threshold]
        panels = [
            ("Ground truth", fine["gt"], (40, 190, 80)),
            ("Pretrained @ 0.25", [box for box in pre["pred"] if float(box[4]) >= BASELINE_CONF], (40, 120, 245)),
            ("Fine-tuned @ 0.25", [box for box in fine["pred"] if float(box[4]) >= BASELINE_CONF], (245, 160, 35)),
            (f"Fine-tuned @ {selected_threshold:.6f}", fine_selected, (235, 55, 55)),
        ]
        montage = Image.new("RGB", (panel_size[0] * 2, (panel_size[1] + header_height) * 2), (25, 25, 25))
        draw = ImageDraw.Draw(montage)
        for index, (title, boxes, color) in enumerate(panels):
            panel = draw_panel(original, title, boxes, color)
            panel.thumbnail(panel_size, Image.Resampling.LANCZOS)
            x = (index % 2) * panel_size[0]
            y = (index // 2) * (panel_size[1] + header_height)
            draw.rectangle((x, y, x + panel_size[0], y + header_height), fill=(25, 25, 25))
            draw.text((x + 8, y + 8), title, fill=(255, 255, 255), font=ImageFont.load_default())
            montage.paste(panel, (x, y + header_height))

        filename = f"set06_{sample['category']}_{Path(image_key).stem}_comparison.png"
        destination = VISUALS / filename
        montage.save(destination, optimize=True)
        outputs.append(
            {
                **sample,
                "file": f"visuals/{filename}",
                "dimensions": list(original.size),
                "ground_truth_boxes": len(fine["gt"]),
                "pretrained_predictions_at_0.25": sum(
                    float(box[4]) >= BASELINE_CONF for box in pre["pred"]
                ),
                "finetuned_predictions_at_0.25": sum(
                    float(box[4]) >= BASELINE_CONF for box in fine["pred"]
                ),
                "finetuned_predictions_at_selected_threshold": len(fine_selected),
            }
        )
    return outputs


def main() -> int:
    global TOTAL_GT
    config = read_json(OUT / "configs" / "calibration_config.json")
    readiness = read_json(OUT / "configs" / "readiness.json")
    original_results = read_json(METRICS / "calibration_results.json")
    fine_raw = read_json(METRICS / f"calibration_{FINE_KEY}.json")
    pre_raw = read_json(METRICS / f"calibration_{PRE_KEY}.json")
    if fine_raw["status"] != "success" or pre_raw["status"] != "success":
        raise RuntimeError("Both original validation calibrations must have succeeded.")
    if not readiness["validation_split"]["only_set06"]:
        raise RuntimeError("Readiness metadata does not confirm validation set06 only.")
    if original_results.get("integrity", {}).get("test_set04_used") is not False:
        raise RuntimeError("Original calibration integrity metadata does not exclude set04.")

    for key in (PRE_KEY, FINE_KEY):
        checkpoint = Path(config["models"][key]["path"])
        actual_hash = sha256(checkpoint)
        if actual_hash != config["models"][key]["sha256"]:
            raise RuntimeError(f"Checkpoint hash changed since calibration: {checkpoint}")

    fine_images = read_jsonl_gz(METRICS / f"per_image_predictions_{FINE_KEY}.jsonl.gz")
    pre_images = read_jsonl_gz(METRICS / f"per_image_predictions_{PRE_KEY}.jsonl.gz")
    if len(fine_images) != 3911 or len(pre_images) != len(fine_images):
        raise RuntimeError("Saved predictions do not match the recorded set06 image count.")
    if [row["image"] for row in fine_images] != [row["image"] for row in pre_images]:
        raise RuntimeError("The saved prediction files do not have matching image order.")
    TOTAL_GT = sum(len(row["gt"]) for row in fine_images)
    if TOTAL_GT != 11200 or any(len(row["gt"]) != len(other["gt"]) for row, other in zip(fine_images, pre_images)):
        raise RuntimeError("Saved validation labels do not match the expected 11,200 boxes.")

    all_scores = []
    all_tp = []
    image_max_scores = []
    for rec in fine_images:
        matched = count_matches(rec)
        scores = np.asarray([float(box[4]) for box in rec["pred"]], dtype=np.float64)
        if len(scores) != len(matched):
            raise AssertionError(f"Matcher output size differs for {rec['image']}.")
        all_scores.append(scores)
        all_tp.append(matched)
        image_max_scores.append(float(scores.max()) if scores.size else -1.0)
    scores = np.concatenate(all_scores)
    true_positives = np.concatenate(all_tp)
    candidates = confidence_candidates(scores, true_positives)
    if not candidates:
        raise RuntimeError("No stored predictions fall in the configured threshold range.")

    baseline = pre_raw["baseline_conf_0.25"]
    target_recall = float(baseline["recall"])
    eligible = [row for row in candidates if row["recall"] + 1e-12 >= target_recall]
    selected = (
        max(eligible, key=lambda row: (row["precision"], row["threshold"], row["recall"]))
        if eligible
        else None
    )
    best_f1 = max(candidates, key=lambda row: (row["f1"], row["threshold"]))
    best_balance = min(
        candidates,
        key=lambda row: (abs(row["precision"] - row["recall"]), -row["f1"], -row["threshold"]),
    )
    selected_threshold = selected["threshold"] if selected else None

    grid = [round(0.05 + step / 1000, 3) for step in range(451)]
    metric_rows = []
    for threshold in grid:
        row = score_metrics(scores, true_positives, threshold)
        row["images_with_predictions"] = sum(max_score >= threshold for max_score in image_max_scores)
        row["threshold_type"] = "0.001_grid"
        metric_rows.append(row)
    if selected and not any(abs(row["threshold"] - selected_threshold) < 1e-12 for row in metric_rows):
        row = dict(selected)
        row["images_with_predictions"] = sum(max_score >= selected_threshold for max_score in image_max_scores)
        row["threshold_type"] = "selected_exact"
        metric_rows.append(row)
    metric_rows.sort(key=lambda row: row["threshold"])

    original_fine_rows = {round(row["threshold"], 2): row for row in fine_raw["threshold_sweep"]}
    for threshold, expected in original_fine_rows.items():
        actual = score_metrics(scores, true_positives, threshold)
        for key in ("tp", "fp", "fn", "predicted_boxes"):
            if actual[key] != expected[key]:
                raise AssertionError(
                    f"Refined {key} at {threshold:.2f} ({actual[key]}) differs from original ({expected[key]})."
                )

    METRICS.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS / "threshold_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "threshold",
            "threshold_type",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "predicted_boxes",
            "images_with_predictions",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    sample_data = choose_visual_samples(fine_images)
    fine_by_image = {row["image"]: row for row in fine_images}
    pre_by_image = {row["image"]: row for row in pre_images}
    visuals = (
        render_visuals(sample_data, fine_by_image, pre_by_image, selected_threshold)
        if selected_threshold is not None
        else []
    )

    integrity = {
        "images_evaluated": len(fine_images),
        "ground_truth_boxes": TOTAL_GT,
        "split": "set06 validation only",
        "test_set04_used": False,
        "same_image_order": True,
        "coarse_metrics_match_original_sweep": True,
        "pretrained_checkpoint_sha256": config["models"][PRE_KEY]["sha256"],
        "finetuned_checkpoint_sha256": config["models"][FINE_KEY]["sha256"],
    }
    refined = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "integrity": integrity,
        "baseline": {
            "model": PRE_KEY,
            "confidence": BASELINE_CONF,
            "tp": baseline["tp"],
            "fp": baseline["fp"],
            "fn": baseline["fn"],
            "precision": baseline["precision"],
            "recall": baseline["recall"],
            "f1": baseline["f1"],
            "predicted_boxes": baseline["predicted_boxes"],
        },
        "fine_tuned_at_0_25": fine_raw["baseline_conf_0.25"],
        "selection_policy": (
            "Scan every distinct stored confidence score in [0.05, 0.50] after reproducing "
            "Ultralytics greedy class-0 matching. Select the greatest precision among points "
            "whose recall is at least the pretrained set06 recall; ties prefer the higher threshold."
        ),
        "selected_threshold": selected,
        "target_recall": target_recall,
        "target_recall_achieved": selected is not None,
        "best_f1_threshold": best_f1,
        "best_precision_recall_balance": {
            **best_balance,
            "definition": "Smallest absolute difference between precision and recall; ties prefer higher F1 then higher threshold.",
        },
        "refinement": {
            "grid_min": MIN_THRESHOLD,
            "grid_max": MAX_THRESHOLD,
            "grid_step": 0.001,
            "distinct_confidence_candidates_scanned": len(candidates),
            "threshold_metrics_csv": "metrics/threshold_metrics.csv",
            "prediction_source": "Existing saved per-image predictions; no re-inference.",
        },
        "visual_samples": visuals,
    }
    refined_path = METRICS / "refined_calibration_results.json"
    refined_path.write_text(json.dumps(refined, indent=2), encoding="utf-8")
    (OUT / "configs" / "refined_calibration_config.json").write_text(
        json.dumps(
            {
                "scope": "set06 validation only",
                "inference_performed": False,
                "matching": "Ultralytics DetectionValidator.match_predictions greedy matching at IoU 0.50",
                "confidence_range": [MIN_THRESHOLD, MAX_THRESHOLD],
                "grid_step": 0.001,
                "selection_rule": refined["selection_policy"],
                "checkpoint_hashes": {
                    key: config["models"][key]["sha256"] for key in (PRE_KEY, FINE_KEY)
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (VISUALS / "visual_manifest.json").write_text(json.dumps(visuals, indent=2), encoding="utf-8")

    report_path = OUT / "DETECTOR_CALIBRATION_REPORT.md"
    if selected:
        selected_line = (
            f"Selected threshold: **{selected_threshold:.6f}** "
            f"(precision {selected['precision']:.4f}, recall {selected['recall']:.4f}, "
            f"F1 {selected['f1']:.4f}, TP {selected['tp']:,}, FP {selected['fp']:,}, "
            f"FN {selected['fn']:,}, predicted boxes {selected['predicted_boxes']:,})."
        )
    else:
        selected_line = (
            "No threshold in the evaluated range [0.05, 0.50] achieved the target recall; "
            "no threshold is recommended."
        )

    def row_line(label: str, row: dict) -> str:
        return (
            f"| {label} | {row['threshold']:.6f} | {row['precision']:.4f} | {row['recall']:.4f} "
            f"| {row['f1']:.4f} | {row['tp']:,} | {row['fp']:,} | {row['fn']:,} "
            f"| {row['predicted_boxes']:,} |"
        )

    compared_rows = [
        row_line("Pretrained baseline @ 0.25", baseline),
        row_line("Fine-tuned @ 0.25", fine_raw["baseline_conf_0.25"]),
    ]
    if selected:
        compared_rows.append(row_line("Selected fine-tuned operating point", selected))
    compared_rows += [
        row_line("Fine-tuned highest F1", best_f1),
        row_line("Fine-tuned closest P/R parity", best_balance),
    ]
    visual_lines = []
    for sample in visuals:
        visual_lines.append(
            f"### {sample['category'].replace('_', ' ').title()}\n\n"
            f"{sample['selection_reason']} Image: `{sample['image']}` "
            f"({sample['ground_truth_boxes']} GT boxes).\n\n"
            f"![{sample['category']} set06 validation comparison]({sample['file']})\n"
        )

    environment = readiness.get("environment", {})
    environment_rows = [
        ("Python", environment.get("python")),
        ("PyTorch", environment.get("torch")),
        ("Ultralytics", environment.get("ultralytics")),
        (
            "CUDA",
            f"{environment.get('cuda_version')} (available={environment.get('cuda_available')})"
            if environment.get("cuda_version")
            else None,
        ),
        ("GPU", ", ".join(environment.get("gpu_names", [])) or None),
    ]
    env_text = "\n".join(f"| {key} | {value or 'Not recorded'} |" for key, value in environment_rows)
    precision_delta_pp = (selected["precision"] - baseline["precision"]) * 100 if selected else None
    fp_reduction_pct = (
        (baseline["fp"] - selected["fp"]) / baseline["fp"] * 100 if selected else None
    )
    report = f"""# Detector Confidence Calibration Report

**Scope:** PIE validation set06 only. The held-out test split set04 was not used.  
**Purpose:** Calibration experiment only; no detector configuration or video runner was changed.

## Result

Pretrained YOLO11s at confidence 0.25 sets the validation recall target at
**{target_recall:.4f}** ({baseline['tp']:,}/{TOTAL_GT:,} pedestrians).

{selected_line}

{"The selected point meets the target recall and improves precision by " + f"{precision_delta_pp:.2f} percentage points" + f", with {fp_reduction_pct:.1f}% fewer false positives than the pretrained baseline at 0.25." if selected else "The recall target was not met in the evaluated range."}

The selection scans every distinct stored fine-tuned confidence score from 0.05 through 0.50; it is not limited to the 0.001 reporting grid. At each score, detections at or above the score are matched in descending confidence order using Ultralytics greedy matching at IoU 0.50. The CSV includes a 0.001 grid and the exact selected threshold when it falls between grid points.

## Validation data and readiness

| Item | Value |
|---|---|
| Split | set06 validation |
| Images evaluated | {len(fine_images):,} |
| Annotated pedestrian boxes | {TOTAL_GT:,} |
| Image size | 640 |
| Inference NMS IoU / confidence floor / max detections | 0.70 / 0.001 / 1,000 |
| Batch / device | 16 / GPU 0 |
| Class restriction | class 0 (pedestrian) |
| Matching IoU | 0.50 |
| Pretrained checkpoint SHA-256 | `{config['models'][PRE_KEY]['sha256']}` |
| Fine-tuned checkpoint SHA-256 | `{config['models'][FINE_KEY]['sha256']}` |
| Same image order / labels | Verified |
| Test set04 used | No |

### Environment recorded during readiness

| Component | Version / device |
|---|---|
{env_text}

## Operating-point comparison

| Operating point | Threshold | Precision | Recall | F1 | TP | FP | FN | Predicted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(compared_rows)}

### Threshold selection details

- **Primary selection:** highest precision among thresholds with recall at least {target_recall:.4f}.
- **Highest F1:** {best_f1['threshold']:.6f} (P={best_f1['precision']:.4f}, R={best_f1['recall']:.4f}, F1={best_f1['f1']:.4f}).
- **Closest precision/recall parity:** {best_balance['threshold']:.6f} (P={best_balance['precision']:.4f}, R={best_balance['recall']:.4f}); defined as minimum absolute difference between precision and recall, with ties preferring higher F1 then higher threshold.
- **Target achieved:** {"Yes" if selected else "No"}.

## Visual verification

Each validation-only montage shows ground truth, pretrained @ 0.25, fine-tuned @ 0.25, and fine-tuned at the selected threshold.

{chr(10).join(visual_lines)}

## Artifacts

- [Threshold metrics CSV](metrics/threshold_metrics.csv): 0.001 confidence grid from 0.05 to 0.50, plus the exact selected point if needed.
- [Refined calibration JSON](metrics/refined_calibration_results.json): selected point, alternate operating points, integrity checks, and visual manifest.
- [Original raw evaluation JSON](metrics/calibration_results.json) and per-image predictions in `metrics/`.
- [Original calibration configuration](configs/calibration_config.json); [refinement configuration](configs/refined_calibration_config.json).
- Validation visual samples and [visual manifest](visuals/visual_manifest.json).

## Interpretation and limitations

This is a set06 validation calibration result, not an unbiased test estimate. The refined sweep reuses the saved per-image predictions and does not rerun inference, train, fine-tune, or modify either checkpoint. The report does not deploy the selected threshold. Review the validation visual examples and downstream behavior before considering any pipeline change; do not use set04 for threshold selection.
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"Selected threshold: {selected_threshold}")
    print(f"Selected metrics: {json.dumps(selected, sort_keys=True)}")
    print(f"Best F1 threshold: {best_f1['threshold']}")
    print(f"Best P/R parity threshold: {best_balance['threshold']}")
    print(f"Wrote {report_path}")
    return 0


TOTAL_GT = 0


if __name__ == "__main__":
    raise SystemExit(main())
