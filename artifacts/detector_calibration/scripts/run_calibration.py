"""Calibrate the fine-tuned YOLO11s confidence threshold on validation set06.

Both models are evaluated once with identical Ultralytics validation settings
on the validation split only. Per-image predictions are captured so every
confidence threshold in the sweep is computed from the exact same inference
pass. Test set04 is never referenced by the calibration data config.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import nms
from ultralytics.utils.metrics import box_iou

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "artifacts" / "detector_calibration"
DATA_YAML = PROJECT_ROOT / "artifacts" / "pie_detection_yolo" / "data.yaml"
PREVIOUS_REPORT = PROJECT_ROOT / "artifacts" / "detector_comparison" / "DETECTOR_COMPARISON_REPORT.md"

MODELS = {
    "pretrained_yolo11s": PROJECT_ROOT / "yolo11s.pt",
    "finetuned_train5_best": PROJECT_ROOT / "runs" / "detect" / "train-5" / "weights" / "best.pt",
}

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
BASELINE_CONF = 0.25
MAX_DET = 1000

EVAL_KWARGS = {
    "split": "val",
    "imgsz": 640,
    "batch": 16,
    "device": "0",
    "workers": 4,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": MAX_DET,
    "rect": True,
    "plots": False,
    "verbose": True,
    "seed": 0,
    "deterministic": True,
    "augment": False,
    "agnostic_nms": False,
    "single_cls": False,
    "save_json": False,
    "save_txt": False,
    "mode": "val",
    "task": "detect",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_calibration_dataset() -> tuple[Path, Path]:
    """Create a set06-only mirror outside the original dataset.

    Only ``images/validation`` is exposed (as a directory junction) and only
    ``labels/validation`` is copied. The mirror data.yaml has no train/test
    entry, so set04 cannot be selected accidentally.
    """
    tmp_root = Path(os.environ.get("TEMP", r"C:\Users\vlctus\AppData\Local\Temp"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mirror_root = tmp_root / "opencode" / "smartped_detector_calibration" / f"set06_{stamp}" / "pie_detection_yolo"
    src_root = DATA_YAML.parent
    src_images_val = src_root / "images" / "validation"
    src_labels_val = src_root / "labels" / "validation"

    if not src_images_val.is_dir():
        raise FileNotFoundError(f"Validation images directory missing: {src_images_val}")
    if not src_labels_val.is_dir():
        raise FileNotFoundError(f"Validation labels directory missing: {src_labels_val}")

    dst_images_val = mirror_root / "images" / "validation"
    dst_labels_val = mirror_root / "labels" / "validation"
    dst_images_val.parent.mkdir(parents=True, exist_ok=True)
    dst_labels_val.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(dst_images_val), str(src_images_val)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not create validation image junction for the calibration mirror: "
            f"{result.stdout} {result.stderr}"
        )
    shutil.copytree(src_labels_val, dst_labels_val)

    mirror_yaml = mirror_root / "data.yaml"
    mirror_yaml.write_text(
        "\n".join(
            [
                f'path: "{mirror_root.as_posix()}"',
                "train: images/validation",
                "val: images/validation",
                "nc: 1",
                "names:",
                "  0: pedestrian",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return mirror_yaml, mirror_root


def relative_image_path(im_file: str | Path, mirror_root: Path) -> Path:
    path = Path(im_file)
    mirror_images = mirror_root / "images"
    src_images = DATA_YAML.parent / "images"
    for root in (mirror_images, src_images):
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    raise ValueError(f"Image path is not under the calibration or source images root: {path}")


def to_original(boxes: torch.Tensor, ratio_pad, ori_shape) -> torch.Tensor:
    """Map boxes from letterboxed model-input coordinates to original pixels."""
    if boxes.numel() == 0:
        return boxes.clone().float()
    ratio, pad = ratio_pad
    if torch.is_tensor(ratio):
        ratio = ratio.tolist()
    if torch.is_tensor(pad):
        pad = pad.tolist()
    rw, rh = float(ratio[0]), float(ratio[1])
    pw, ph = float(pad[0]), float(pad[1])
    if rw <= 0 or rh <= 0:
        raise ValueError(f"Invalid letterbox ratio: {ratio_pad}")
    h, w = float(ori_shape[0]), float(ori_shape[1])
    out = boxes.clone().float()
    out[:, [0, 2]] = (out[:, [0, 2]] - pw) / rw
    out[:, [1, 3]] = (out[:, [1, 3]] - ph) / rh
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, max(w - 1, 0))
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, max(h - 1, 0))
    return out


def label_pixel_boxes(label_path: Path, width: float, height: float) -> list[list[float]]:
    boxes: list[list[float]] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = (float(v) for v in parts[:5])
        boxes.append(
            [
                (xc - bw / 2.0) * width,
                (yc - bh / 2.0) * height,
                (xc + bw / 2.0) * width,
                (yc + bh / 2.0) * height,
            ]
        )
    return boxes


class CalibrationValidator(DetectionValidator):
    """DetectionValidator restricted to class 0 that captures per-image boxes.

    The standard Ultralytics metrics are still computed by ``super()``. The
    capture stores boxes in original-image pixel coordinates, converted with
    the same ``ratio_pad`` used for validation matching.
    """

    def __init__(self, *args, model_key: str = "", mirror_root: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_key = model_key
        self.mirror_root = mirror_root
        self.ped_images: list[dict] = []
        self.ped_sanity_checked = 0
        self.ped_sanity_max_diff = 0.0

    def init_metrics(self, model):
        super().init_metrics(model)
        self.names = {0: "pedestrian"}
        self.nc = 1
        self.metrics.names = self.names

    def postprocess(self, preds):
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            classes=[0],
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [
            {
                "bboxes": x[:, :4],
                "conf": x[:, 4],
                "cls": x[:, 5],
                "extra": x[:, 6:],
            }
            for x in outputs
        ]

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            pbox = predn["bboxes"].detach().cpu()
            pconf = predn["conf"].detach().cpu()
            gbox = pbatch["bboxes"].detach().cpu()

            if pconf.numel():
                order = torch.argsort(pconf, descending=True)
                pbox = pbox[order]
                pconf = pconf[order]

            ori_h, ori_w = float(pbatch["ori_shape"][0]), float(pbatch["ori_shape"][1])
            gbox_orig = to_original(gbox, pbatch["ratio_pad"], pbatch["ori_shape"])
            pbox_orig = to_original(pbox, pbatch["ratio_pad"], pbatch["ori_shape"])

            rel = relative_image_path(pbatch["im_file"], self.mirror_root)
            source_image = DATA_YAML.parent / "images" / rel
            label_path = DATA_YAML.parent / "labels" / rel.with_suffix(".txt")

            if self.ped_sanity_checked < 5:
                expected = label_pixel_boxes(label_path, ori_w, ori_h)
                got = gbox_orig.tolist()
                if len(expected) != len(got):
                    raise AssertionError(
                        f"GT sanity check failed for {rel}: expected {len(expected)} boxes, got {len(got)}"
                    )
                if expected:
                    diff = float(np.abs(np.asarray(expected) - np.asarray(got)).max())
                    self.ped_sanity_max_diff = max(self.ped_sanity_max_diff, diff)
                    if diff > 2.0:
                        raise AssertionError(
                            f"GT coordinate sanity check failed for {rel}: max diff {diff:.3f} px"
                        )
                self.ped_sanity_checked += 1

            self.ped_images.append(
                {
                    "image": rel.as_posix(),
                    "source_image": str(source_image),
                    "width": int(ori_w),
                    "height": int(ori_h),
                    "gt": gbox_orig.tolist(),
                    "pred": (
                        torch.cat([pbox_orig, pconf[:, None]], dim=1).tolist()
                        if pbox_orig.numel()
                        else []
                    ),
                }
            )
        super().update_metrics(preds, batch)


def match_counts_for_image(validator: DetectionValidator, gt: list, pred: list, threshold: float) -> tuple[int, int, int, int]:
    """Return tp, fp, fn, predicted_count at a confidence threshold and IoU 0.50."""
    kept = [p for p in pred if float(p[4]) >= threshold]
    n_pred = len(kept)
    n_gt = len(gt)
    if n_pred == 0:
        return 0, 0, n_gt, 0
    if n_gt == 0:
        return 0, n_pred, 0, n_pred

    kept.sort(key=lambda p: float(p[4]), reverse=True)
    gbox = torch.tensor([b[:4] for b in gt], dtype=torch.float32)
    pbox = torch.tensor([b[:4] for b in kept], dtype=torch.float32)
    iou = box_iou(gbox, pbox)
    pcls = torch.zeros(n_pred, dtype=torch.float32)
    gcls = torch.zeros(n_gt, dtype=torch.float32)
    correct = validator.match_predictions(pcls, gcls, iou)
    tp = int(correct[:, 0].sum())
    return tp, n_pred - tp, n_gt - tp, n_pred


def compute_sweep(validator: DetectionValidator, images: list[dict], thresholds: list[float]) -> list[dict]:
    rows: list[dict] = []
    for threshold in thresholds:
        tp = fp = fn = predicted_boxes = images_with_predictions = 0
        for rec in images:
            t, f, n, n_pred = match_counts_for_image(validator, rec["gt"], rec["pred"], threshold)
            tp += t
            fp += f
            fn += n
            predicted_boxes += n_pred
            images_with_predictions += int(n_pred > 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted_boxes": int(predicted_boxes),
                "images_with_predictions": int(images_with_predictions),
            }
        )
    return rows


def standard_metrics(validator: DetectionValidator, images: list[dict]) -> dict:
    m = validator.metrics
    speed = {k: float(v) for k, v in dict(m.speed).items()}
    ap50 = float(m.box.ap50[0]) if np.asarray(m.box.ap50).size else None
    ap50_95 = float(m.box.ap[0]) if np.asarray(m.box.ap).size else None
    return {
        "results_dict": {k: float(v) for k, v in m.results_dict.items()},
        "ap50": ap50,
        "ap50_95": ap50_95,
        "speed_ms_per_image": speed,
        "images_evaluated": len(images),
        "ground_truth_boxes": int(np.asarray(m.nt_per_class).sum()),
        "gt_boxes_per_class": np.asarray(m.nt_per_class).tolist(),
        "names": {int(k): v for k, v in m.names.items()},
    }


def write_predictions(path: Path, images: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in images:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def main() -> int:
    (OUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "configs").mkdir(parents=True, exist_ok=True)

    readiness_path = OUT_DIR / "configs" / "readiness.json"
    if not readiness_path.is_file():
        raise FileNotFoundError(f"Run Phase 1 readiness first: {readiness_path}")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if not readiness.get("compatibility", {}).get("ok"):
        raise RuntimeError(f"Phase 1 readiness did not pass: {readiness.get('compatibility')}")

    expected_images = int(readiness["validation_dataset"]["image_count"])
    expected_boxes = int(readiness["validation_dataset"]["ground_truth_boxes"])

    mirror_yaml, mirror_root = prepare_calibration_dataset()
    eval_kwargs = {**EVAL_KWARGS, "data": str(mirror_yaml)}

    config_record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "validation split set06 only; test set04 is not used",
        "original_dataset_yaml": str(DATA_YAML),
        "calibration_dataset_yaml": str(mirror_yaml),
        "calibration_dataset_mirror": {
            "root": str(mirror_root),
            "images": "junction to the original images/validation directory (set06 only)",
            "labels": "copied labels/validation (set06 only)",
            "train_key": "aliases images/validation because Ultralytics requires train and val",
            "test_split_present": False,
        },
        "split": "val (set06 only)",
        "thresholds": THRESHOLDS,
        "baseline_conf": BASELINE_CONF,
        "max_det": MAX_DET,
        "evaluation_kwargs": eval_kwargs,
        "matching": {
            "method": "Ultralytics DetectionValidator.match_predictions greedy matching",
            "iou": 0.50,
            "class_restriction": "NMS restricted to class index 0 (pedestrian) for both models",
            "coordinate_space": "original image pixels after conversion with validation ratio_pad",
        },
        "sweep_method": (
            "One inference pass per model at conf=0.001, max_det=1000; each threshold is "
            "computed by filtering the same stored per-image predictions."
        ),
        "models": {
            key: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for key, path in MODELS.items()
        },
        "previous_comparison_report": str(PREVIOUS_REPORT),
        "readiness_config": str(readiness_path),
        "per_image_predictions": "metrics/per_image_predictions_<model>.jsonl.gz",
    }
    (OUT_DIR / "configs" / "calibration_config.json").write_text(
        json.dumps(config_record, indent=2), encoding="utf-8"
    )

    combined: dict = {
        "generated_utc": config_record["generated_utc"],
        "config": config_record,
        "validation_dataset": readiness["validation_dataset"],
        "models": {},
    }

    for key, path in MODELS.items():
        log_path = OUT_DIR / "logs" / f"calibration_{key}.log"
        started = time.time()
        record: dict = {
            "model_key": key,
            "model_path": str(path),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        lines = [f"Calibrating {key}: {path}", f"Dataset: {mirror_yaml} (val/set06 only)"]
        try:
            model = YOLO(str(path))
            args = {
                **eval_kwargs,
                "model": str(path),
                "project": str(OUT_DIR / "runs"),
                "name": f"calibration_{key}_set06",
                "exist_ok": False,
            }
            validator = CalibrationValidator(
                args=args,
                _callbacks=model.callbacks,
                model_key=key,
                mirror_root=mirror_root,
            )
            validator(model=model.model)

            images = validator.ped_images
            if len(images) != expected_images:
                raise AssertionError(
                    f"{key}: captured {len(images)} images, expected {expected_images}"
                )
            gt_boxes = sum(len(rec["gt"]) for rec in images)
            if gt_boxes != expected_boxes:
                raise AssertionError(
                    f"{key}: captured {gt_boxes} GT boxes, expected {expected_boxes}"
                )
            bad_paths = [rec["image"] for rec in images if "set04" in rec["image"] or "/test/" in rec["image"]]
            if bad_paths:
                raise AssertionError(f"{key}: test/set04 images appeared in calibration: {bad_paths[:5]}")

            sweep = compute_sweep(validator, images, THRESHOLDS)
            baseline = next(row for row in sweep if abs(row["threshold"] - BASELINE_CONF) < 1e-9)
            per_image_path = OUT_DIR / "metrics" / f"per_image_predictions_{key}.jsonl.gz"
            write_predictions(per_image_path, images)
            order_hash = hashlib.sha256(
                json.dumps([rec["image"] for rec in images]).encode("utf-8")
            ).hexdigest()

            try:
                standard = standard_metrics(validator, images)
                standard_error = None
            except Exception as exc:  # noqa: BLE001 - supplementary metrics must not lose the sweep
                standard = None
                standard_error = f"{type(exc).__name__}: {exc}"

            record.update(
                {
                    "standard_metrics": standard,
                    "standard_metrics_error": standard_error,
                    "threshold_sweep": sweep,
                    "baseline_conf_0.25": baseline,
                    "images_captured": len(images),
                    "ground_truth_boxes_captured": gt_boxes,
                    "image_order_sha256": order_hash,
                    "gt_sanity_checked_images": validator.ped_sanity_checked,
                    "gt_sanity_max_abs_diff_px": validator.ped_sanity_max_diff,
                    "per_image_predictions_file": str(per_image_path),
                    "validator_save_dir": str(validator.save_dir),
                    "status": "success",
                }
            )
            lines += [
                f"status=success images={len(images)} gt_boxes={gt_boxes}",
                f"image_order_sha256={order_hash}",
                f"gt_sanity_max_abs_diff_px={validator.ped_sanity_max_diff:.6f}",
                f"standard_metrics={standard['results_dict'] if standard else standard_error}",
                f"baseline_conf_0.25={baseline}",
                "threshold_sweep:",
            ]
            for row in sweep:
                lines.append(
                    f"  t={row['threshold']:.2f} tp={row['tp']} fp={row['fp']} fn={row['fn']} "
                    f"P={row['precision']:.6f} R={row['recall']:.6f} F1={row['f1']:.6f} "
                    f"pred={row['predicted_boxes']}"
                )
            del model, validator
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - report exact error, do not substitute results
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            lines.append(f"status=failed error={record['error']}")
            lines.append(record["traceback"])
        record["wall_clock_seconds"] = round(time.time() - started, 3)
        combined["models"][key] = record
        (OUT_DIR / "metrics" / f"calibration_{key}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))

    models = combined["models"]
    if all(r["status"] == "success" for r in models.values()):
        pre = models["pretrained_yolo11s"]
        fine = models["finetuned_train5_best"]
        if pre["image_order_sha256"] != fine["image_order_sha256"]:
            raise AssertionError("Pretrained and fine-tuned models saw different image orders.")
        if pre["ground_truth_boxes_captured"] != fine["ground_truth_boxes_captured"]:
            raise AssertionError("Pretrained and fine-tuned GT box counts differ.")
        combined["integrity"] = {
            "same_image_order": True,
            "same_gt_boxes": True,
            "image_order_sha256": pre["image_order_sha256"],
            "gt_boxes": pre["ground_truth_boxes_captured"],
            "test_set04_used": False,
        }
    else:
        combined["integrity"] = {"same_image_order": False, "test_set04_used": False}

    (OUT_DIR / "metrics" / "calibration_results.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    return 0 if all(r["status"] == "success" for r in models.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
