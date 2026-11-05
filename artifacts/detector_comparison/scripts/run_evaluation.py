"""Phase 2 controlled pedestrian-only evaluation on PIE test split (set04).

Both models are evaluated with identical Ultralytics validation settings.
Predictions are restricted to class index 0 (pretrained: COCO person;
fine-tuned: pedestrian) before NMS, so non-pedestrian COCO classes cannot
enter the metrics or crowd the per-image max_det budget.

Writes only under artifacts/detector_comparison/.
"""

from __future__ import annotations

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
from ultralytics.utils.metrics import ConfusionMatrix, box_iou, smooth

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "artifacts" / "detector_comparison"
DATA_YAML = PROJECT_ROOT / "artifacts" / "pie_detection_yolo" / "data.yaml"

MODELS = {
    "pretrained_yolo11s": PROJECT_ROOT / "yolo11s.pt",
    "finetuned_train5_best": PROJECT_ROOT / "runs" / "detect" / "train-5" / "weights" / "best.pt",
}

MAX_DET = int(os.environ.get("SMARTPED_EVAL_MAX_DET", "300"))
TAG = os.environ.get("SMARTPED_EVAL_TAG", "")

EVAL_KWARGS = {
    "split": "test",
    "imgsz": 640,
    "batch": 16,
    "device": "0",
    "workers": 4,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": MAX_DET,
    "rect": True,
    "plots": True,
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


def prepare_eval_dataset() -> tuple[Path, Path]:
    """Create a writable evaluation mirror outside the original dataset.

    Ultralytics writes per-video ``*.cache`` files next to label files. To keep
    the original dataset untouched, images are exposed through a directory
    junction (read-only use) and labels are copied to a temp mirror where the
    cache files can be written.
    """
    tmp_root = Path(os.environ.get("TEMP", r"C:\Users\vlctus\AppData\Local\Temp"))
    mirror_root = tmp_root / "opencode" / "smartped_detector_eval" / "pie_detection_yolo"
    src_root = DATA_YAML.parent
    mirror_root.mkdir(parents=True, exist_ok=True)

    src_labels = src_root / "labels"
    dst_labels = mirror_root / "labels"
    expected = sum(1 for _ in src_labels.rglob("*.txt"))
    have = sum(1 for _ in dst_labels.rglob("*.txt")) if dst_labels.exists() else 0
    if have != expected:
        if dst_labels.exists():
            shutil.rmtree(dst_labels)
        shutil.copytree(src_labels, dst_labels)

    src_images = src_root / "images"
    dst_images = mirror_root / "images"
    if not dst_images.exists():
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dst_images), str(src_images)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Could not create image junction for the evaluation mirror: "
                f"{result.stdout} {result.stderr}"
            )

    mirror_yaml = mirror_root / "data.yaml"
    mirror_yaml.write_text(
        "\n".join(
            [
                f'path: "{mirror_root.as_posix()}"',
                "train: images/train",
                "val: images/validation",
                "test: images/test",
                "nc: 1",
                "names:",
                "  0: pedestrian",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return mirror_yaml, mirror_root


class PedestrianOnlyValidator(DetectionValidator):
    """DetectionValidator restricted to class 0 (pedestrian) predictions.

    AP/P/R are computed by the unmodified Ultralytics DetMetrics machinery.
    The only methodological change is that NMS is restricted to class 0 and
    metrics/confusion-matrix names are collapsed to the single pedestrian class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ped_pred_counts = {"conf_0.001": 0, "conf_0.250": 0, "conf_0.500": 0}
        self.ped_max_pred_boxes = 0
        self.ped_images_with_pred_025 = 0
        self.ped_match_counts = {"tp_iou50_conf025": 0, "fp_iou50_conf025": 0, "fn_iou50_conf025": 0}
        self.ped_per_image_pred_025: list[int] = []

    def init_metrics(self, model):
        super().init_metrics(model)
        self.names = {0: "pedestrian"}
        self.nc = 1
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(
            names=self.names, save_matches=self.args.plots and self.args.visualize
        )

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
            confs = predn["conf"].detach().cpu().numpy() if predn["conf"].numel() else np.zeros(0)
            n = int(confs.size)
            self.ped_max_pred_boxes = max(self.ped_max_pred_boxes, n)
            for threshold in (0.001, 0.25, 0.5):
                self.ped_pred_counts[f"conf_{threshold:.3f}"] += int((confs >= threshold).sum())

            keep = confs >= 0.25
            n25 = int(keep.sum())
            self.ped_per_image_pred_025.append(n25)
            if n25:
                self.ped_images_with_pred_025 += 1

            pbox = predn["bboxes"][keep]
            pcls = predn["cls"][keep]
            gbox = pbatch["bboxes"]
            gcls = pbatch["cls"]
            tp = 0
            if len(pbox) and len(gbox):
                iou = box_iou(gbox, pbox)
                correct = self.match_predictions(pcls, gcls, iou)
                tp = int(correct[:, 0].sum())
            self.ped_match_counts["tp_iou50_conf025"] += tp
            self.ped_match_counts["fp_iou50_conf025"] += int(len(pbox) - tp)
            self.ped_match_counts["fn_iou50_conf025"] += int(len(gbox) - tp)
        super().update_metrics(preds, batch)

    def finalize_metrics(self):
        super().finalize_metrics()
        self.metrics.seen = int(self.seen)
        self.metrics.pred_counts = dict(self.ped_pred_counts)
        self.metrics.max_pred_boxes = int(self.ped_max_pred_boxes)
        self.metrics.images_with_pred_025 = int(self.ped_images_with_pred_025)
        self.metrics.match_counts = dict(self.ped_match_counts)
        self.metrics.per_image_pred_025 = list(self.ped_per_image_pred_025)
        self.metrics.pedestrian_names = dict(self.names)
        self.metrics.pedestrian_nc = int(self.nc)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def extract_metrics(validator) -> dict:
    m = validator.metrics
    speed = dict(m.speed)
    inference_ms = float(speed.get("inference", 0.0) or 0.0)
    pre_ms = float(speed.get("preprocess", 0.0) or 0.0)
    post_ms = float(speed.get("postprocess", 0.0) or 0.0)
    total_ms = pre_ms + inference_ms + post_ms

    f1_curve = np.asarray(m.box.f1_curve)
    best_idx = int(smooth(f1_curve.mean(0), 0.1).argmax()) if f1_curve.size else 0
    best_conf = float(np.asarray(m.box.px)[best_idx]) if f1_curve.size else None

    match = m.match_counts
    tp = match["tp_iou50_conf025"]
    fp = match["fp_iou50_conf025"]
    fn = match["fn_iou50_conf025"]
    p_fixed = tp / (tp + fp) if (tp + fp) else 0.0
    r_fixed = tp / (tp + fn) if (tp + fn) else 0.0
    f1_fixed = 2 * p_fixed * r_fixed / (p_fixed + r_fixed) if (p_fixed + r_fixed) else 0.0

    per_image = np.asarray(m.per_image_pred_025, dtype=int)
    return {
        "results_dict": {k: float(v) for k, v in m.results_dict.items()},
        "per_class": {
            "names": m.names,
            "ap_class_index": _jsonable(m.box.ap_class_index),
            "precision_at_max_f1": _jsonable(m.box.p),
            "recall_at_max_f1": _jsonable(m.box.r),
            "f1_at_max_f1": _jsonable(m.box.f1),
            "ap50": _jsonable(m.box.ap50),
            "ap50_95": _jsonable(m.box.ap),
            "all_ap_10_iou": _jsonable(m.box.all_ap),
        },
        "best_f1_confidence": best_conf,
        "speed_ms_per_image": {k: float(v) for k, v in speed.items()},
        "fps": {
            "inference_only": 1000.0 / inference_ms if inference_ms > 0 else None,
            "preprocess_plus_inference_plus_postprocess": 1000.0 / total_ms if total_ms > 0 else None,
        },
        "images_evaluated": int(m.seen),
        "ground_truth_boxes": int(np.asarray(m.nt_per_class).sum()),
        "gt_boxes_per_class": _jsonable(m.nt_per_class),
        "gt_images_per_class": _jsonable(m.nt_per_image),
        "predicted_boxes": dict(m.pred_counts),
        "max_predicted_boxes_single_image": int(m.max_pred_boxes),
        "images_with_at_least_one_prediction_conf025": int(m.images_with_pred_025),
        "per_image_pred_boxes_conf025_summary": {
            "min": int(per_image.min()) if per_image.size else None,
            "median": float(np.median(per_image)) if per_image.size else None,
            "mean": float(per_image.mean()) if per_image.size else None,
            "p95": float(np.percentile(per_image, 95)) if per_image.size else None,
            "max": int(per_image.max()) if per_image.size else None,
        },
        "fixed_conf_0.25_iou_0.50_matching": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p_fixed,
            "recall": r_fixed,
            "f1": f1_fixed,
            "annotated_pedestrians_detected": tp,
        },
        "confusion_matrix": _jsonable(m.confusion_matrix.matrix) if m.confusion_matrix is not None else None,
        "confusion_matrix_names": m.confusion_matrix.names if m.confusion_matrix is not None else None,
        "save_dir": str(validator.save_dir),
    }


def main() -> int:
    (OUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "configs").mkdir(parents=True, exist_ok=True)

    mirror_yaml, mirror_root = prepare_eval_dataset()
    eval_kwargs = {**EVAL_KWARGS, "data": str(mirror_yaml)}

    config_record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "original_dataset_yaml": str(DATA_YAML),
        "evaluation_dataset_yaml": str(mirror_yaml),
        "evaluation_dataset_mirror": {
            "root": str(mirror_root),
            "images": "directory junction to the original images (read-only use)",
            "labels": "copied to temp mirror so Ultralytics *.cache files do not touch the original dataset",
        },
        "split": "test (set04 only)",
        "max_det": MAX_DET,
        "output_tag": TAG,
        "evaluation_kwargs": eval_kwargs,
        "method": {
            "framework": "Ultralytics DetectionValidator + DetMetrics",
            "class_restriction": "NMS restricted to class index 0 (pedestrian) for both models",
            "ap": f"Ultralytics mAP: IoU 0.50:0.05:0.95, 10 thresholds; conf=0.001; NMS IoU=0.7; max_det={MAX_DET}",
            "precision_recall_f1_reported": (
                "Ultralytics P/R/F1 at the confidence maximizing mean F1, plus fixed conf=0.25 "
                "IoU=0.50 greedy matching P/R/F1"
            ),
            "speed": "Ultralytics validator speed (preprocess/inference/postprocess ms per image)",
        },
    }
    (OUT_DIR / "configs" / f"evaluation_config{TAG}.json").write_text(
        json.dumps(config_record, indent=2), encoding="utf-8"
    )

    combined = {
        "generated_utc": config_record["generated_utc"],
        "config": config_record,
        "models": {},
    }

    for key, path in MODELS.items():
        log_path = OUT_DIR / "logs" / f"eval_{key}{TAG}.log"
        started = time.time()
        record = {
            "model_key": key,
            "model_path": str(path),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        lines = [f"Evaluating {key}: {path}"]
        try:
            model = YOLO(str(path))
            args = {
                **eval_kwargs,
                "model": str(path),
                "project": str(OUT_DIR / "runs"),
                "name": f"{key}_test_set04{TAG}",
                "exist_ok": False,
            }
            validator = PedestrianOnlyValidator(args=args, _callbacks=model.callbacks)
            validator(model=model.model)
            metrics = extract_metrics(validator)
            record.update(metrics)
            record["status"] = "success"
            record["save_dir"] = str(validator.save_dir)
            lines.append(f"status=success images={metrics['images_evaluated']}")
            lines.append(f"results_dict={metrics['results_dict']}")
            lines.append(f"speed_ms_per_image={metrics['speed_ms_per_image']}")
            lines.append(f"fps={metrics['fps']}")
            lines.append(f"predicted_boxes={metrics['predicted_boxes']}")
            lines.append(f"fixed_conf_0.25={metrics['fixed_conf_0.25_iou_0.50_matching']}")
            lines.append(f"save_dir={validator.save_dir}")
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
        (OUT_DIR / "metrics" / f"eval_{key}{TAG}.json").write_text(
            json.dumps(_jsonable(record), indent=2), encoding="utf-8"
        )
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))

    (OUT_DIR / "metrics" / f"evaluation_summary{TAG}.json").write_text(
        json.dumps(_jsonable(combined), indent=2), encoding="utf-8"
    )
    return 0 if all(r["status"] == "success" for r in combined["models"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
