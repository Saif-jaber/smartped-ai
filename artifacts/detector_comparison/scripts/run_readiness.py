"""Phase 1 readiness checks for SmartPed AI detector comparison.

Read-only: verifies checkpoints, dataset config, test split, class mappings,
and runtime environment. Writes only under artifacts/detector_comparison/.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "artifacts" / "detector_comparison"
DATA_YAML = PROJECT_ROOT / "artifacts" / "pie_detection_yolo" / "data.yaml"
PRETRAINED = PROJECT_ROOT / "yolo11s.pt"
FINETUNED = PROJECT_ROOT / "runs" / "detect" / "train-5" / "weights" / "best.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_simple_yaml(path: Path) -> dict:
    """Parse the small, flat data.yaml used by this project without PyYAML."""
    out: dict = {}
    names: dict[int, str] = {}
    in_names = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            if line.startswith((" ", "\t")):
                key, _, value = line.strip().partition(":")
                names[int(key.strip())] = value.strip().strip("'\"")
                continue
            in_names = False
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip().strip("'\"")
    out["names"] = names
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "phase": "readiness",
    }

    report["checkpoints"] = {}
    for label, path in (("pretrained_yolo11s", PRETRAINED), ("finetuned_train5_best", FINETUNED)):
        report["checkpoints"][label] = {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
            "last_write_time_utc": (
                datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
                if path.is_file()
                else None
            ),
        }

    report["data_yaml"] = {
        "path": str(DATA_YAML),
        "exists": DATA_YAML.is_file(),
        "content": parse_simple_yaml(DATA_YAML) if DATA_YAML.is_file() else None,
    }

    manifest_path = DATA_YAML.parent / "split_manifest.json"
    summary_path = DATA_YAML.parent / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    report["split_manifest"] = manifest
    report["test_split"] = {
        "assigned_sets": manifest.get("assignments", {}).get("test"),
        "only_set04": manifest.get("assignments", {}).get("test") == ["set04"],
        "summary_test": summary.get("split_distribution", {}).get("test"),
    }

    images_test = DATA_YAML.parent / "images" / "test"
    labels_test = DATA_YAML.parent / "labels" / "test"
    image_files = sorted(p for p in images_test.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    label_files = sorted(labels_test.rglob("*.txt"))
    image_stems = {p.relative_to(images_test).with_suffix("").as_posix() for p in image_files}
    label_stems = {p.relative_to(labels_test).with_suffix("").as_posix() for p in label_files}

    classes: list[int] = []
    box_counts: list[int] = []
    box_areas: list[float] = []
    empty_labels = 0
    for path in label_files:
        rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        box_counts.append(len(rows))
        if not rows:
            empty_labels += 1
        for row in rows:
            classes.append(int(float(row[0])))
            box_areas.append(float(row[3]) * float(row[4]))

    class_ids, class_counts = np.unique(np.asarray(classes, dtype=int), return_counts=True) if classes else ([], [])
    counts = np.asarray(box_counts, dtype=int)
    report["test_dataset"] = {
        "images_dir": str(images_test),
        "labels_dir": str(labels_test),
        "image_subdirs": sorted(p.name for p in images_test.iterdir() if p.is_dir()),
        "label_subdirs": sorted(p.name for p in labels_test.iterdir() if p.is_dir()),
        "image_count": len(image_files),
        "label_count": len(label_files),
        "paired_images_without_label": sorted(image_stems - label_stems)[:10],
        "paired_labels_without_image": sorted(label_stems - image_stems)[:10],
        "paired_image_label_count": len(image_stems & label_stems),
        "empty_label_files": empty_labels,
        "ground_truth_boxes": int(len(classes)),
        "class_ids": {int(k): int(v) for k, v in zip(class_ids, class_counts)},
        "boxes_per_image": {
            "min": int(counts.min()) if counts.size else None,
            "median": float(np.median(counts)) if counts.size else None,
            "mean": float(counts.mean()) if counts.size else None,
            "p95": float(np.percentile(counts, 95)) if counts.size else None,
            "max": int(counts.max()) if counts.size else None,
        },
        "normalized_box_area": {
            "min": float(np.min(box_areas)) if box_areas else None,
            "median": float(np.median(box_areas)) if box_areas else None,
            "mean": float(np.mean(box_areas)) if box_areas else None,
            "max": float(np.max(box_areas)) if box_areas else None,
        },
    }

    from PIL import Image

    sample_images = image_files[:3]
    report["image_properties"] = {
        "sample": [
            {"path": str(p), "size": list(Image.open(p).size), "mode": Image.open(p).mode}
            for p in sample_images
        ]
    }

    import torch
    import ultralytics
    import torchvision

    try:
        import cv2

        cv2_version = cv2.__version__
    except Exception:
        cv2_version = None

    report["environment"] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "opencv": cv2_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
    }

    from ultralytics import YOLO

    report["models"] = {}
    for label, path in (("pretrained_yolo11s", PRETRAINED), ("finetuned_train5_best", FINETUNED)):
        model = YOLO(str(path))
        report["models"][label] = {
            "path": str(path),
            "task": model.task,
            "class_count": int(model.model.nc),
            "names": {int(k): v for k, v in model.names.items()},
            "pedestrian_class_index": 0,
            "pedestrian_class_name": model.names.get(0),
            "head_type": type(model.model.model[-1]).__name__,
            "stride": [float(x) for x in model.model.stride.tolist()] if hasattr(model.model, "stride") else None,
        }

    compat_notes = []
    pre_names = report["models"]["pretrained_yolo11s"]["names"]
    fine_names = report["models"]["finetuned_train5_best"]["names"]
    if pre_names.get(0) != "person":
        compat_notes.append(f"Pretrained class 0 is {pre_names.get(0)!r}, expected 'person'.")
    if fine_names.get(0) != "pedestrian":
        compat_notes.append(f"Fine-tuned class 0 is {fine_names.get(0)!r}, expected 'pedestrian'.")
    if report["models"]["pretrained_yolo11s"]["class_count"] != 80:
        compat_notes.append("Pretrained class count is not 80; verify provenance.")
    if report["models"]["finetuned_train5_best"]["class_count"] != 1:
        compat_notes.append("Fine-tuned class count is not 1; verify provenance.")
    if report["test_dataset"]["class_ids"] not in ({0: report["test_dataset"]["ground_truth_boxes"]},):
        compat_notes.append("Test labels contain classes other than 0.")
    if not report["test_split"]["only_set04"]:
        compat_notes.append("Test split is not exactly set04.")
    if not report["environment"]["cuda_available"]:
        compat_notes.append("CUDA is not available; evaluation would fall back to CPU.")
    if report["test_dataset"]["paired_images_without_label"] or report["test_dataset"]["paired_labels_without_image"]:
        compat_notes.append("Image/label pairing mismatch in test split.")
    report["compatibility"] = {
        "issues": compat_notes,
        "ok": not compat_notes,
        "pedestrian_class_mapping": {
            "pretrained": {"index": 0, "name": pre_names.get(0)},
            "finetuned": {"index": 0, "name": fine_names.get(0)},
            "evaluation_plan": "Restrict both models to class index 0 predictions during validation.",
        },
    }

    out_path = OUT_DIR / "configs" / "readiness.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log_lines = [
        "SmartPed AI detector comparison - Phase 1 readiness",
        f"Generated UTC: {report['generated_utc']}",
        f"Project root: {PROJECT_ROOT}",
        "",
        "Checkpoints:",
    ]
    for label, info in report["checkpoints"].items():
        log_lines.append(
            f"  {label}: exists={info['exists']} size={info['size_bytes']} sha256={info['sha256']}"
        )
    log_lines += [
        "",
        f"data.yaml: {DATA_YAML}",
        f"Test split sets: {report['test_split']['assigned_sets']} (only set04={report['test_split']['only_set04']})",
        f"Test images: {report['test_dataset']['image_count']}",
        f"Test labels: {report['test_dataset']['label_count']}",
        f"Test GT pedestrian boxes: {report['test_dataset']['ground_truth_boxes']}",
        f"Test class distribution: {report['test_dataset']['class_ids']}",
        f"Boxes/image max: {report['test_dataset']['boxes_per_image']['max']}",
        "",
        f"CUDA available: {report['environment']['cuda_available']}",
        f"GPU: {report['environment']['gpu_names']}",
        f"Ultralytics: {report['environment']['ultralytics']}",
        f"Torch: {report['environment']['torch']}",
        "",
        "Class mappings:",
        f"  Pretrained: nc={report['models']['pretrained_yolo11s']['class_count']} class0={pre_names.get(0)!r}",
        f"  Fine-tuned: nc={report['models']['finetuned_train5_best']['class_count']} class0={fine_names.get(0)!r}",
        "",
        f"Compatibility OK: {report['compatibility']['ok']}",
        f"Compatibility issues: {compat_notes}",
    ]
    (OUT_DIR / "logs" / "phase1_readiness.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))
    return 0 if report["compatibility"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
