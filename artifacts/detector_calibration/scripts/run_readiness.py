"""Phase 1 readiness checks for SmartPed AI detector calibration.

Read-only with respect to the original dataset, checkpoints, pipeline, and
previous comparison artifacts. Writes only under artifacts/detector_calibration/.
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
OUT_DIR = PROJECT_ROOT / "artifacts" / "detector_calibration"
DATA_YAML = PROJECT_ROOT / "artifacts" / "pie_detection_yolo" / "data.yaml"
PREVIOUS_REPORT = PROJECT_ROOT / "artifacts" / "detector_comparison" / "DETECTOR_COMPARISON_REPORT.md"
PRETRAINED = PROJECT_ROOT / "yolo11s.pt"
FINETUNED = PROJECT_ROOT / "runs" / "detect" / "train-5" / "weights" / "best.pt"

EXPECTED_HASHES = {
    "pretrained_yolo11s": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
    "finetuned_train5_best": "9e1d876ff9ccc49d03f43da9dbe5bb4df71c2a7514dbaad24585e4a1f855ecfb",
}


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


def dataset_stats(images_dir: Path, labels_dir: Path) -> dict:
    image_files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    label_files = sorted(labels_dir.rglob("*.txt"))
    image_stems = {p.relative_to(images_dir).with_suffix("").as_posix() for p in image_files}
    label_stems = {p.relative_to(labels_dir).with_suffix("").as_posix() for p in label_files}

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

    class_ids, class_counts = (
        np.unique(np.asarray(classes, dtype=int), return_counts=True) if classes else ([], [])
    )
    counts = np.asarray(box_counts, dtype=int)
    return {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "image_subdirs": sorted(p.name for p in images_dir.iterdir() if p.is_dir()),
        "label_subdirs": sorted(p.name for p in labels_dir.iterdir() if p.is_dir()),
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


def main() -> int:
    (OUT_DIR / "configs").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    report: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "phase": "calibration_readiness",
        "scope": "validation split set06 only; set04 is not used",
    }

    report["checkpoints"] = {}
    for label, path in (("pretrained_yolo11s", PRETRAINED), ("finetuned_train5_best", FINETUNED)):
        computed = sha256(path) if path.is_file() else None
        report["checkpoints"][label] = {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": computed,
            "expected_sha256": EXPECTED_HASHES[label],
            "hash_matches_expected": computed == EXPECTED_HASHES[label],
            "last_write_time_utc": (
                datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
                if path.is_file()
                else None
            ),
        }

    report["previous_comparison_report"] = {
        "path": str(PREVIOUS_REPORT),
        "exists": PREVIOUS_REPORT.is_file(),
    }
    previous_text = PREVIOUS_REPORT.read_text(encoding="utf-8") if PREVIOUS_REPORT.is_file() else ""
    report["previous_comparison_report"]["hashes_present"] = {
        label: expected in previous_text for label, expected in EXPECTED_HASHES.items()
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
    report["validation_split"] = {
        "assigned_sets": manifest.get("assignments", {}).get("validation"),
        "only_set06": manifest.get("assignments", {}).get("validation") == ["set06"],
        "summary_validation": summary.get("split_distribution", {}).get("validation"),
        "test_sets": manifest.get("assignments", {}).get("test"),
        "test_is_set04_only": manifest.get("assignments", {}).get("test") == ["set04"],
    }

    images_val = DATA_YAML.parent / "images" / "validation"
    labels_val = DATA_YAML.parent / "labels" / "validation"
    report["validation_dataset"] = dataset_stats(images_val, labels_val)

    from PIL import Image

    image_files = sorted(p for p in images_val.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sample_images = image_files[:3]
    report["image_properties"] = {
        "sample": [
            {"path": str(p), "size": list(Image.open(p).size), "mode": Image.open(p).mode}
            for p in sample_images
        ]
    }

    import torch
    import torchvision
    import ultralytics

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
        "gpu_names": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
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

    issues: list[str] = []
    if not report["previous_comparison_report"]["exists"]:
        issues.append("Previous detector comparison report is missing.")
    for label, present in report["previous_comparison_report"].get("hashes_present", {}).items():
        if not present:
            issues.append(f"Expected {label} hash is not present in the previous comparison report.")
    for label, info in report["checkpoints"].items():
        if not info["exists"]:
            issues.append(f"Checkpoint missing: {info['path']}")
        elif not info["hash_matches_expected"]:
            issues.append(f"Checkpoint hash mismatch for {label}.")
    if not report["validation_split"]["only_set06"]:
        issues.append("Validation split is not exactly set06.")
    if report["validation_dataset"]["image_subdirs"] != ["set06"]:
        issues.append("Validation image subdirectories are not exactly ['set06'].")
    if report["validation_dataset"]["label_subdirs"] != ["set06"]:
        issues.append("Validation label subdirectories are not exactly ['set06'].")
    if report["validation_dataset"]["class_ids"] != {0: report["validation_dataset"]["ground_truth_boxes"]}:
        issues.append("Validation labels contain classes other than 0.")
    if report["validation_dataset"]["paired_images_without_label"] or report["validation_dataset"][
        "paired_labels_without_image"
    ]:
        issues.append("Image/label pairing mismatch in validation split.")
    if report["models"]["pretrained_yolo11s"]["names"].get(0) != "person":
        issues.append("Pretrained class 0 is not 'person'.")
    if report["models"]["finetuned_train5_best"]["names"].get(0) != "pedestrian":
        issues.append("Fine-tuned class 0 is not 'pedestrian'.")
    if report["models"]["pretrained_yolo11s"]["class_count"] != 80:
        issues.append("Pretrained class count is not 80.")
    if report["models"]["finetuned_train5_best"]["class_count"] != 1:
        issues.append("Fine-tuned class count is not 1.")
    if not report["environment"]["cuda_available"]:
        issues.append("CUDA is not available; calibration would fall back to CPU.")

    report["compatibility"] = {
        "issues": issues,
        "ok": not issues,
        "calibration_scope": "Validation set06 only. Test set04 is explicitly out of scope.",
    }

    (OUT_DIR / "configs" / "readiness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    val = report["validation_dataset"]
    log_lines = [
        "SmartPed AI detector calibration - Phase 1 readiness",
        f"Generated UTC: {report['generated_utc']}",
        f"Project root: {PROJECT_ROOT}",
        "",
        "Checkpoints:",
    ]
    for label, info in report["checkpoints"].items():
        log_lines.append(
            f"  {label}: exists={info['exists']} size={info['size_bytes']} "
            f"sha256={info['sha256']} matches={info['hash_matches_expected']}"
        )
    log_lines += [
        "",
        f"Previous comparison report: {PREVIOUS_REPORT}",
        f"  exists={report['previous_comparison_report']['exists']} "
        f"hashes_present={report['previous_comparison_report']['hashes_present']}",
        "",
        f"data.yaml: {DATA_YAML}",
        f"Validation split sets: {report['validation_split']['assigned_sets']} "
        f"(only set06={report['validation_split']['only_set06']})",
        f"Test split sets (not used): {report['validation_split']['test_sets']}",
        f"Validation images: {val['image_count']}",
        f"Validation labels: {val['label_count']}",
        f"Validation GT pedestrian boxes: {val['ground_truth_boxes']}",
        f"Validation class distribution: {val['class_ids']}",
        f"Boxes/image min/median/mean/p95/max: {val['boxes_per_image']}",
        f"Empty label files: {val['empty_label_files']}",
        "",
        f"CUDA available: {report['environment']['cuda_available']}",
        f"GPU: {report['environment']['gpu_names']}",
        f"Ultralytics: {report['environment']['ultralytics']}",
        f"Torch: {report['environment']['torch']}",
        "",
        "Class mappings:",
        f"  Pretrained: nc={report['models']['pretrained_yolo11s']['class_count']} "
        f"class0={report['models']['pretrained_yolo11s']['names'].get(0)!r}",
        f"  Fine-tuned: nc={report['models']['finetuned_train5_best']['class_count']} "
        f"class0={report['models']['finetuned_train5_best']['names'].get(0)!r}",
        "",
        f"Compatibility ok: {report['compatibility']['ok']}",
        f"Issues: {issues if issues else 'none'}",
    ]
    (OUT_DIR / "logs" / "phase1_readiness.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))
    return 0 if report["compatibility"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
