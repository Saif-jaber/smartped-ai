"""Build the final comparison tables and derived deltas from the uncapped run."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "artifacts" / "detector_comparison"
SUMMARY = OUT_DIR / "metrics" / "evaluation_summary_maxdet1000.json"
SUMMARY_300 = OUT_DIR / "metrics" / "evaluation_summary.json"
READINESS = OUT_DIR / "configs" / "readiness.json"


def pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0 if old else float("nan")


def main() -> int:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary_300 = json.loads(SUMMARY_300.read_text(encoding="utf-8"))
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    pre = summary["models"]["pretrained_yolo11s"]
    fine = summary["models"]["finetuned_train5_best"]
    pre300 = summary_300["models"]["pretrained_yolo11s"]
    fine300 = summary_300["models"]["finetuned_train5_best"]

    def row(name: str, a, b, fmt: str = "{:.4f}", rel: bool = True) -> str:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = b - a
            delta_text = f"{delta:+,}" if isinstance(a, int) and isinstance(b, int) else f"{delta:+.4f}"
            rel_text = f"{pct(b, a):+.1f}%" if rel else "-"
            return f"| {name} | {fmt.format(a)} | {fmt.format(b)} | {delta_text} | {rel_text} |"
        return f"| {name} | {a} | {b} | - | - |"

    lines: list[str] = []
    lines.append("### AP and max-F1 operating point (test set04, n=10,593 images, 38,047 GT boxes)")
    lines.append("")
    lines.append("| Metric | Pretrained YOLO11s | Fine-tuned YOLO11s | Delta (fine - pre) | Relative |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(row("Precision (max-F1)", pre["results_dict"]["metrics/precision(B)"], fine["results_dict"]["metrics/precision(B)"]))
    lines.append(row("Recall (max-F1)", pre["results_dict"]["metrics/recall(B)"], fine["results_dict"]["metrics/recall(B)"]))
    lines.append(row("mAP@0.50", pre["results_dict"]["metrics/mAP50(B)"], fine["results_dict"]["metrics/mAP50(B)"]))
    lines.append(row("mAP@0.50:0.95", pre["results_dict"]["metrics/mAP50-95(B)"], fine["results_dict"]["metrics/mAP50-95(B)"]))
    lines.append("")
    lines.append("### Fixed operating point: conf=0.25, IoU=0.50 greedy matching (matches pipeline default)")
    lines.append("")
    lines.append("| Metric | Pretrained YOLO11s | Fine-tuned YOLO11s | Delta (fine - pre) | Relative |")
    lines.append("|---|---:|---:|---:|---:|")
    p = pre["fixed_conf_0.25_iou_0.50_matching"]
    f = fine["fixed_conf_0.25_iou_0.50_matching"]
    lines.append(row("True positives", p["tp"], f["tp"], "{:,}"))
    lines.append(row("False positives", p["fp"], f["fp"], "{:,}"))
    lines.append(row("False negatives", p["fn"], f["fn"], "{:,}"))
    lines.append(row("Precision", p["precision"], f["precision"]))
    lines.append(row("Recall", p["recall"], f["recall"]))
    lines.append(row("F1", p["f1"], f["f1"]))
    lines.append("")
    lines.append("### Prediction volume and per-image behaviour")
    lines.append("")
    lines.append("| Metric | Pretrained YOLO11s | Fine-tuned YOLO11s | Delta (fine - pre) | Relative |")
    lines.append("|---|---:|---:|---:|---:|")
    pp = pre["predicted_boxes"]
    fp = fine["predicted_boxes"]
    lines.append(row("Boxes @ conf 0.001", pp["conf_0.001"], fp["conf_0.001"], "{:,}"))
    lines.append(row("Boxes @ conf 0.250", pp["conf_0.250"], fp["conf_0.250"], "{:,}"))
    lines.append(row("Boxes @ conf 0.500", pp["conf_0.500"], fp["conf_0.500"], "{:,}"))
    lines.append(row("Max boxes in one image", pre["max_predicted_boxes_single_image"], fine["max_predicted_boxes_single_image"], "{:,}"))
    ps = pre["per_image_pred_boxes_conf025_summary"]
    fs = fine["per_image_pred_boxes_conf025_summary"]
    lines.append(row("Per-image @0.25 median", ps["median"], fs["median"], "{:.1f}"))
    lines.append(row("Per-image @0.25 mean", ps["mean"], fs["mean"], "{:.2f}"))
    lines.append(row("Per-image @0.25 p95", ps["p95"], fs["p95"], "{:.1f}"))
    lines.append(row("Per-image @0.25 max", ps["max"], fs["max"], "{:.0f}"))
    lines.append("")
    lines.append("### Speed (as measured on this machine, RTX 4070 Laptop GPU)")
    lines.append("")
    lines.append("| Metric | Pretrained YOLO11s | Fine-tuned YOLO11s | Delta (fine - pre) | Relative |")
    lines.append("|---|---:|---:|---:|---:|")
    sp = pre["speed_ms_per_image"]
    sf = fine["speed_ms_per_image"]
    lines.append(row("Preprocess (ms/image)", sp["preprocess"], sf["preprocess"], "{:.2f}"))
    lines.append(row("Inference (ms/image)", sp["inference"], sf["inference"], "{:.2f}"))
    lines.append(row("Postprocess (ms/image)", sp["postprocess"], sf["postprocess"], "{:.2f}"))
    fp_ = pre["fps"]
    ff_ = fine["fps"]
    lines.append(row("FPS inference-only", fp_["inference_only"], ff_["inference_only"], "{:.2f}"))
    lines.append(row("FPS end-to-end", fp_["preprocess_plus_inference_plus_postprocess"], ff_["preprocess_plus_inference_plus_postprocess"], "{:.2f}"))
    lines.append("")
    lines.append("### Checkpoint / model complexity")
    lines.append("")
    lines.append("| Property | Pretrained YOLO11s | Fine-tuned YOLO11s |")
    lines.append("|---|---:|---:|")
    lines.append(f"| File size (bytes) | {readiness['checkpoints']['pretrained_yolo11s']['size_bytes']:,} | {readiness['checkpoints']['finetuned_train5_best']['size_bytes']:,} |")
    lines.append(f"| SHA-256 | `{readiness['checkpoints']['pretrained_yolo11s']['sha256']}` | `{readiness['checkpoints']['finetuned_train5_best']['sha256']}` |")
    lines.append("| Parameters | 9,458,752 | 9,428,179 |")
    lines.append("| GFLOPs @ 640 | 21.84 | 21.67 |")
    lines.append("| Classes | 80 (class 0 = person) | 1 (class 0 = pedestrian) |")
    lines.append("")
    lines.append("### max_det sensitivity (300 vs 1000)")
    lines.append("")
    lines.append("| Model | max_det | Boxes @0.001 | Max boxes/image | mAP50 | mAP50-95 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| Pretrained | 300 | {pre300['predicted_boxes']['conf_0.001']:,} | {pre300['max_predicted_boxes_single_image']} | {pre300['results_dict']['metrics/mAP50(B)']:.6f} | {pre300['results_dict']['metrics/mAP50-95(B)']:.8f} |")
    lines.append(f"| Pretrained | 1000 | {pre['predicted_boxes']['conf_0.001']:,} | {pre['max_predicted_boxes_single_image']} | {pre['results_dict']['metrics/mAP50(B)']:.6f} | {pre['results_dict']['metrics/mAP50-95(B)']:.8f} |")
    lines.append(f"| Fine-tuned | 300 | {fine300['predicted_boxes']['conf_0.001']:,} | {fine300['max_predicted_boxes_single_image']} | {fine300['results_dict']['metrics/mAP50(B)']:.6f} | {fine300['results_dict']['metrics/mAP50-95(B)']:.8f} |")
    lines.append(f"| Fine-tuned | 1000 | {fine['predicted_boxes']['conf_0.001']:,} | {fine['max_predicted_boxes_single_image']} | {fine['results_dict']['metrics/mAP50(B)']:.6f} | {fine['results_dict']['metrics/mAP50-95(B)']:.8f} |")

    table = "\n".join(lines) + "\n"
    (OUT_DIR / "metrics" / "comparison_tables.md").write_text(table, encoding="utf-8")

    derived = {
        "primary_run": "max_det=1000 (uncapped)",
        "test_images": pre["images_evaluated"],
        "ground_truth_boxes": pre["ground_truth_boxes"],
        "ap": {
            "mAP50_abs_delta": fine["results_dict"]["metrics/mAP50(B)"] - pre["results_dict"]["metrics/mAP50(B)"],
            "mAP50_rel_pct": pct(fine["results_dict"]["metrics/mAP50(B)"], pre["results_dict"]["metrics/mAP50(B)"]),
            "mAP50_95_abs_delta": fine["results_dict"]["metrics/mAP50-95(B)"] - pre["results_dict"]["metrics/mAP50-95(B)"],
            "mAP50_95_rel_pct": pct(fine["results_dict"]["metrics/mAP50-95(B)"], pre["results_dict"]["metrics/mAP50-95(B)"]),
        },
        "max_f1": {
            "precision_abs_delta": fine["results_dict"]["metrics/precision(B)"] - pre["results_dict"]["metrics/precision(B)"],
            "precision_rel_pct": pct(fine["results_dict"]["metrics/precision(B)"], pre["results_dict"]["metrics/precision(B)"]),
            "recall_abs_delta": fine["results_dict"]["metrics/recall(B)"] - pre["results_dict"]["metrics/recall(B)"],
            "recall_rel_pct": pct(fine["results_dict"]["metrics/recall(B)"], pre["results_dict"]["metrics/recall(B)"]),
        },
        "fixed_conf_0_25": {
            "precision_abs_delta": f["precision"] - p["precision"],
            "precision_rel_pct": pct(f["precision"], p["precision"]),
            "recall_abs_delta": f["recall"] - p["recall"],
            "recall_rel_pct": pct(f["recall"], p["recall"]),
            "f1_abs_delta": f["f1"] - p["f1"],
            "f1_rel_pct": pct(f["f1"], p["f1"]),
            "fp_reduction": p["fp"] - f["fp"],
            "fp_reduction_pct": pct(f["fp"], p["fp"]),
            "fn_increase": f["fn"] - p["fn"],
            "fn_increase_pct": pct(f["fn"], p["fn"]),
        },
        "prediction_volume": {
            "boxes_0_25_reduction": pp["conf_0.250"] - fp["conf_0.250"],
            "boxes_0_25_reduction_pct": pct(fp["conf_0.250"], pp["conf_0.250"]),
            "boxes_0_50_reduction": pp["conf_0.500"] - fp["conf_0.500"],
            "boxes_0_50_reduction_pct": pct(fp["conf_0.500"], pp["conf_0.500"]),
        },
        "speed": {
            "inference_ms_delta": sf["inference"] - sp["inference"],
            "inference_ms_rel_pct": pct(sf["inference"], sp["inference"]),
            "fps_inference_only_delta": ff_["inference_only"] - fp_["inference_only"],
            "fps_end_to_end_delta": ff_["preprocess_plus_inference_plus_postprocess"] - fp_["preprocess_plus_inference_plus_postprocess"],
        },
        "max_det_sensitivity": {
            "pretrained_extra_boxes_1000_vs_300": pre["predicted_boxes"]["conf_0.001"] - pre300["predicted_boxes"]["conf_0.001"],
            "pretrained_map50_95_delta": pre["results_dict"]["metrics/mAP50-95(B)"] - pre300["results_dict"]["metrics/mAP50-95(B)"],
            "finetuned_extra_boxes_1000_vs_300": fine["predicted_boxes"]["conf_0.001"] - fine300["predicted_boxes"]["conf_0.001"],
        },
    }
    (OUT_DIR / "metrics" / "derived_deltas.json").write_text(json.dumps(derived, indent=2), encoding="utf-8")
    print(table)
    print(json.dumps(derived, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
