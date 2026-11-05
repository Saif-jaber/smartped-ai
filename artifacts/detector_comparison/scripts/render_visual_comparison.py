"""Render side-by-side pedestrian detection comparisons on selected set04 frames.

Panels: original + GT, pretrained YOLO11s, fine-tuned YOLO11s, and an overlay.
All predictions use conf=0.25, iou=0.7, imgsz=640, class 0 only. This display
threshold is intentionally higher than the AP evaluation threshold (0.001).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "artifacts" / "detector_comparison"
DATASET = PROJECT_ROOT / "artifacts" / "pie_detection_yolo"
SELECTION = OUT_DIR / "configs" / "visual_selection.json"
READINESS = OUT_DIR / "configs" / "readiness.json"

PRETRAINED_PATH = PROJECT_ROOT / "yolo11s.pt"
FINETUNED_PATH = PROJECT_ROOT / "runs" / "detect" / "train-5" / "weights" / "best.pt"

PANEL_W = 800
PANEL_H = 450
HEADER_H = 46
FOOTER_H = 22
GAP = 20
TITLE_H = 82

COLOR_GT = (0, 200, 0)  # BGR green
COLOR_PRE = (255, 120, 30)  # BGR blue-ish
COLOR_FINE = (0, 140, 255)  # BGR orange
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_HEADER = (32, 32, 32)
COLOR_FOOTER = (24, 24, 24)

PRED_KWARGS = dict(
    imgsz=640,
    conf=0.25,
    iou=0.7,
    max_det=300,
    classes=[0],
    device=0,
    verbose=False,
)


def read_gt(label_path: Path, width: int, height: int) -> list[dict]:
    boxes: list[dict] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        boxes.append(
            {
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "cls": cls,
                "area": float(bw * bh),
            }
        )
    return boxes


def result_to_boxes(result) -> list[dict]:
    boxes: list[dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    for coords, score, klass in zip(xyxy, conf, cls):
        boxes.append(
            {
                "xyxy": [float(v) for v in coords],
                "conf": float(score),
                "cls": int(klass),
            }
        )
    boxes.sort(key=lambda b: b["conf"], reverse=True)
    return boxes


def predict_one(model: YOLO, image_path: Path) -> list[dict]:
    result = model.predict(source=str(image_path), **PRED_KWARGS)[0]
    return result_to_boxes(result)


def predict_many(model: YOLO, image_paths: list[Path]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    results = model.predict(
        source=[str(p) for p in image_paths],
        stream=True,
        batch=16,
        **PRED_KWARGS,
    )
    for path, result in zip(image_paths, results):
        output[str(path)] = result_to_boxes(result)
    return output


def draw_box(
    canvas: np.ndarray,
    xyxy: list[float],
    color: tuple[int, int, int],
    thickness: int,
    label: str | None = None,
    label_at_bottom: bool = False,
    top_limit: int = 0,
    bottom_limit: int | None = None,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = canvas.shape[:2]
    if bottom_limit is None:
        bottom_limit = h
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(top_limit, min(h - 1, y1))
    y2 = max(top_limit, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    if label_at_bottom:
        if y2 + th + 4 <= bottom_limit:
            ty = y2 + th + 4
        else:
            ty = max(top_limit + th + 4, y1 - 4)
    else:
        if y1 - 4 >= top_limit + th + 4:
            ty = y1 - 4
        else:
            ty = min(bottom_limit - 2, y1 + th + 4)
    top = ty - th - 3
    left = max(0, min(w - tw - 6, x1))
    cv2.rectangle(canvas, (left, top), (left + tw + 4, ty + baseline - 2), COLOR_BLACK, -1)
    cv2.putText(
        canvas,
        label,
        (left + 2, ty - 1),
        font,
        font_scale,
        COLOR_WHITE,
        text_thickness,
        cv2.LINE_AA,
    )


def scale_xyxy(xyxy: list[float], width: int, height: int) -> list[float]:
    sx = PANEL_W / float(width)
    sy = PANEL_H / float(height)
    return [
        xyxy[0] * sx,
        xyxy[1] * sy + HEADER_H,
        xyxy[2] * sx,
        xyxy[3] * sy + HEADER_H,
    ]


def draw_legend(panel: np.ndarray, items: list[tuple[str, tuple[int, int, int]]], right_text: str) -> None:
    y0 = HEADER_H + PANEL_H + 4
    x = 8
    for name, color in items:
        cv2.rectangle(panel, (x, y0 + 2), (x + 11, y0 + 13), color, -1)
        cv2.putText(
            panel,
            name,
            (x + 15, y0 + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )
        x += 15 + int(cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]) + 14
    (tw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(
        panel,
        right_text,
        (PANEL_W - tw - 8, y0 + 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def make_panel(
    image: np.ndarray,
    gt: list[dict],
    preds: list[dict] | None,
    title: str,
    subtitle: str,
    mode: str,
) -> np.ndarray:
    height, width = image.shape[:2]
    panel = np.zeros((HEADER_H + PANEL_H + FOOTER_H, PANEL_W, 3), dtype=np.uint8)
    panel[:HEADER_H] = COLOR_HEADER
    panel[HEADER_H + PANEL_H :] = COLOR_FOOTER

    resized = cv2.resize(image, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    panel[HEADER_H : HEADER_H + PANEL_H] = resized

    cv2.putText(
        panel,
        title,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        subtitle,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    top_limit = HEADER_H
    bottom_limit = HEADER_H + PANEL_H
    if mode in {"gt", "pretrained", "finetuned", "overlay"}:
        gt_thickness = 2 if mode == "gt" else 1
        for box in gt:
            draw_box(
                panel,
                scale_xyxy(box["xyxy"], width, height),
                COLOR_GT,
                gt_thickness,
                top_limit=top_limit,
                bottom_limit=bottom_limit,
            )

    if mode in {"pretrained", "overlay"} and preds is not None:
        for box in preds:
            draw_box(
                panel,
                scale_xyxy(box["xyxy"], width, height),
                COLOR_PRE,
                2,
                label=f"{box['conf']:.2f}",
                top_limit=top_limit,
                bottom_limit=bottom_limit,
            )

    if mode in {"finetuned", "overlay"} and preds is not None:
        for box in preds:
            draw_box(
                panel,
                scale_xyxy(box["xyxy"], width, height),
                COLOR_FINE,
                2,
                label=f"{box['conf']:.2f}",
                label_at_bottom=(mode == "overlay"),
                top_limit=top_limit,
                bottom_limit=bottom_limit,
            )

    legend: list[tuple[str, tuple[int, int, int]]] = [("GT", COLOR_GT)]
    if mode in {"pretrained", "overlay"}:
        legend.append(("Pretrained", COLOR_PRE))
    if mode in {"finetuned", "overlay"}:
        legend.append(("Fine-tuned", COLOR_FINE))
    draw_legend(panel, legend, "conf>=0.25 | IoU 0.70")
    return panel


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_montage(panels: list[np.ndarray], title: str, reason: str) -> np.ndarray:
    panel_h, panel_w = panels[0].shape[:2]
    width = GAP + panel_w + GAP + panel_w + GAP
    height = TITLE_H + GAP + panel_h + GAP + panel_h + GAP
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.putText(
        canvas,
        title,
        (GAP, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )
    for i, line in enumerate(wrap_text(reason, 130)[:2]):
        cv2.putText(
            canvas,
            line,
            (GAP, 52 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )
    positions = [
        (GAP, TITLE_H),
        (GAP + panel_w + GAP, TITLE_H),
        (GAP, TITLE_H + panel_h + GAP),
        (GAP + panel_w + GAP, TITLE_H + panel_h + GAP),
    ]
    for panel, (x, y) in zip(panels, positions):
        canvas[y : y + panel_h, x : x + panel_w] = panel
    return canvas


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    selected = selection["frames"]
    used_keys = {(r["video_id"], r["frame_index"]) for r in selected}

    pretrained = YOLO(str(PRETRAINED_PATH))
    finetuned = YOLO(str(FINETUNED_PATH))

    # Deterministic 20-frame-per-video pool for the neutral disagreement supplement.
    all_images = sorted((DATASET / "images" / "test" / "set04").glob("*/*.jpg"))
    by_video: dict[str, list[Path]] = {}
    for path in all_images:
        by_video.setdefault(path.parent.name, []).append(path)
    pool: list[Path] = []
    for video_id in sorted(by_video):
        paths = sorted(by_video[video_id])
        if not paths:
            continue
        idxs = np.linspace(0, len(paths) - 1, num=min(20, len(paths)), dtype=int)
        for idx in idxs:
            path = paths[int(idx)]
            frame_index = int(path.stem.split("_")[-1])
            if (video_id, frame_index) not in used_keys:
                pool.append(path)

    print(f"disagreement pool: {len(pool)} frames")
    pre_pool = predict_many(pretrained, pool)
    fine_pool = predict_many(finetuned, pool)
    candidates = []
    for path in pool:
        key = str(path)
        n_pre = len(pre_pool.get(key, []))
        n_fine = len(fine_pool.get(key, []))
        video_id = path.parent.name
        frame_index = int(path.stem.split("_")[-1])
        candidates.append(
            {
                "path": path,
                "video_id": video_id,
                "frame_index": frame_index,
                "n_pre": n_pre,
                "n_fine": n_fine,
                "abs_diff": abs(n_pre - n_fine),
                "total": n_pre + n_fine,
            }
        )
    candidates.sort(
        key=lambda c: (
            -c["abs_diff"],
            -c["total"],
            c["video_id"] in {r["video_id"] for r in selected},
            c["video_id"],
            c["frame_index"],
        )
    )
    disagreement = candidates[:2]
    print("disagreement frames:")
    for c in disagreement:
        print(
            f"  {c['video_id']} frame_{c['frame_index']:06d} "
            f"pre={c['n_pre']} fine={c['n_fine']} |diff|={c['abs_diff']}"
        )

    visual_frames = []
    for record in selected:
        visual_frames.append(
            {
                **record,
                "pretrained": None,
                "finetuned": None,
                "pool_predictions": None,
            }
        )
    for c in disagreement:
        visual_frames.append(
            {
                "video_id": c["video_id"],
                "frame_index": c["frame_index"],
                "image": str(c["path"]),
                "label": str(
                    DATASET
                    / "labels"
                    / "test"
                    / "set04"
                    / c["video_id"]
                    / f"frame_{c['frame_index']:06d}.txt"
                ),
                "n_boxes": None,
                "n_none": None,
                "n_part": None,
                "n_full": None,
                "median_area": None,
                "max_pair_iou": None,
                "category": "model_disagreement_supplement",
                "selection_reason": (
                    "Largest absolute difference in detection count at conf 0.25 among a "
                    "deterministic 20-frame-per-video sample; symmetric criterion, not a "
                    "quality judgment."
                ),
                "pretrained": None,
                "finetuned": None,
                "pool_predictions": {
                    "pretrained": pre_pool[str(c["path"])],
                    "finetuned": fine_pool[str(c["path"])],
                },
            }
        )

    manifest = {
        "generated_by": "render_visual_comparison.py",
        "display_settings": {
            **{k: v for k, v in PRED_KWARGS.items() if k != "device"},
            "device": "0",
            "note": "Display threshold conf=0.25; AP evaluation uses conf=0.001.",
        },
        "checkpoints": {
            "pretrained_yolo11s": {
                "path": str(PRETRAINED_PATH),
                "sha256": readiness["checkpoints"]["pretrained_yolo11s"]["sha256"],
            },
            "finetuned_train5_best": {
                "path": str(FINETUNED_PATH),
                "sha256": readiness["checkpoints"]["finetuned_train5_best"]["sha256"],
            },
        },
        "frames": [],
    }

    for record in visual_frames:
        image_path = Path(record["image"])
        label_path = Path(record["label"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        gt = read_gt(label_path, width, height)
        if record["pool_predictions"] is not None:
            pre_boxes = record["pool_predictions"]["pretrained"]
            fine_boxes = record["pool_predictions"]["finetuned"]
        else:
            pre_boxes = predict_one(pretrained, image_path)
            fine_boxes = predict_one(finetuned, image_path)

        category = record["category"]
        stem = f"{record['video_id']}_frame_{record['frame_index']:06d}"
        out_dir = OUT_DIR / "visuals" / category
        out_dir.mkdir(parents=True, exist_ok=True)

        panels = [
            make_panel(
                image,
                gt,
                None,
                "Original + GT",
                f"GT boxes: {len(gt)}",
                "gt",
            ),
            make_panel(
                image,
                gt,
                pre_boxes,
                "Pretrained YOLO11s",
                f"GT: {len(gt)} | detections: {len(pre_boxes)}",
                "pretrained",
            ),
            make_panel(
                image,
                gt,
                fine_boxes,
                "Fine-tuned YOLO11s",
                f"GT: {len(gt)} | detections: {len(fine_boxes)}",
                "finetuned",
            ),
            make_panel(
                image,
                gt,
                None,
                "Overlay",
                f"GT: {len(gt)} | pre: {len(pre_boxes)} | fine: {len(fine_boxes)}",
                "overlay",
            ),
        ]
        # The overlay panel needs both prediction sets; draw them explicitly.
        overlay = panels[3]
        for box in pre_boxes:
            draw_box(
                overlay,
                scale_xyxy(box["xyxy"], width, height),
                COLOR_PRE,
                2,
                label=f"{box['conf']:.2f}",
                top_limit=HEADER_H,
                bottom_limit=HEADER_H + PANEL_H,
            )
        for box in fine_boxes:
            draw_box(
                overlay,
                scale_xyxy(box["xyxy"], width, height),
                COLOR_FINE,
                2,
                label=f"{box['conf']:.2f}",
                label_at_bottom=True,
                top_limit=HEADER_H,
                bottom_limit=HEADER_H + PANEL_H,
            )

        names = ["original_gt", "pretrained", "finetuned", "overlay"]
        for panel, name in zip(panels, names):
            out_path = out_dir / f"{stem}_{name}.png"
            if not cv2.imwrite(str(out_path), panel):
                raise IOError(f"Failed to write {out_path}")

        title = f"{category} | {record['video_id']} frame_{record['frame_index']:06d}"
        montage = make_montage(panels, title, record["selection_reason"])
        montage_path = out_dir / f"{stem}_montage.png"
        if not cv2.imwrite(str(montage_path), montage):
            raise IOError(f"Failed to write {montage_path}")

        manifest["frames"].append(
            {
                "category": category,
                "selection_reason": record["selection_reason"],
                "video_id": record["video_id"],
                "frame_index": record["frame_index"],
                "image": str(image_path),
                "label": str(label_path),
                "width": width,
                "height": height,
                "gt_boxes": gt,
                "pretrained_boxes": pre_boxes,
                "finetuned_boxes": fine_boxes,
                "counts": {
                    "gt": len(gt),
                    "pretrained": len(pre_boxes),
                    "finetuned": len(fine_boxes),
                },
                "montage": str(montage_path),
                "panels": {
                    name: str(out_dir / f"{stem}_{name}.png") for name in names
                },
            }
        )
        print(
            f"rendered {category:30s} {stem} "
            f"gt={len(gt)} pre={len(pre_boxes)} fine={len(fine_boxes)}"
        )

    manifest_path = OUT_DIR / "visuals" / "visual_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
