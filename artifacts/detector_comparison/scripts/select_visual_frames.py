"""Select representative set04 test frames for visual comparison.

Selection uses ground-truth geometry and PIE occlusion metadata only. It does
not look at either model's predictions, except for the separately labelled
"model disagreement" supplement selected later by a neutral count-difference
criterion.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = PROJECT_ROOT / "artifacts" / "pie_detection_yolo"
META = DATASET / "annotation_metadata.jsonl.gz"
OUT_DIR = PROJECT_ROOT / "artifacts" / "detector_comparison"


def xywhn_to_xyxy(box):
    x, y, w, h = box
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=float)


def pairwise_iou(boxes: np.ndarray) -> float:
    if len(boxes) < 2:
        return 0.0
    a = boxes[:, None, :]
    b = boxes[None, :, :]
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    iou = inter / (area_a + area_b - inter + 1e-12)
    np.fill_diagonal(iou, 0.0)
    return float(iou.max())


def main() -> int:
    frames: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"boxes": [], "occlusion": {"none": 0, "part": 0, "full": 0}}
    )
    with gzip.open(META, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("split") != "test" or not d.get("label_included"):
                continue
            key = (d["video_id"], int(d["frame_index"]))
            box = d.get("yolo_xywh")
            if not box:
                continue
            frames[key]["boxes"].append(box)
            occ = d.get("occlusion", "none")
            frames[key]["occlusion"][occ if occ in {"none", "part", "full"} else "none"] += 1

    records = []
    for (video_id, frame_index), info in frames.items():
        boxes = np.asarray(info["boxes"], dtype=float)
        xyxy = np.asarray([xywhn_to_xyxy(b) for b in boxes])
        areas = boxes[:, 2] * boxes[:, 3]
        heights = boxes[:, 3]
        records.append(
            {
                "video_id": video_id,
                "frame_index": frame_index,
                "image": str(
                    DATASET
                    / "images"
                    / "test"
                    / "set04"
                    / video_id
                    / f"frame_{frame_index:06d}.jpg"
                ),
                "label": str(
                    DATASET
                    / "labels"
                    / "test"
                    / "set04"
                    / video_id
                    / f"frame_{frame_index:06d}.txt"
                ),
                "n_boxes": int(len(boxes)),
                "n_none": int(info["occlusion"]["none"]),
                "n_part": int(info["occlusion"]["part"]),
                "n_full": int(info["occlusion"]["full"]),
                "median_area": float(np.median(areas)),
                "mean_area": float(areas.mean()),
                "min_area": float(areas.min()),
                "median_height": float(np.median(heights)),
                "max_pair_iou": pairwise_iou(xyxy),
                "boxes_xywhn": boxes.tolist(),
            }
        )

    all_areas = np.asarray([b[2] * b[3] for r in records for b in r["boxes_xywhn"]])
    all_counts = np.asarray([r["n_boxes"] for r in records])
    print(
        "frames",
        len(records),
        "box_area p10/p25/p50/p75/p90",
        [round(float(np.percentile(all_areas, p)), 6) for p in (10, 25, 50, 75, 90)],
        "counts p50/p90/p95/max",
        [float(np.percentile(all_counts, p)) for p in (50, 90, 95)] + [int(all_counts.max())],
    )

    small_threshold = float(np.percentile(all_areas, 25))
    clear_area_threshold = float(np.percentile(all_areas, 75))

    used_videos: set[str] = set()
    used_frames: set[tuple[str, int]] = set()
    selected: list[dict] = []

    def pick(category: str, candidates: list[dict], reason: str, n: int = 2) -> None:
        picked = 0
        for r in candidates:
            key = (r["video_id"], r["frame_index"])
            if key in used_frames or r["video_id"] in used_videos:
                continue
            item = dict(r)
            item["category"] = category
            item["selection_reason"] = reason
            selected.append(item)
            used_frames.add(key)
            used_videos.add(r["video_id"])
            picked += 1
            if picked >= n:
                break
        if picked < n:
            # allow repeated videos if unique-video selection cannot fill the quota
            for r in candidates:
                key = (r["video_id"], r["frame_index"])
                if key in used_frames:
                    continue
                item = dict(r)
                item["category"] = category
                item["selection_reason"] = reason
                selected.append(item)
                used_frames.add(key)
                picked += 1
                if picked >= n:
                    break
        print(f"{category}: picked {picked}")

    # 1. Small or distant pedestrians: at least two boxes, smallest median box area.
    small = [
        r
        for r in records
        if r["n_boxes"] >= 2 and r["median_area"] <= small_threshold
    ]
    small.sort(key=lambda r: (r["median_area"], -r["n_boxes"], r["video_id"], r["frame_index"]))
    pick("small_or_distant", small, "median GT box area in the smallest quartile; >=2 GT boxes")

    # 2. Partially occluded pedestrians: at least one 'part' occlusion box, >=2 boxes.
    occluded = [r for r in records if r["n_part"] >= 1 and r["n_boxes"] >= 2]
    occluded.sort(
        key=lambda r: (
            -r["n_part"],
            r["n_full"],
            -r["n_boxes"],
            -r["max_pair_iou"],
            r["video_id"],
            r["frame_index"],
        )
    )
    pick("partially_occluded", occluded, "contains >=1 PIE 'part' occlusion box and >=2 GT boxes")

    # 3. Multiple pedestrians close together: >=3 boxes with the strongest pairwise GT overlap.
    close = [r for r in records if r["n_boxes"] >= 3 and r["max_pair_iou"] > 0.0]
    close.sort(key=lambda r: (-r["max_pair_iou"], -r["n_boxes"], r["video_id"], r["frame_index"]))
    pick("multiple_close", close, ">=3 GT boxes; highest pairwise GT box overlap")

    # 4. Crowded scenes: highest GT box counts.
    crowded = [r for r in records if r["n_boxes"] >= 9]
    crowded.sort(key=lambda r: (-r["n_boxes"], -r["mean_area"], r["video_id"], r["frame_index"]))
    pick("crowded", crowded, "high GT box count (>=9 pedestrians)")

    # 5. Clear, unobstructed pedestrians: 1-3 boxes, no occlusion, larger boxes, no overlap.
    clear = [
        r
        for r in records
        if 1 <= r["n_boxes"] <= 3
        and r["n_none"] == r["n_boxes"]
        and r["median_area"] >= clear_area_threshold
        and r["max_pair_iou"] == 0.0
    ]
    clear.sort(key=lambda r: (-r["median_area"], r["video_id"], r["frame_index"]))
    pick("clear_unobstructed", clear, "1-3 unoccluded GT boxes with large area and no GT overlap")

    out = {
        "selection_policy": (
            "Ground-truth geometry and PIE occlusion metadata only. Required categories are "
            "selected without reference to model predictions. Model-disagreement examples are "
            "a separately labelled supplement chosen by absolute detection-count difference."
        ),
        "thresholds": {
            "small_median_area_p25": small_threshold,
            "clear_median_area_p75": clear_area_threshold,
        },
        "frames": selected,
    }
    (OUT_DIR / "configs" / "visual_selection.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    for r in selected:
        print(
            f"{r['category']:20s} {r['video_id']} frame_{r['frame_index']:06d} "
            f"n={r['n_boxes']} part={r['n_part']} full={r['n_full']} "
            f"med_area={r['median_area']:.6f} pair_iou={r['max_pair_iou']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
