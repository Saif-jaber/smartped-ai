"""Small-subset smoke test for the pedestrian-only validator. Not a result."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_evaluation import PedestrianOnlyValidator, extract_metrics, prepare_eval_dataset  # noqa: E402

from ultralytics import YOLO  # noqa: E402

TMP = Path(r"C:\Users\vlctus\AppData\Local\Temp\opencode\detector_smoke")
TMP.mkdir(parents=True, exist_ok=True)
_, mirror_root = prepare_eval_dataset()
images = sorted((mirror_root / "images" / "test" / "set04").rglob("*.jpg"))[:16]
data_root = mirror_root
list_file = TMP / "smoke_images.txt"
list_file.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")
data_yaml = TMP / "smoke_data.yaml"
data_yaml.write_text(
    "\n".join(
        [
            f'path: "{data_root.as_posix()}"',
            "train: images/train",
            "val: images/validation",
            f'test: "{list_file.as_posix()}"',
            "nc: 1",
            "names:",
            "  0: pedestrian",
            "",
        ]
    ),
    encoding="utf-8",
)

for key, path in (
    ("pretrained_yolo11s", PROJECT_ROOT / "yolo11s.pt"),
    ("finetuned_train5_best", PROJECT_ROOT / "runs" / "detect" / "train-5" / "weights" / "best.pt"),
):
    model = YOLO(str(path))
    args = {
        "model": str(path),
        "data": str(data_yaml),
        "split": "test",
        "imgsz": 640,
        "batch": 4,
        "device": "0",
        "workers": 0,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "rect": True,
        "plots": False,
        "verbose": False,
        "seed": 0,
        "deterministic": True,
        "mode": "val",
        "task": "detect",
        "project": str(TMP / "runs"),
        "name": f"smoke_{key}",
        "exist_ok": True,
    }
    validator = PedestrianOnlyValidator(args=args, _callbacks=model.callbacks)
    validator(model=model.model)
    metrics = extract_metrics(validator)
    print(key, "OK", metrics["images_evaluated"], metrics["ground_truth_boxes"])
    print("  results", metrics["results_dict"])
    print("  pred", metrics["predicted_boxes"])
    print("  match", metrics["fixed_conf_0.25_iou_0.50_matching"])
