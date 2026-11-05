from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import run_extraction

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "trajectory_extraction.json"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "artifacts" / "trajectory_extraction"
_DEFAULT_MODEL = _PROJECT_ROOT / "yolo11s.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-pedestrian image-space movement features from tracked "
            "detections without changing dataset files, model weights, or tracker config."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Tracking output (CSV/JSONL), PIE spatial XML or annotation directory, or video.",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "csv", "jsonl", "pie-xml", "video"),
        default="auto",
        help="Input format (default: infer from the path).",
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--fps",
        type=float,
        help="Override frames per second used when input records have no timestamps.",
    )
    parser.add_argument(
        "--set-id",
        help="Default recording set for tracking files that do not include set_id.",
    )
    parser.add_argument(
        "--video-id",
        help="Default video ID for tracking files that do not include video_id.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_MODEL,
        help=f"Existing detector weights for --input-format video (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--tracker-config",
        type=Path,
        help=(
            "Existing Ultralytics tracker YAML for video input, for example "
            "smartped_botsort.yaml. If omitted, the installed Ultralytics default is used."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detector confidence for video input (default: 0.25).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help=(
            "Process only the first N seconds of video input. Inference is stopped "
            "after the corresponding frame count; the source video is not trimmed."
        ),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help=(
            "Optional PIE *_annt.xml file or annotation directory for a prediction-"
            "versus-ground-truth comparison."
        ),
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for annotation box matching (default: 0.5).",
    )
    args = parser.parse_args(argv)

    try:
        result = run_extraction(
            args.input,
            config_path=args.config,
            output_dir=args.output_dir,
            input_format=args.input_format,
            default_set_id=args.set_id,
            default_video_id=args.video_id,
            model_path=args.model,
            tracker_config=args.tracker_config,
            confidence=args.confidence,
            fps_override=args.fps,
            max_seconds=args.max_seconds,
            annotations=args.annotations,
            comparison_iou_threshold=args.iou_threshold,
        )
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"Trajectory extraction failed: {exc}", file=sys.stderr)
        return 2

    counts = result["summary"]["counts"]
    print(
        f"Extracted {counts['observations']} observations for {counts['tracks']} tracks "
        f"from {result['input_format']} input."
    )
    print(f"CSV: {result['csv_path']}")
    print(f"JSONL: {result['jsonl_path']}")
    if result["comparison_path"] is not None:
        print(f"Comparison: {result['comparison_path']}")
    print(f"Report: {result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
