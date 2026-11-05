"""Create a Markdown index of the rendered visual comparisons."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "artifacts" / "detector_comparison"
MANIFEST = OUT_DIR / "visuals" / "visual_manifest.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    lines = [
        "| Category | Frame | GT | Pretrained @0.25 | Fine-tuned @0.25 | Montage |",
        "|---|---|---:|---:|---:|---|",
    ]
    for frame in manifest["frames"]:
        montage = Path(frame["montage"]).relative_to(OUT_DIR).as_posix()
        lines.append(
            f"| {frame['category']} | {frame['video_id']} / "
            f"frame_{frame['frame_index']:06d} | {frame['counts']['gt']} | "
            f"{frame['counts']['pretrained']} | {frame['counts']['finetuned']} | "
            f"[PNG]({montage}) |"
        )
    text = "\n".join(lines) + "\n"
    (OUT_DIR / "metrics" / "visual_index.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
