from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    default_config = Path(__file__).resolve().parents[2] / "configs" / "pie_dataset.json"
    parser = argparse.ArgumentParser(
        description="Ingest and validate local PIE dataset annotations without extracting video frames."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Dataset configuration JSON (default: {default_config})",
    )
    args = parser.parse_args(argv)
    try:
        result = run_pipeline(args.config)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Dataset validation could not run: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Dataset validation failed with an I/O error: {exc}", file=sys.stderr)
        return 2

    status = "FAIL" if result["error_count"] else "PASS"
    print(
        f"{status}: {result['statistics']['videos']} videos, "
        f"{result['error_count']} errors, {result['warning_count']} warnings."
    )
    print(f"Report: {result['output_dir'] / 'validation_report.md'}")
    print(f"Manifest: {result['output_dir'] / 'dataset_manifest.json'}")
    return 1 if result["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
