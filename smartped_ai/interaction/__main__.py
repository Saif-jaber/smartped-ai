from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import _DEFAULT_CONFIG, run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze image-space pedestrian-vehicle interactions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to the interaction analysis JSON configuration.",
    )
    args = parser.parse_args()
    result = run_analysis(args.config)
    print(
        "Interaction analysis complete: "
        f"{result['metrics']['candidate_pair_observations']} pair observations, "
        f"{result['metrics']['persistent_pair_count']} persistent pairs."
    )


if __name__ == "__main__":
    main()
