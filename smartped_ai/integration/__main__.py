from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import _DEFAULT_CONFIG, run_integration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate saved crossing-intention outputs with interactions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to the integration JSON configuration.",
    )
    args = parser.parse_args()
    result = run_integration(args.config)
    metrics = result["metrics"]
    print(
        "Integration complete: "
        f"{metrics['prediction_association']['available_interaction_predictions']} "
        "interactions with available intention predictions; "
        f"{metrics['prediction_association']['unavailable_interaction_predictions']} "
        "unavailable."
    )


if __name__ == "__main__":
    main()
