from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offensive_ft.data import prepare_dataset_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ZHateBench SFT data.")
    parser.add_argument("--dataset-root", default="Dataset")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset_artifacts(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
