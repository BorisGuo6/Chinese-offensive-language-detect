from __future__ import annotations

import argparse
from pathlib import Path

from offensive_ft.config import load_config
from offensive_ft.data import prepare_dataset_artifacts
from offensive_ft.trainer import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune gpt-oss-20b-heretic on ZHateBench.")
    parser.add_argument(
        "--config",
        default="configs/gpt_oss_20b_heretic_lora.yaml",
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset preparation if processed files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if not args.skip_prepare:
        train_path = Path(config.train_file)
        eval_path = Path(config.eval_file)
        if not train_path.exists() or not eval_path.exists():
            prepare_dataset_artifacts()

    train_model(config)


if __name__ == "__main__":
    main()
