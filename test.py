from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the gpt-oss offensive detector.")
    parser.add_argument(
        "--config",
        default="configs/gpt_oss_20b_heretic_lora.yaml",
        help="Path to the YAML config.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "scripts/evaluate_model.py",
        "--config",
        args.config,
    ]
    if args.adapter_path:
        command.extend(["--adapter-path", args.adapter_path])
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
