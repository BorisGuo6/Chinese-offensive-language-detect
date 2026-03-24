from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot.")
    parser.add_argument(
        "--repo-id",
        default="p-e-w/gpt-oss-20b-heretic",
        help="Model repo on Hugging Face.",
    )
    parser.add_argument(
        "--local-dir",
        default="models/p-e-w/gpt-oss-20b-heretic",
        help="Target directory for the downloaded snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            local_dir=str(local_dir),
        )
    except KeyboardInterrupt:
        print(f"Download interrupted. Partial files remain in: {local_dir}")
        print("Re-run the same command to resume downloading.")
        raise SystemExit(130) from None

    print(f"Model downloaded to: {local_dir}")


if __name__ == "__main__":
    main()
