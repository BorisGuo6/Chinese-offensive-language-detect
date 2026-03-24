from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offensive_ft.config import FineTuneConfig, load_config
from offensive_ft.trainer import load_inference_model, predict_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a gpt-oss offensive detector.")
    parser.add_argument(
        "--config",
        default="configs/gpt_oss_20b_heretic_lora.yaml",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter directory. If omitted, only the base model is evaluated.",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help="Override test dataset path.",
    )
    parser.add_argument(
        "--output-file",
        default="outputs/eval_predictions.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: FineTuneConfig = load_config(args.config)
    test_file = args.test_file or config.test_file

    frame = pd.read_json(test_file, lines=True)
    model, tokenizer = load_inference_model(config, adapter_path=args.adapter_path)

    predictions = []
    y_true = []
    y_pred = []

    for row in tqdm(frame.to_dict(orient="records"), total=len(frame), desc="Evaluating"):
        prediction = predict_one(
            model,
            tokenizer,
            text=row["text"],
            max_new_tokens=config.max_new_tokens,
        )
        y_true.append(row["label"])
        y_pred.append(prediction["category"])
        prediction_row = {
            "record_id": row["record_id"],
            "text": row["text"],
            "label": row["label"],
            "prediction": prediction["category"],
            "is_harmful": prediction["is_harmful"],
            "raw_output": prediction["raw_output"],
        }
        predictions.append(prediction_row)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Predictions written to: {output_path}")
    print(f"Metrics written to: {report_path}")


if __name__ == "__main__":
    main()
