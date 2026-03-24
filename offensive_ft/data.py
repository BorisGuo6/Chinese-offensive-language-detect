from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = [
    "safe",
    "sexual",
    "abuse",
    "gender_bias",
    "occupation_bias",
    "race_bias",
    "region_bias",
]

DEVELOPER_PROMPT = """你是中文有害内容检测模型。
你只能输出 JSON，不能输出解释、前后缀、Markdown 或多余文本。
JSON 必须严格满足以下模式：
{"is_harmful": true|false, "category": "safe|sexual|abuse|gender_bias|occupation_bias|race_bias|region_bias"}"""

SOURCE_SPECS = {
    "AbuseSet/AbuseSet.csv": {
        "source_dataset": "AbuseSet",
        "harmful_category": "abuse",
    },
    "BiasSet/BiasSet_genden.csv": {
        "source_dataset": "BiasSet_gender",
        "harmful_category": "gender_bias",
    },
    "BiasSet/Bias_occupation.csv": {
        "source_dataset": "BiasSet_occupation",
        "harmful_category": "occupation_bias",
    },
    "BiasSet/Bias_race.csv": {
        "source_dataset": "BiasSet_race",
        "harmful_category": "race_bias",
    },
    "BiasSet/Bias_region.csv": {
        "source_dataset": "BiasSet_region",
        "harmful_category": "region_bias",
    },
    "SexHarmset/SexHarmSet.csv": {
        "source_dataset": "SexHarmSet",
        "harmful_category": "sexual",
    },
}


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns=lambda column: str(column).lstrip("\ufeff").strip())
    expected = {"Keyword", "Type", "Sentence"}
    missing = expected - set(renamed.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return renamed


def _target_payload(is_harmful: bool, harmful_category: str) -> dict[str, object]:
    category = harmful_category if is_harmful else "safe"
    return {"is_harmful": bool(is_harmful), "category": category}


def _messages_for_text(text: str, target: dict[str, object]) -> list[dict[str, str]]:
    user_prompt = (
        "请判断下面这段中文文本是否包含有害内容，并输出分类结果。\n"
        f"可选类别：{', '.join(LABELS)}\n"
        f"文本：{text}"
    )
    return [
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def load_raw_dataset(dataset_root: str | Path = "Dataset") -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    frames: list[pd.DataFrame] = []

    for relative_path, spec in SOURCE_SPECS.items():
        csv_path = dataset_root / relative_path
        frame = _normalize_columns(pd.read_csv(csv_path))
        frame = frame.assign(
            text=frame["Sentence"].astype(str).str.strip(),
            keyword=frame["Keyword"].astype(str).str.strip(),
            source_dataset=spec["source_dataset"],
            source_category=spec["harmful_category"],
        )
        frame["is_harmful"] = frame["Type"].map({"Harmful": True, "Safe": False})
        if frame["is_harmful"].isna().any():
            invalid = sorted(frame.loc[frame["is_harmful"].isna(), "Type"].unique())
            raise ValueError(f"Unexpected label values in {csv_path}: {invalid}")
        frame["label"] = frame.apply(
            lambda row: row["source_category"] if row["is_harmful"] else "safe",
            axis=1,
        )
        frame["split_group"] = frame.apply(
            lambda row: (
                row["source_category"]
                if row["is_harmful"]
                else f"safe::{row['source_category']}"
            ),
            axis=1,
        )
        frames.append(
            frame[
                [
                    "text",
                    "keyword",
                    "Type",
                    "is_harmful",
                    "label",
                    "source_dataset",
                    "source_category",
                    "split_group",
                ]
            ].copy()
        )

    dataset = pd.concat(frames, ignore_index=True)
    dataset.insert(0, "record_id", [f"sample-{index:06d}" for index in range(len(dataset))])
    return dataset


def split_dataset(
    dataset: pd.DataFrame,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    if validation_ratio <= 0 or test_ratio <= 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be > 0 and sum to < 1")

    stratify_key = dataset["split_group"]
    train_frame, temp_frame = train_test_split(
        dataset,
        test_size=validation_ratio + test_ratio,
        random_state=seed,
        stratify=stratify_key,
    )

    temp_test_fraction = test_ratio / (validation_ratio + test_ratio)
    temp_stratify_key = temp_frame["split_group"]
    validation_frame, test_frame = train_test_split(
        temp_frame,
        test_size=temp_test_fraction,
        random_state=seed,
        stratify=temp_stratify_key,
    )

    return {
        "train": train_frame.reset_index(drop=True),
        "validation": validation_frame.reset_index(drop=True),
        "test": test_frame.reset_index(drop=True),
    }


def build_sft_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        target = _target_payload(
            is_harmful=bool(row["is_harmful"]),
            harmful_category=str(row["source_category"]),
        )
        records.append(
            {
                "record_id": row["record_id"],
                "text": row["text"],
                "keyword": row["keyword"],
                "is_harmful": bool(row["is_harmful"]),
                "label": row["label"],
                "source_dataset": row["source_dataset"],
                "source_category": row["source_category"],
                "target": target,
                "messages": _messages_for_text(str(row["text"]), target),
            }
        )
    return records


def _write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_dataset_artifacts(
    dataset_root: str | Path = "Dataset",
    output_dir: str | Path = "data/processed",
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_raw_dataset(dataset_root)
    splits = split_dataset(
        dataset,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    dataset.to_csv(output_dir / "zhatebench_all.csv", index=False)

    summary: dict[str, object] = {
        "total_examples": len(dataset),
        "labels": dataset["label"].value_counts().sort_index().to_dict(),
        "source_datasets": dataset["source_dataset"].value_counts().sort_index().to_dict(),
        "splits": {},
    }

    for split_name, split_frame in splits.items():
        split_frame.to_csv(output_dir / f"zhatebench_{split_name}.csv", index=False)
        records = build_sft_records(split_frame)
        _write_jsonl(records, output_dir / f"zhatebench_sft_{split_name}.jsonl")
        summary["splits"][split_name] = {
            "size": len(split_frame),
            "labels": split_frame["label"].value_counts().sort_index().to_dict(),
        }

    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary
