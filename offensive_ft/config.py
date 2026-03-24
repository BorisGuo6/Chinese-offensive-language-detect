from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FineTuneConfig:
    model_name_or_path: str = "models/p-e-w/gpt-oss-20b-heretic"
    train_file: str = "data/processed/zhatebench_sft_train.jsonl"
    eval_file: str = "data/processed/zhatebench_sft_validation.jsonl"
    test_file: str = "data/processed/zhatebench_sft_test.jsonl"
    output_dir: str = "outputs/gpt-oss-20b-heretic-lora"
    max_length: int = 1024
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    num_train_epochs: float = 2.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    seed: int = 42
    attn_implementation: str = "eager"
    torch_dtype: str = "bfloat16"
    device_map: str | None = "auto"
    use_mxfp4: bool = True
    dequantize_mxfp4: bool = True
    gradient_checkpointing: bool = True
    assistant_only_loss: bool = True
    report_to: list[str] = field(default_factory=list)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    target_parameters: list[str] = field(
        default_factory=lambda: [
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ]
    )
    max_new_tokens: int = 64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> FineTuneConfig:
    config = FineTuneConfig()
    if path is None:
        return config

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    for key, value in payload.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown config field: {key}")
        setattr(config, key, value)

    return config
