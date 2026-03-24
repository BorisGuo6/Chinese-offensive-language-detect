from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mxfp4Config,
)
from trl import SFTConfig, SFTTrainer

from .config import FineTuneConfig
from .inference import build_inference_messages, extract_json_payload, normalize_prediction


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def _model_kwargs(config: FineTuneConfig, for_inference: bool = False) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "attn_implementation": config.attn_implementation,
        "torch_dtype": _resolve_dtype(config.torch_dtype),
        "use_cache": for_inference,
    }
    if config.use_mxfp4:
        kwargs["quantization_config"] = Mxfp4Config(dequantize=config.dequantize_mxfp4)
    if config.device_map:
        kwargs["device_map"] = config.device_map
    return kwargs


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_peft_model(config: FineTuneConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **_model_kwargs(config, for_inference=False),
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=config.target_modules,
        target_parameters=config.target_parameters,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def train_model(config: FineTuneConfig) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = load_tokenizer(config.model_name_or_path)
    model = build_peft_model(config)

    train_dataset = load_dataset("json", data_files=config.train_file, split="train")
    eval_dataset = load_dataset("json", data_files=config.eval_file, split="train")

    args = SFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        max_length=config.max_length,
        bf16=config.torch_dtype == "bfloat16",
        fp16=config.torch_dtype == "float16",
        gradient_checkpointing=config.gradient_checkpointing,
        assistant_only_loss=config.assistant_only_loss,
        report_to=config.report_to or None,
        seed=config.seed,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.output_dir) / "training_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_inference_model(
    config: FineTuneConfig,
    adapter_path: str | None = None,
):
    tokenizer = load_tokenizer(config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **_model_kwargs(config, for_inference=True),
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def predict_one(
    model,
    tokenizer,
    text: str,
    max_new_tokens: int = 64,
) -> dict[str, object]:
    messages = build_inference_messages(text)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    outputs = model.generate(
        inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = outputs[0][inputs.shape[-1] :]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    payload = extract_json_payload(decoded)
    prediction = normalize_prediction(payload)
    prediction["raw_output"] = decoded
    return prediction
