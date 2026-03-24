from __future__ import annotations

import json
import re
from typing import Any

from .data import DEVELOPER_PROMPT, LABELS


def build_inference_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {
            "role": "user",
            "content": (
                "请判断下面这段中文文本是否包含有害内容，并输出分类结果。\n"
                f"可选类别：{', '.join(LABELS)}\n"
                f"文本：{text}"
            ),
        },
    ]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValueError(f"Cannot parse boolean from value: {value!r}")


def extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"Model output does not contain JSON: {text!r}") from None
        return json.loads(match.group(0))


def normalize_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    is_harmful = _coerce_bool(payload["is_harmful"])
    category = str(payload.get("category", "safe")).strip()
    if category not in LABELS:
        category = "safe" if not is_harmful else category

    if not is_harmful:
        category = "safe"

    return {"is_harmful": is_harmful, "category": category}
