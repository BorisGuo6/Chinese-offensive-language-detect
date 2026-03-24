# Chinese-offensive-language-detect

本仓库现在包含两部分：

1. 原始中文有害文本数据集 `Dataset/`
2. 基于 `p-e-w/gpt-oss-20b-heretic` 的 LoRA 微调框架

同时，`heretic` 仓库已作为子模块接入到 [third_party/heretic](/home/boris/workspace/Chinese-offensive-language-detect/third_party/heretic)，便于后续继续研究其 abliteration 流程和模型处理逻辑。

## 说明

本数据集用于中文攻击性语言检测任务的研究与开发，包含大量具有攻击性、侮辱性、歧视性、涉黄等内容的文本示例。这些文本仅用于自然语言处理领域的分类建模、对抗鲁棒性测试、语言安全性研究等学术与工程目的。

请注意：

- 数据中的有害文本仅用于研究，不代表作者立场。
- 数据集禁止用于传播、模仿、仇恨动员或任何非研究用途。
- 若对内容存在伦理或合规顾虑，请在使用前自行完成审查。

## 仓库结构

- [Dataset](/home/boris/workspace/Chinese-offensive-language-detect/Dataset): 原始 CSV 数据集
- [offensive_ft](/home/boris/workspace/Chinese-offensive-language-detect/offensive_ft): 新的微调与推理代码
- [scripts](/home/boris/workspace/Chinese-offensive-language-detect/scripts): 数据准备、模型下载、评估脚本
- [configs](/home/boris/workspace/Chinese-offensive-language-detect/configs): `gpt-oss-20b-heretic` LoRA 配置
- [third_party/heretic](/home/boris/workspace/Chinese-offensive-language-detect/third_party/heretic): `https://github.com/p-e-w/heretic.git` 子模块
- [Demo](/home/boris/workspace/Chinese-offensive-language-detect/Demo): 原始前后端演示，保留不动

## 数据集概览

仓库内共有 6 个原始 CSV：

- `Dataset/AbuseSet/AbuseSet.csv`
- `Dataset/BiasSet/BiasSet_genden.csv`
- `Dataset/BiasSet/Bias_occupation.csv`
- `Dataset/BiasSet/Bias_race.csv`
- `Dataset/BiasSet/Bias_region.csv`
- `Dataset/SexHarmset/SexHarmSet.csv`

整合后共有 53,608 条样本，统一映射为以下标签集合：

- `safe`
- `sexual`
- `abuse`
- `gender_bias`
- `occupation_bias`
- `race_bias`
- `region_bias`

预处理脚本会额外生成二分类字段 `is_harmful`，并把数据转换成适合 `gpt-oss` 聊天模板的 `messages` 格式 JSONL。

## 环境准备

建议使用单独虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-finetune.txt
```

如果你是新 clone 的仓库，需要先初始化子模块：

```bash
git submodule update --init --recursive
```

## 下载模型

下载 `p-e-w/gpt-oss-20b-heretic` 到本地默认目录 `models/p-e-w/gpt-oss-20b-heretic`：

```bash
python scripts/download_model.py
```

如果你想改成本地其他目录：

```bash
python scripts/download_model.py \
  --repo-id p-e-w/gpt-oss-20b-heretic \
  --local-dir /path/to/models/p-e-w/gpt-oss-20b-heretic
```

## 数据预处理

将 6 个 CSV 汇总为 `train/validation/test`，并生成 SFT JSONL：

```bash
python scripts/prepare_sft_dataset.py
```

默认输出：

- `data/processed/zhatebench_all.csv`
- `data/processed/zhatebench_train.csv`
- `data/processed/zhatebench_validation.csv`
- `data/processed/zhatebench_test.csv`
- `data/processed/zhatebench_sft_train.jsonl`
- `data/processed/zhatebench_sft_validation.jsonl`
- `data/processed/zhatebench_sft_test.jsonl`
- `data/processed/dataset_summary.json`

## 微调

默认配置见 [configs/gpt_oss_20b_heretic_lora.yaml](/home/boris/workspace/Chinese-offensive-language-detect/configs/gpt_oss_20b_heretic_lora.yaml)。

直接训练：

```bash
python train.py
```

或显式指定配置：

```bash
python train.py --config configs/gpt_oss_20b_heretic_lora.yaml
```

训练脚本会：

- 自动检查并准备 `data/processed/` 数据
- 使用 `TRL + PEFT` 构建 LoRA 微调
- 对 `gpt-oss-20b-heretic` 应用 `target_modules="all-linear"`
- 额外纳入 MoE 专家层参数 `mlp.experts.*` 的 LoRA 目标

默认训练目标是让模型对输入文本输出严格 JSON：

```json
{"is_harmful": true, "category": "abuse"}
```

## 评估

评估 base model：

```bash
python test.py
```

评估 LoRA adapter：

```bash
python test.py --adapter-path outputs/gpt-oss-20b-heretic-lora
```

也可以直接调用脚本：

```bash
python scripts/evaluate_model.py \
  --config configs/gpt_oss_20b_heretic_lora.yaml \
  --adapter-path outputs/gpt-oss-20b-heretic-lora
```

输出包括：

- 预测结果 JSONL
- 分类报告 `*.report.json`

## 硬件说明

这里接入的是 `gpt-oss-20b-heretic`，它是基于 `gpt-oss-20b` 的 MXFP4 权重模型。当前配置遵循 OpenAI 官方 `gpt-oss` 的 Hugging Face 微调建议：

- `transformers >= 4.57`
- `attn_implementation: eager`
- `use_cache: false`
- `Mxfp4Config(dequantize=True)` + LoRA

实际训练时请注意：

- 原生 MXFP4 更适合 Hopper / RTX 50 系 / 更高端显卡
- 单张 24GB 显卡更适合做数据准备、代码验证、少量实验或 CPU offload
- 想稳定完整微调 `20B`，通常仍建议更大的显存或更强的多卡环境

## 与原始仓库的关系

原始仓库中的 `train.py` / `test.py` 是基于 `macbert` 的旧脚本，且存在路径写死与脚本不可直接复现的问题。当前这两个入口已经被替换为新的 `gpt-oss-20b-heretic` 微调与评估入口；`Demo/` 目录保留，但不再作为主训练流程的一部分。

## 数据集引用

如果需要使用该数据集，请引用：

```bibtex
@dataset{luo2025zhatebench,
  author       = {Luo, Yi},
  title        = {ZHateBench: A Comprehensive Chinese Offensive Language Dataset with Harmful–Safe Pairs},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.16812052},
  url          = {https://doi.org/10.5281/zenodo.16812052}
}
```

Plain text:

`Luo, Y. (2025). ZHateBench: A Comprehensive Chinese Offensive Language Dataset with Harmful–Safe Pairs [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16812052`
