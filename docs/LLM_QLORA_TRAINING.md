# ChestCT-Agent：本地 QLoRA 微调

训练目标是把工具证据整合为 Agent 所需的 JSON，不是训练模型独立诊断 CT。

## 已生成的数据

- `artifacts/llm_sft/train.jsonl`：21,745 条。
- `artifacts/llm_sft/valid.jsonl`：2,383 条。
- 按患者而非 CT 重建版本划分，训练和验证患者不重叠。
- 标签来自 `train_predicted_labels.csv`，属于报告派生伪标签；目标中强制保留人工复核提示。

当前数据尚未融合训练集的 CT-CLIP 概率，因为本地只有验证集的 100 个 CT-CLIP 推理结果。应先用本数据进行**结构化输出 SFT**；等训练集 CT-CLIP 批量结果可用后，以 `--ct-predictions` 重建数据，进行第二阶段证据冲突对齐。

## 推荐模型和资源

使用本地已下载的 `Qwen/Qwen3.5-9B`。两张 RTX 4090 上以 4-bit QLoRA、每卡 batch size 1、梯度累积 16 开始。不要对当前远程 `Qwen/Qwen3.6-35B-A3B` 做全量微调。

## 训练步骤

在服务器项目根目录、已激活的 conda 环境中安装训练依赖：

```bash
pip install -r requirements-llm-train.txt
```

将模型完整目录放至服务器，例如 `/root/summer_zhl/models/Qwen3.5-9B`，然后离线训练：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/train_llm_qlora.py \
  --model-path /root/summer_zhl/models/Qwen3.5-9B \
  --train-file artifacts/llm_sft/train.jsonl \
  --valid-file artifacts/llm_sft/valid.jsonl \
  --output-dir artifacts/llm_qlora/qwen3_5_9b_evidence_json
```

产物为 `adapter/`，须与同一基座模型共同加载。训练前需确认模型权重已由你下载到本地或服务器；训练脚本只允许本地目录并强制离线，不会下载模型。
