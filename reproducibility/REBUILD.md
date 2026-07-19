# 500 例 Stage-2 训练复现步骤

本目录不包含 CT 图像、报告正文、CT-CLIP 预测 JSONL 或 SFT JSONL。它们必须由具有 CT-RATE 访问权限的使用者在本地生成。本仓库提供的是精确病例 ID、弱标签、450/50 划分、超参数和 SHA256 对照。

## 1. 校验公开元数据

```bash
sha256sum -c reproducibility/SHA256SUMS
```

前两项 adapter SHA256 应与 `SHA256SUMS` 一致；若通过 Git LFS 克隆，请先执行 `git lfs pull`。

## 2. 获得受授权的 CT-RATE 数据

按照 [ASSET_SETUP.md](../docs/ASSET_SETUP.md) 在 Hugging Face 接受 CT-RATE 的访问条款，下载所需元数据和本目录 `ctrate_500_weak_label_manifest.csv` 中列出的 500 个 `hf_relative_path`。不要把 CT 或报告正文提交回本仓库。

生成本地私有工作清单时，可将公开 manifest 与授权下载的数据路径关联；需要的字段包括 `ct_volume_path`、`report_impression` 和 8 个弱标签。

## 3. 重建 CT-CLIP 预测（私有本地输出）

```bash
export CTCLIP_TEXT_MODEL_DIR="$PWD/models/cxrbert"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python scripts/batch_ctclip_infer.py \
  --manifest /private/path/train_manifest_500_with_reports.csv \
  --out artifacts/ctclip_stage2/ctclip_predictions_500.jsonl \
  --checkpoint models/ctclip/CT-CLIP_v2.pt \
  --source-dir external/CT-CLIP-main \
  --text-model-dir models/cxrbert \
  --device cuda:0
```

CT-CLIP 必须保持冻结；不要把该 JSONL 上传到公共仓库。

## 4. 重建 Stage-2 SFT 文件（私有本地输出）

```bash
python scripts/build_ctclip_stage2_sft.py \
  --predictions artifacts/ctclip_stage2/ctclip_predictions_500.jsonl \
  --out-dir artifacts/ctclip_stage2/sft_compact_500_v1 \
  --val-fraction 0.1 \
  --seed 20260718
```

使用 `reproducibility/patient_disjoint_split.csv` 和 `experiment_config.json` 交叉核验 450/50 划分。该 SFT 会包含报告文本，因此必须保持私有。

## 5. 训练 Stage-2 adapter

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/train_llm_qlora.py \
  --model-path models/Qwen3.5-9B \
  --adapter-path artifacts/llm_qlora/qwen3_5_9b_evidence_json_1ep_single/adapter \
  --train-file artifacts/ctclip_stage2/sft_compact_500_v1/train.jsonl \
  --valid-file artifacts/ctclip_stage2/sft_compact_500_v1/valid.jsonl \
  --output-dir artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep \
  --epochs 2 \
  --learning-rate 5e-5 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-length 2048 \
  --rank 16
```

训练环境、LoRA target modules 和 adapter SHA256 见 `experiment_config.json`、`SHA256SUMS` 与 [AGENT_PIPELINE_HANDOVER.md](../docs/AGENT_PIPELINE_HANDOVER.md)。由于 GPU 算子、软件版本和数据预处理差异，重新训练不保证二进制权重完全相同；应以数据划分、配置和留出集指标进行复核。
