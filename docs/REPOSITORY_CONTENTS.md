# 仓库内容与外部资产清单

本文件说明 GitHub 仓库中是否包含运行本项目所需的文件，以及未包含资产的明确来源。

## 已随仓库提供

| 类别 | 内容 | 获取方式 |
| --- | --- | --- |
| 源代码 | `chestct_agent/`、`scripts/`、`demo/`、`tests/` | 直接 `git clone` |
| 使用与交接文档 | `README.md`、`docs/` | 直接 `git clone` |
| 文本分类器 | `artifacts/text_classifier.joblib` | 项目本地训练产物，普通 Git 文件 |
| Stage-1 LoRA | `artifacts/llm_qlora/qwen3_5_9b_evidence_json_1ep_single/adapter/` | 项目本地训练产物；`adapter_model.safetensors` 由 Git LFS 提供 |
| Stage-2 CT-CLIP 融合 LoRA | `artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep/adapter/` | 项目本地训练产物；`adapter_model.safetensors` 由 Git LFS 提供 |

两个 LoRA adapter 目录随仓库包含：权重、`adapter_config.json` 和 `training_args.bin`。使用仓库前请安装 Git LFS：

```bash
git lfs install
git clone https://github.com/zhangsan2233/ct_agent.git
cd ct_agent
git lfs pull
```

## 不随仓库提供

| 资产 | 原因 | 官方来源与放置位置 |
| --- | --- | --- |
| Qwen3.5-9B 基座模型 | 可从官方重新获取，体积大 | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) → `models/Qwen3.5-9B/` |
| CT-CLIP 权重 | 官方受访问控制资产，禁止在此公开再分发 | [CT-CLIP_v2.pt](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/models/CT-CLIP-Related/CT-CLIP_v2.pt) → `models/ctclip/CT-CLIP_v2.pt` |
| CT-CLIP 源码 | 可从官方 GitHub 获取，不复制第三方源码 | [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) → `external/CT-CLIP-main/` |
| CXR-BERT | 可从官方重新获取，体积大 | [microsoft/BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized) → `models/cxrbert/` |
| CT-RATE 数据与 NIfTI | 数据访问受控、体积大、不可公开再分发 | [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) → `data/` 或 `dataset/` |
| 评估结果、图表、日志 | 可由训练/评估流程生成，且可能含数据派生内容 | 本地 `artifacts/`；生成方法见 `docs/AGENT_PIPELINE_HANDOVER.md` |
| `.env` 和 token | 可能包含密钥 | 使用 `.env.example` 创建本地 `.env`，绝不提交 |

详细下载命令、访问申请、路径验证和离线运行规则见 [ASSET_SETUP.md](ASSET_SETUP.md)。

## 最小运行条件

要运行最终的 CT-CLIP + Stage-2 Demo，需要：

1. 本仓库提供的 Stage-2 adapter；
2. 自行从官方来源下载的 Qwen3.5-9B、CT-CLIP 权重/源码和 CXR-BERT；
3. 一个有授权访问权的 CT NIfTI 与报告文本；
4. CUDA 环境及 `requirements-llm-train.txt` 所列依赖。

缺少任何外部资产时，`scripts/run_stage2_agent.py` 会在运行前提示具体缺失路径，而不会静默下载或伪造结果。
