# 未随仓库提供的模型与数据资产

本仓库只包含代码、配置模板和文档；不包含模型权重、CT-RATE 数据、NIfTI 体数据、训练产物或访问令牌。这样做是为了避免超大文件、访问受限资产、数据再分发和密钥泄露问题。

本项目的最终答辩管线需要以下资产。请只从相应官方来源下载，并遵守各页面显示的许可、访问控制和数据使用条款。

## 1. 目录总览

在项目根目录下准备以下结构：

```text
models/
├── Qwen3.5-9B/                    # Qwen 基座模型
├── ctclip/CT-CLIP_v2.pt           # CT-CLIP 官方权重
└── cxrbert/                       # CXR-BERT 文本编码器
external/
└── CT-CLIP-main/                  # CT-CLIP 官方源码
data/                              # CT-RATE 元数据和/或 CT 体数据
artifacts/
└── llm_qlora/.../adapter/         # 本项目训练得到的 Stage-2 adapter
```

所有这些目录均被 `.gitignore` 排除；不要将它们、`.env` 或任何 token 提交到 GitHub。

## 2. Qwen3.5-9B 基座模型

- 官方页面：[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- 本项目目录：`models/Qwen3.5-9B`
- 许可证：以官方模型页当前标示为准。

在完成 Hugging Face 登录后，可使用：

```bash
hf download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B
```

下载完成后，目录中应包含模型配置、processor/tokenizer 文件及多个 `model.safetensors-*-of-*` 分片。最终 Demo 以 4-bit 方式加载该模型，不会在运行时联网下载。

## 3. CT-CLIP 源码与权重

### 官方源码

- 官方仓库：[ibrahimethemhamamci/CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP)
- 本项目目录：`external/CT-CLIP-main`

```bash
git clone --depth 1 https://github.com/ibrahimethemhamamci/CT-CLIP.git external/CT-CLIP-main
```

### 官方权重

- 权重文件：`CT-CLIP_v2.pt`
- 官方数据页面：[CT-RATE / CT-CLIP-related model file](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/models/CT-CLIP-Related/CT-CLIP_v2.pt)
- 本项目目录：`models/ctclip/CT-CLIP_v2.pt`

该文件位于受访问控制的 CT-RATE 资产中。先登录 Hugging Face、在网页接受访问条款并取得访问权限；随后从该页面下载，放置到上述路径。若命令行下载已获授权，也可使用：

```bash
hf download ibrahimhamamci/CT-RATE models/CT-CLIP-Related/CT-CLIP_v2.pt \
  --repo-type dataset \
  --local-dir .cache/ct-rate-download
```

再将下载出的 `models/CT-CLIP-Related/CT-CLIP_v2.pt` 移至 `models/ctclip/CT-CLIP_v2.pt`。请不要将该权重再次上传到公开 GitHub 仓库。

## 4. CXR-BERT 文本编码器

- 官方页面：[microsoft/BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized)
- 本项目目录：`models/cxrbert`

```bash
hf download microsoft/BiomedVLP-CXR-BERT-specialized --local-dir models/cxrbert
```

CT-CLIP 推理时，环境变量 `CTCLIP_TEXT_MODEL_DIR` 会指向该本地目录，且管线强制离线模式，避免演示过程中发生网络下载。

## 5. CT-RATE 数据

- 官方页面：[ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- 访问要求：先在官方页面接受访问条款；不要将 CT 图像、报告或派生结果上传至公开仓库。

只下载元数据时：

```bash
python scripts/download_ct_rate.py \
  --repo-id ibrahimhamamci/CT-RATE \
  --data-dir data \
  --metadata-only
```

本项目的答辩运行只需一个可访问的 `.nii` / `.nii.gz` CT 和对应报告文本；不需要重新下载或扩展病例规模。若需重建 500 例实验，请按 `docs/AGENT_PIPELINE_HANDOVER.md` 与 `scripts/prepare_ctclip_stage2_train_manifest.py` 的流程执行。

## 6. Stage-2 QLoRA adapter

最终答辩系统还需要本项目训练得到的 adapter。它不是公开基座模型，仓库中也不包含其二进制权重。

预期目录：

```text
artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep/adapter/
```

获得方式二选一：

1. 从项目的授权服务器或已有备份复制该 adapter 目录；或
2. 使用已经准备的 500 例数据和 CT-CLIP 分数，依次运行 `scripts/build_ctclip_stage2_sft.py` 与 `scripts/train_llm_qlora.py` 重新训练。训练参数、数据划分和服务器命令见 [LLM_QLORA_TRAINING.md](LLM_QLORA_TRAINING.md) 与 [AGENT_PIPELINE_HANDOVER.md](AGENT_PIPELINE_HANDOVER.md)。

只下载 Qwen 基座模型并不足以运行最终 Stage-2 Demo；缺少 adapter 时，`run_stage2_agent.py` 会在启动前明确报告路径缺失。

## 7. 下载前检查与安全规则

```bash
hf auth login
hf auth whoami
```

- token 只能存放在用户本地凭据管理器或未提交的 `.env`；不得写入源码、脚本、截图或 Issue。
- 不要在本仓库中执行 `git add -f models data dataset artifacts`。
- Demo 默认使用本地已下载资产；设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 后可验证离线可运行性。
- 如果路径不在默认位置，使用 `scripts/run_stage2_agent.py` 的 `--model-dir`、`--adapter-dir`、`--ctclip-checkpoint`、`--ctclip-source`、`--text-model-dir` 参数指定实际路径。
