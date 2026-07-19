# ChestCT-Agent 最终阶段报告

日期：2026-07-19
定位：课程/科研演示原型，不用于临床诊断。

## 1. 阶段结论

项目已完成从受控 CT 与报告输入，到冻结 CT-CLIP 证据、Qwen Stage-2 结构化输出、可审计 Agent 工作流和本地完整模型导出的闭环。最终 Stage-2 QLoRA adapter 已成功与 Qwen3.5-9B 基础模型合并为可独立加载的 BF16 SafeTensors 模型；固定病例的 adapter 链路和合并模型链路均已通过 JSON 结构校验。

## 2. 已完成的系统能力

```text
CT NIfTI / DICOM ZIP + 报告
  -> 安全导入、预处理与病例隔离
  -> 冻结 CT-CLIP 证据（18 标签运行时；Stage-2 使用其中 8 标签）
  -> 报告解析 / RadGraph 证据 / RAG / 相似病例 / 一致性检查
  -> Qwen 结构化输出、中文解释、审计轨迹与人工纠错接口
```

主要组成如下：

| 模块 | 当前状态 |
| --- | --- |
| 多模态 Agent | 已有 LangGraph 动态工具规划、白名单、重试/降级和输出校验 |
| 输入与 Demo | 已支持 NIfTI、DICOM ZIP、报告文本；含 FastAPI 与 Streamlit 界面 |
| CT 证据 | 官方 CT-CLIP_v2 冻结使用；本地 CXR-BERT 离线加载 |
| Stage-1/Stage-2 微调 | 已完成 Qwen3.5-9B QLoRA；Stage-2 融合 CT-CLIP 8 标签证据 |
| RAG 与解释 | 混合检索、相似病例、证据轨迹、RadGraph/RadGenome 设计已接入 |
| 复现材料 | 已提供无 CT 图像、无报告正文的 500 例清单、固定切分、参数、SHA256 与重建步骤 |

## 3. 训练与模型资产

### 3.1 QLoRA 训练

- 基础模型：Qwen3.5-9B。
- Stage-1：报告到结构化 JSON 的 LoRA adapter。
- Stage-2：在 Stage-1 基础上继续训练，学习将冻结 CT-CLIP 分数与报告信息融合为严格 JSON。
- Stage-2 训练数据：500 例受控 CT-RATE 子集，患者级 450/50 划分。
- 关键参数：2 epochs、4-bit NF4、LoRA rank 16、alpha 32、dropout 0.05、学习率 `5e-5`、batch size 1、gradient accumulation 16、最大长度 2048。
- 训练和验证标签为 CT-RATE 报告派生弱标签，不是放射科专家裁定的 CT 金标准。

### 3.2 完整模型导出

服务器已使用双 RTX 4090 将基础模型与最终 Stage-2 adapter 以 BF16 合并。合并时不使用 4-bit 权重，避免将 LoRA 合并到量化权重造成不可靠导出。

| 项目 | 值 |
| --- | --- |
| 基础模型路径 | `/root/summer_zhl/models/Qwen3.5-9B` |
| adapter 路径 | `/root/summer_zhl/artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep/adapter` |
| 完整模型路径 | `/root/summer_zhl/artifacts/merged_models/qwen3_5_9b_stage2_merged` |
| 格式 | BF16、SafeTensors、4 个分片 |
| 输出大小 | 约 18GB |
| 溯源文件 | `merge_provenance.json` |

导出脚本：[merge_stage2_adapter.py](../scripts/merge_stage2_adapter.py)。该完整模型仅存于服务器，不提交 GitHub。

## 4. 服务器端端到端验收

验收服务器为 2 × NVIDIA RTX 4090（各 24GB）。固定病例选择为 `train_7773_a_1`；本报告不包含其 CT 内容或报告正文。

### 4.1 Adapter 原始链路

```text
CT + 受控报告
-> 冻结 CT-CLIP_v2（zero-shot）
-> Qwen3.5-9B + Stage-2 adapter（4-bit 推理）
-> Stage-2 JSON
```

- 状态：成功。
- 耗时：118.122 秒。
- 结果：`/root/summer_zhl/artifacts/acceptance/stage2_adapter/train_7773_a_1/result.json`。
- 日志：`/root/summer_zhl/artifacts/logs/acceptance_stage2_adapter.log`。

### 4.2 合并完整模型链路

使用上一次 CT-CLIP 验收产生的同一份证据，对独立加载的完整模型进行生成验证：

- 状态：成功。
- 模型标识：`merged_qwen3_5_9b_stage2`。
- JSON 可解析：是。
- 8 标签 schema 校验：通过。
- 校验错误：无。
- 结果：`/root/summer_zhl/artifacts/acceptance/stage2_merged/train_7773_a_1/result.json`。
- 日志：`/root/summer_zhl/artifacts/logs/acceptance_stage2_merged.log`。

## 5. 本轮修复

在 PR 集成与服务器验收中识别并修复两项兼容问题：

1. `CtClipRuntime` 现在优先使用 `CTCLIP_TEXT_MODEL_DIR` 指定的本地 CXR-BERT；未指定时会自动发现与 CT-CLIP 同级的 `models/cxrbert`。这避免了离线环境错误访问 Hugging Face。
2. Stage-2 证据管线显式锁定 `CT-CLIP_v2` 的 `zeroshot` 变体。PR 中的 LiPro 是独立对照分类器头，不能直接加载此零样本权重。

## 6. 复现与仓库边界

仓库已公开源代码、两份项目训练得到的 LoRA adapter、文本分类器和安全复现元数据。以下内容不上传：Qwen 基础模型、CT-CLIP/CXR-BERT、CT-RATE 数据、NIfTI、报告正文、推理 JSON、日志、完整合并模型和任何密钥。

复现入口：

- [ASSET_SETUP.md](ASSET_SETUP.md)：外部受控资产来源与放置位置。
- [REBUILD.md](../reproducibility/REBUILD.md)：500 例清单、切分、SHA256 与授权后本地重建流程。
- [AGENT_PIPELINE_HANDOVER.md](AGENT_PIPELINE_HANDOVER.md)：历史实验与管线交接说明。

## 7. 当前边界与下一步

- 目前评估使用弱监督标签和有限规模留出集，不能作为临床性能结论。
- PR 的 18 标签 Agent 运行时与 Stage-2 的 8 标签微调输出属于不同实验层级；答辩时应分别说明，不应混合指标。
- 已完成 CLI/JSON 端到端验收；后续可基于相同固定病例在 Streamlit 界面生成演示截图。
- 若要部署完整模型，可用合并目录直接通过 Transformers 加载；若显存有限，继续使用基础模型 + 4-bit adapter 的方式更合适。
