# ChestCT-Agent 完整交接文档

交接日期：2026-08-23  
仓库：[zhangsan2233/ct_agent](https://github.com/zhangsan2233/ct_agent)  
代码基线：本地 `main` 位于 `9ec5b7f`，相对当时 `origin/main` ahead 1。  
定位：胸部 CT 证据整合的课程/科研原型，**不用于临床诊断、筛查或治疗决策**。

> 本文只记录代码、受控资产来源与非敏感服务器路径；不记录服务器地址、密码、token、CT 图像、报告正文或可识别信息。访问凭据必须通过安全渠道单独移交。任何曾暴露于聊天或终端的 token 应在交接前轮换。

## 1. 项目交付范围

项目有两条能力线，答辩应优先使用最终 Stage-2 管线，不能把两条线的指标混合表述。

| 线 | 用途 | 推荐入口 | 状态 |
| --- | --- | --- | --- |
| 最终 Stage-2 | CT + 报告 -> CT-CLIP 证据 -> Qwen 结构化 JSON | `scripts/run_stage2_agent.py`、`demo/stage2_streamlit_app.py` | GPU 服务器端到端验收通过 |
| 全 Agent/RAG 原型 | API、LangGraph、RAG、RadGraph、归因、专科工具、反馈 | `chestct_agent/api/main.py`、`demo/streamlit_app.py` | 功能集成完成，部分工具依赖外部资产 |
| 多模态接口（封笔） | 胸部 CT 正式 + 胸部 X 光示意全链路 | `demo/multimodal_app.py`、`chestct_agent/modalities.py` | CT 用 CT-CLIP + 正式 Stage-2 adapter；CXR 用 TorchXRayVision 映射 8 分 + **平行** `qwen3_5_9b_cxr_stage2` adapter |

最终 Stage-2 数据流：

```text
CT NIfTI (.nii/.nii.gz) + 报告文本
  -> 冻结 CT-CLIP_v2：8 项影像证据分数
  -> Qwen3.5-9B + Stage-2 QLoRA adapter
  -> 严格 JSON、中文解释、人工复核提示
  -> JSON/schema/CT 证据完整性与数值一致性校验、result.json
```

Stage-2 标签：`arterial_wall_calcification`、`atelectasis`、`coronary_artery_wall_calcification`、`emphysema`、`lung_opacity`、`lymphadenopathy`、`pulmonary_fibrotic_sequela`、`pulmonary_nodule`。

Qwen 不读取完整三维 CT；CT 图像证据由冻结 CT-CLIP 产生。Qwen 的职责是融合报告与分数、输出可解析且可审计的结构化结论。

全 Agent/RAG 原型还包括：FastAPI 上传/流式 API、会话与多轮对话、工具白名单和重试降级、BM25 + local embedding/reranker + Qdrant、RadGraph、相似病例、CT attribution、TotalSegmentator 工具与反馈工作流。缺失依赖时应在返回结果中明确标示降级，而不能伪装成完整模型推理。

## 2. 已完成改动与当前 Git 状态

| 提交 | 内容 |
| --- | --- |
| `25e4577` | 支持本地 Qwen3.5 Stage-2 QLoRA 后端。 |
| `fe2c91f` | 增加 Stage-2 adapter 合并为独立模型的脚本。 |
| `e5b32db`、`7d912d7` | 优先本地 CXR-BERT；Stage-2 锁定 CT-CLIP_v2 zero-shot。 |
| `dc49b36` | 100 例展示/消融评测结果。 |
| `9dd4b6a` 至 `3e4dce6` | 反馈存储、审核、候选校准、SFT 构建和维护循环。 |
| `8d4eb9e`、`ea014cb` | CT attribution 与 specialist tools。 |
| `9ec5b7f` | 模拟反馈、冻结评估、候选 adapter 回归门禁。 |

本地还有未提交的 `deliverables/`，其中的 `weekly_report_20260727` 是可公开的汇报包与 zip。它不含模型和受控数据；接手者应审阅后自行决定是否提交。

## 3. 仓库、模型和数据资产

### 仓库已有

| 内容 | 路径 |
| --- | --- |
| 核心实现 | `chestct_agent/` |
| Demo | `demo/` |
| 数据/训练/推理/评估脚本 | `scripts/` |
| 500 例复现元数据 | `reproducibility/` |
| Stage-1/Stage-2 LoRA adapter | `artifacts/llm_qlora/.../adapter/`（Git LFS） |
| 文本分类器 | `artifacts/text_classifier.joblib` |

克隆时必须取得 LFS 权重：

```bash
git lfs install
git clone https://github.com/zhangsan2233/ct_agent.git
cd ct_agent
git lfs pull
git lfs ls-files
```

### 受控外部资产

| 资产 | 来源 | 预期路径 |
| --- | --- | --- |
| Qwen3.5-9B | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `models/Qwen3.5-9B/` |
| CT-CLIP 源码 | [CT-CLIP](https://github.com/ibrahimethemhamci/CT-CLIP) | `external/CT-CLIP-main/` |
| CT-CLIP_v2.pt | [CT-RATE 受控页面](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/models/CT-CLIP-Related/CT-CLIP_v2.pt) | `models/ctclip/CT-CLIP_v2.pt` |
| CXR-BERT | [BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized) | `models/cxrbert/` |
| CT-RATE | [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | `data/` 或 `dataset/` |

CT-CLIP 权重和 CT-RATE 均受访问控制。先完成官方条款/授权，再下载；禁止公开再分发。

严禁提交 `.env`、token、CT-RATE CT/DICOM/NIfTI、报告正文、完整基座模型、CT-CLIP/CXR-BERT、推理 JSON、缓存、日志、完整合并模型或含报告的 SFT JSONL。`.gitignore` 已设置此边界，但交接前仍须审阅 `git status --short`。

完整资产说明见 `docs/ASSET_SETUP.md` 和 `docs/REPOSITORY_CONTENTS.md`。

## 4. 服务器部署快照

凭据由管理员安全移交。已验证环境：

```text
项目根目录：/root/summer_zhl
Python：/root/summer_zhl/conda_env/bin/python
GPU：2 × NVIDIA RTX 4090（各 24 GB）
```

关键资产路径：

```text
/root/summer_zhl/models/ctclip/CT-CLIP_v2.pt
/root/summer_zhl/external/CT-CLIP-main
/root/summer_zhl/models/cxrbert
/root/summer_zhl/models/Qwen3.5-9B
/root/summer_zhl/artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep/adapter
/root/summer_zhl/data/train_fixed
```

已合并的完整 BF16 SafeTensors 模型：

```text
/root/summer_zhl/artifacts/merged_models/qwen3_5_9b_stage2_merged
```

该模型约 18 GB，未进 Git；迁移时仅通过受控文件传输复制或重新合并。

推荐使用物理 GPU 1：

```bash
export CUDA_VISIBLE_DEVICES=1
```

进程内继续使用 `cuda:0`。CT-CLIP 与 Qwen 不应同时常驻一张 24 GB GPU；最终管线先跑 CT-CLIP、释放显存、再加载 Qwen。冷启动单例约 2–3 分钟；空闲 GPU 历史验收约 101 秒。

服务器中的受控结果：

```text
/root/summer_zhl/artifacts/llm_eval/stage2_demo_100/
/root/summer_zhl/artifacts/feedback_simulation_20260724/
/root/summer_zhl/artifacts/acceptance/stage2_adapter/
/root/summer_zhl/artifacts/acceptance/stage2_merged/
/root/summer_zhl/logs/
```

这些目录可能含受控派生数据，不应直接公开上传。

## 5. 新环境最小启动

主环境：

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

本地 QLoRA 推理/训练额外安装：

```bash
pip install -r requirements-llm-train.txt
```

该依赖使用 Transformers Git main。迁移时记录 CUDA、PyTorch、驱动与 `pip freeze` 版本；无 GPU 机器不应启用 local-qlora。

最终管线关键环境变量：

```text
MODEL_BACKEND=local-qlora
LOCAL_LLM_MODEL_DIR=./models/Qwen3.5-9B
LOCAL_LLM_ADAPTER_DIR=./artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep/adapter
LOCAL_LLM_DEVICE=auto
LOCAL_LLM_LOAD_IN_4BIT=true
CTCLIP_VARIANT=zeroshot
CTCLIP_CHECKPOINT=./models/ctclip/CT-CLIP_v2.pt
CTCLIP_SOURCE_DIR=./external/CT-CLIP-main
CTCLIP_TEXT_MODEL_DIR=./models/cxrbert
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

`.env.example` 中历史上也出现过 `models/qwen3_5_9B/Qwen3.5-9B`，与上述目录写法不同。二者都能使用，但必须和真实磁盘路径一致；推荐统一为 `models/Qwen3.5-9B`。

基础检查：

```bash
python scripts/run_stage2_agent.py --help
python -m pytest tests/test_feedback_sft.py
```

若无 pytest，安装 `pip install pytest`；不影响本体运行。

## 6. 最终 Stage-2 运行和验收

### 单病例 CLI

```bash
cd /root/summer_zhl
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

/root/summer_zhl/conda_env/bin/python scripts/run_stage2_agent.py single \
  --case-id demo_case \
  --ct /absolute/path/to/demo_case.nii.gz \
  --report-file /absolute/path/to/report.txt \
  --runs-dir artifacts/agent_runs
```

也可用 `--report "..."` 传入短报告。不要将真实报告粘贴进 Git、公开 issue 或截图。

输出位置：

```text
artifacts/agent_runs/<timestamp>_<case_id>/result.json
```

验收必须核对：`ctclip_scores` 有 8 个数值、`stage2_json` 非空、`validation.parseable_json=true`、`validation.schema_valid=true`、`validation.errors` 为空，且 `provenance` 指向预期模型/adapter/CT-CLIP。若失败，查看 `raw_stage2_output`，不可手工修补后冒充模型输出。

### 批处理

CSV 必须包含 `case_id`、`ct_path` 和 `report_text` 或 `report_path`：

```bash
python scripts/run_stage2_agent.py batch \
  --manifest /absolute/path/to/batch_manifest.csv \
  --runs-dir artifacts/agent_runs \
  --out artifacts/agent_runs/batch_results.jsonl
```

逐例落盘；失败不会清除之前结果。该策略优先稳定，不追求吞吐。

### Streamlit 答辩页面

```bash
cd /root/summer_zhl
export CUDA_VISIBLE_DEVICES=1
/root/summer_zhl/conda_env/bin/streamlit run demo/stage2_streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8501
```

请用 SSH 隧道访问，不要公网开放：

```bash
ssh -L 8501:127.0.0.1:8501 <authorized-user>@<server>
```

页面支持上传 NIfTI、输入/上传报告、展示 CT-CLIP 分数、Stage-2 JSON、中文完整报告与 8 标签纠错面板，并下载本次 `result.json`。

### 多模态接口（胸部 X 光封笔）

```bash
python scripts/init_cxr_adapter.py
pip install torchxrayvision && python scripts/fetch_cxr_encoder.py
python scripts/run_modality_agent.py --modality cxr_chest --image /path/cxr.png --report "..." --case-id cxr_demo
streamlit run demo/multimodal_app.py --server.address 127.0.0.1 --server.port 8502
```

CXR 平行 adapter：`artifacts/llm_qlora/qwen3_5_9b_cxr_stage2/adapter/`（不进 Git）。反馈按 `model_version` 前缀 `cxr_chest:` 隔离；`scripts/compare_modality_adapter_gate.py` 在冻结集 micro-F1 下降时拒绝发布。

### 全 Agent FastAPI 原型

```bash
uvicorn chestct_agent.api.main:app --host 127.0.0.1 --port 8080
streamlit run demo/streamlit_app.py
```

核心 API：

| 端点 | 作用 |
| --- | --- |
| `GET /health` | 健康检查 |
| `POST /api/analyze` | 文本 JSON 分析 |
| `POST /api/analyze/upload` | CT + 报告 multipart 分析 |
| `POST /api/analyze/upload/stream` | 流式上传分析 |
| `POST /api/chat`、`/api/chat/stream` | `session_id` 多轮问答 |
| `POST /api/cases/{case_id}/feedback` | 反馈提交 |
| `GET /api/feedback` | 反馈队列 |
| `POST /api/feedback/{event_id}/review` | 批准或拒绝 |

默认 SQLite：`artifacts/memory/agent_memory.sqlite3`，应按受控数据保存。

## 7. 训练和完整模型

已完成训练：Qwen3.5-9B 基座，Stage-1 报告结构化 adapter，Stage-2 在 Stage-1 上继续训练，融合 CT-CLIP 8 项证据。训练数据为 500 例受控 CT-RATE 子集，患者级 450/50 切分。关键参数：2 epochs、NF4 4-bit、rank 16、alpha 32、dropout 0.05、学习率 `5e-5`、batch 1、累积 16、最大长度 2048。CT-CLIP 始终冻结，未微调。标签为报告派生弱标签，而非影像专家金标准。

授权后重建顺序：

1. 依据 `reproducibility/ctrate_500_weak_label_manifest.csv` 获取 500 例；
2. `scripts/batch_ctclip_infer.py` 生成私有 CT-CLIP 输出；
3. `scripts/build_ctclip_stage2_sft.py` 生成私有 SFT；
4. `scripts/train_llm_qlora.py` 训练候选 Stage-2 adapter；
5. 在患者独立留出集评估后再考虑发布。

详细命令见 `reproducibility/REBUILD.md`。历史文档中出现的 `train_qlora.py` 是旧名称/笔误，当前实际文件是 `scripts/train_llm_qlora.py`。

如需完整模型（非日常演示必需）：

```bash
python scripts/merge_stage2_adapter.py \
  --model-dir models/Qwen3.5-9B \
  --adapter-dir artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep/adapter \
  --out-dir artifacts/merged_models/qwen3_5_9b_stage2_merged
```

合并后以 `scripts/verify_merged_stage2_model.py` 验证。完整模型约 18 GB，禁止 GitHub 上传。

## 8. 实验结果与答辩口径

### 100 例 Stage-2 展示/消融

该集合用于展示覆盖与接口稳定性；原始 50 例才是独立留出集，不能把 100 例整体称为独立泛化性能。

| 设置 | JSON 有效率 | 弱标签 Micro-F1 | CT 证据覆盖/一致性 |
| --- | ---: | ---: | ---: |
| Stage-1 仅报告 | 100% | — | — |
| Stage-2 去 CT 字段 | 76% | 不适用（大量非 JSON） | 无 CT 输入 |
| Stage-2 报告 + CT-CLIP | 100% | 约 0.745–0.749 | 100% / 100% |

可表述为：CT 证据字段有助于 Stage-2 保持结构化输出和字段一致性。不能表述为临床准确率、外部验证或 CT-only 优于报告。

### 早期 CT/报告/融合评估

在 48 校准 / 32 独立测试患者的小样本弱标签集上：CT-only selective Micro-F1 0.398；Report-only 0.945；CT+report 0.905。由于标签源自报告，报告表现高不代表临床影像诊断能力。

### 模拟反馈闭环（2026-07-24）

100 例中按患者分为反馈 60、冻结 40，零重叠。使用隐藏 CT-RATE 报告弱标签模拟反馈，产生 58 条事件（批准 54、拒绝 4），形成 29 条候选 SFT（训练 23、验证 6），以 1 epoch、6 steps 训练候选 adapter。

| 指标 | 正式 Stage-2 | 候选 adapter | 变化 |
| --- | ---: | ---: | ---: |
| Precision | 0.8788 | 0.7750 | -0.1038 |
| Recall | 0.6824 | 0.7294 | +0.0471 |
| Micro-F1 | 0.7682 | 0.7515 | -0.0167 |
| JSON 有效率 | 100% | 100% | 0 |
| CT 证据覆盖/一致性 | 100% / 100% | 100% / 100% | 0 |

候选 TP +4、FN -4，但 FP +10。因此结论是：**候选 adapter 已被拒绝，未部署，正式 Stage-2 adapter 未被覆盖。** 这验证了回归门禁，而不是医学性能提升。模拟标签并非真实人工反馈，不能用于临床校准声明。

聚合结果包：`deliverables/weekly_report_20260727/`。

## 9. 反馈、校准和再训练机制

工作流：

```text
提交反馈 -> SQLite 保存快照/pending
-> 审核者 approved/rejected
-> 候选校准统计（dry-run）
-> 仅 approved 构建候选 SFT
-> 患者独立冻结集回归评估
-> 人工审核
-> 通过才允许发布
```

初始化与候选统计：

```bash
python scripts/initialize_feedback_store.py --db artifacts/memory/agent_memory.sqlite3
python scripts/run_feedback_calibration.py \
  --db artifacts/memory/agent_memory.sqlite3 \
  --out artifacts/feedback/candidate_calibration.json \
  --minimum-approved 50
python scripts/build_feedback_sft.py \
  --db artifacts/memory/agent_memory.sqlite3 \
  --out-dir artifacts/feedback/sft_candidate_YYYYMMDD
```

`run_feedback_calibration.py`、`run_feedback_maintenance.sh` 和循环维护脚本都不得自动改线上阈值、adapter 或完整模型。模拟脚本 `simulate_weak_label_feedback.py`、`prepare_simulation_frozen_eval.py` 仅用于课程功能验证，产出须标注为非临床模拟。

真实更新的最低要求：合格审核者反馈、患者级隔离、类别平衡、混入原始 SFT、防遗忘训练、固定冻结集不退化、人工批准后发布。

## 10. 常见问题

| 现象 | 处理 |
| --- | --- |
| 路径找不到 | 使用 `run_stage2_agent.py` 的 `--model-dir`、`--adapter-dir`、`--ctclip-checkpoint`、`--ctclip-source`、`--text-model-dir` 显式覆盖。 |
| CT-CLIP 想联网 | 核对 CXR-BERT、`CTCLIP_TEXT_MODEL_DIR`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。 |
| CUDA OOM | `nvidia-smi` 后释放 GPU；只运行一个完整管线；推荐 `CUDA_VISIBLE_DEVICES=1`。 |
| JSON 校验失败 | 查看 `raw_stage2_output` 和 `validation.errors`，重跑，不伪造。 |
| Streamlit 访问失败 | 保持服务仅绑定 `127.0.0.1`，通过 SSH 隧道访问，不暴露公网。 |
| adapter 是小文本 | 未正确 LFS 拉取，执行 `git lfs pull`。 |
| RAG/RadGraph 失败 | 读取响应中的 `degraded`/fallback 标记，不将其说成完整模型运行。 |

## 11. 建议交接验收顺序

1. 安全取得服务器访问，轮换旧 token/密码，确认仓库无敏感未提交文件。
2. `git lfs pull`，确认两个 adapter 不是 LFS 指针。
3. 核对 Qwen、CT-CLIP_v2、CT-CLIP 源码、CXR-BERT 和 adapter 路径。
4. 用一个受控病例完成单例 CLI，检查 `result.json` 的 schema、证据和 provenance。
5. 经 SSH 隧道运行 Streamlit；确认不开放公网端口。
6. 阅读 `docs/FINAL_STAGE_REPORT_20260719.md`、`docs/FEEDBACK_CALIBRATION.md`、`reproducibility/REBUILD.md` 和本地周报包。
7. 答辩前冻结正式 adapter，不在最后阶段重训或替换。

## 12. 文档导航与历史差异

| 文档 | 用途 |
| --- | --- |
| `README.md` | 总览、安装、API、完整 Agent 说明。 |
| `docs/ASSET_SETUP.md` | 外部资产下载与放置路径。 |
| `docs/AGENT_PIPELINE_HANDOVER.md` | 最终 Stage-2 的短操作版。 |
| `docs/FEEDBACK_CALIBRATION.md` | 反馈 API 与安全维护。 |
| `docs/FINAL_STAGE_REPORT_20260719.md` | 阶段结果与合并模型记录。 |
| `reproducibility/REBUILD.md` | 授权后重建训练的流程。 |

已知差异：正式脚本为 `train_llm_qlora.py`；基座模型目录有两种历史写法，须显式统一；早期文档曾称 adapter 不随仓库，但当前实际由 Git LFS 提供，应以 `git lfs ls-files` 和真实文件大小为准；18 标签全 Agent 与 8 标签 Stage-2 是不同实验层级，不能拼接指标。

## 13. 最终签核清单

- [ ] 接手者已获得安全访问方式，凭据未写入代码或文档。
- [ ] LFS adapter 已可加载，外部模型/数据路径均已核对。
- [ ] 已完成一例通过 schema 的 Stage-2 CLI 推理。
- [ ] Streamlit 已通过 SSH 隧道验证，服务未公开暴露。
- [ ] 已确认反馈候选未部署，正式 adapter 未覆盖。
- [ ] 已清楚弱标签、小样本和非临床使用边界。
- [ ] 所有新增 Git 文件均不含 CT、报告、密钥、日志或完整模型。

满足以上项目后，即可由新成员安全接手维护、复现和课程答辩演示。
