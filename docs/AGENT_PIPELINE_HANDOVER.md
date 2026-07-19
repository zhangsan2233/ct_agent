# ChestCT-Agent Stage-2 管线交接文档

> 交接目标：在已配置模型的 GPU 服务器上，输入一个胸部 CT NIfTI 文件和一段报告文本，稳定产出 CT-CLIP 的 8 项影像证据、Stage-2 结构化 JSON、校验结果和可追溯的本地记录。该系统仅供课程/科研答辩演示，不能用于临床诊断。

## 1. 当前交付范围

本次交付完成的是最终答辩使用的 **冻结 CT-CLIP + Qwen3.5-9B Stage-2 adapter** 管线，而不是早期的 RAG 原型。

```text
CT (.nii/.nii.gz) + 报告文本
          │
          ├─ 冻结 CT-CLIP（官方 CT-CLIP_v2.pt）
          │       └─ 8 项 CT-CLIP 概率
          │
          └─ Qwen3.5-9B + Stage-2 QLoRA adapter
                  └─ 紧凑 8 标签 JSON
                         ├─ JSON 解析与 Schema 校验
                         ├─ CT 分数完整性/数值一致性校验
                         └─ 运行记录 result.json
```

已具备的入口：

| 用途 | 入口 |
| --- | --- |
| 单病例命令行 | `scripts/run_stage2_agent.py single` |
| 批处理命令行 | `scripts/run_stage2_agent.py batch` |
| 答辩页面 | `demo/stage2_streamlit_app.py` |
| 旧版兼容单例脚本 | `scripts/run_stage2_demo.py` |
| 核心实现 | `chestct_agent/stage2_pipeline.py` |

## 2. 已完成实验资产

### 数据与模型

- 数据规模固定为 500 例 CT-RATE 子集；450 例训练，50 例患者级不重叠留出评测。
- CT-CLIP：官方 `CT-CLIP_v2.pt`，本项目中**冻结使用，不进行微调**。
- 语言模型：Qwen3.5-9B + 训练完成的 Stage-2 QLoRA adapter。
- Stage-2 的 8 个标签：`arterial_wall_calcification`、`atelectasis`、`coronary_artery_wall_calcification`、`emphysema`、`lung_opacity`、`lymphadenopathy`、`pulmonary_fibrotic_sequela`、`pulmonary_nodule`。

### 留出集结果

- 三臂 JSON 有效率：Stage-1 仅报告 100%、Stage-2 去 CT 76%、Stage-2 加 CT-CLIP 100%。
- Stage-2 加 CT-CLIP 的弱标签 micro-F1 为 0.749。
- Stage-2 加 CT-CLIP 的 CT 证据字段覆盖率、数值一致性均为 100%。

这些指标是与报告弱标签的一致性/格式结果，不是临床诊断准确率。完整结果见 `artifacts/final_stage2_result_package_20260719/`。

## 3. 服务器目录与运行环境

服务器工程根目录：

```text
/root/summer_zhl
```

推荐 Python：

```text
/root/summer_zhl/conda_env/bin/python
```

答辩页面依赖 `streamlit`。当前服务器环境已安装；如迁移到新环境，执行：

```bash
/root/summer_zhl/conda_env/bin/python -m pip install 'streamlit>=1.36,<2'
```

模型和主要资产的预期路径：

```text
/root/summer_zhl/models/ctclip/CT-CLIP_v2.pt
/root/summer_zhl/external/CT-CLIP-main
/root/summer_zhl/models/cxrbert
/root/summer_zhl/models/Qwen3.5-9B
/root/summer_zhl/artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep/adapter
/root/summer_zhl/data/train_fixed
```

所有入口默认从项目根目录推断上述位置。若路径有变，可用命令行的 `--model-dir`、`--adapter-dir`、`--ctclip-checkpoint`、`--ctclip-source`、`--text-model-dir` 覆盖。

### GPU 约定

单例端到端运行已在服务器第二张 RTX 4090 验证通过。推荐把物理 GPU 1 映射为进程内 `cuda:0`：

```bash
export CUDA_VISIBLE_DEVICES=1
```

随后 CLI 的默认 `--device cuda:0` 就会使用该可见 GPU。不要让 CT-CLIP 与 Qwen 同时常驻同一张 24 GB GPU；核心管线会在 CT-CLIP 评分完成后主动释放显存，再加载 Qwen。

## 4. 单病例命令行运行

### 报告放在文本文件中（推荐）

```bash
cd /root/summer_zhl
export CUDA_VISIBLE_DEVICES=1

/root/summer_zhl/conda_env/bin/python scripts/run_stage2_agent.py single \
  --case-id demo_case \
  --ct /absolute/path/to/demo_case.nii.gz \
  --report-file /absolute/path/to/report.txt \
  --runs-dir artifacts/agent_runs
```

### 直接传入简短报告

```bash
/root/summer_zhl/conda_env/bin/python scripts/run_stage2_agent.py single \
  --case-id demo_case \
  --ct /absolute/path/to/demo_case.nii.gz \
  --report "Pulmonary nodule is suspected. No pleural effusion."
```

单例通常需要约 2–3 分钟：CT-CLIP 3D 预处理/推理约占前半段，Qwen + adapter 加载和生成占后半段。首次加载速度受磁盘缓存和 GPU 空闲情况影响。

## 5. 批处理运行

批处理 CSV 必须包含 `case_id`、`ct_path`，并提供 `report_text` 或 `report_path` 之一。

示例 `batch_manifest.csv`：

```csv
case_id,ct_path,report_text
case_001,/root/summer_zhl/data/train_fixed/train_2994/train_2994_a/train_2994_a_1.nii.gz,"Pulmonary opacity is present."
```

执行：

```bash
cd /root/summer_zhl
export CUDA_VISIBLE_DEVICES=1

/root/summer_zhl/conda_env/bin/python scripts/run_stage2_agent.py batch \
  --manifest /absolute/path/to/batch_manifest.csv \
  --runs-dir artifacts/agent_runs \
  --out artifacts/agent_runs/batch_results.jsonl
```

批处理逐行落盘：某个病例失败时，该行会写入 `ok:false` 和错误信息，不会中断此前成功结果。因为显存安全策略，每例都先运行并释放 CT-CLIP，再加载/运行 Stage-2；这优先保证稳定性，不追求最高吞吐量。

## 6. Streamlit 答辩页面

在服务器中运行：

```bash
cd /root/summer_zhl
export CUDA_VISIBLE_DEVICES=1

/root/summer_zhl/conda_env/bin/streamlit run demo/stage2_streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

页面支持：

1. 上传 `.nii` / `.nii.gz`，或输入服务器已有 CT 的绝对路径；
2. 粘贴报告或上传 `.txt` 报告；
3. 展示 8 项 CT-CLIP 分数柱状图和表格；
4. 展示 Stage-2 JSON；
5. 显示 JSON/Schema 校验状态；
6. 下载本次 `result.json`。

如需从本地浏览器访问远程页面，需使用服务器可达 IP 和端口；网络/防火墙策略由服务器管理员配置。不要把 Streamlit 对公网开放。

## 7. 输出、校验与追溯

每次成功单例运行会写入：

```text
artifacts/agent_runs/<时间戳>_<case_id>/result.json
```

关键字段：

| 字段 | 含义 |
| --- | --- |
| `input` | 病例 ID、CT 绝对路径、实际使用的报告文本 |
| `ctclip_scores` | 8 项冻结 CT-CLIP 概率 |
| `stage2_json` | 解析后的 Stage-2 JSON；无法解析时为 `null` |
| `raw_stage2_output` | 未修改的语言模型原始输出，用于排错 |
| `validation.parseable_json` | 是否可解析 JSON |
| `validation.schema_valid` | 是否满足 8 标签、分数范围、证据一致性和人工复核标志 |
| `validation.errors` | 失败原因，供重试/排错 |
| `provenance` | 模型、adapter、CT-CLIP 路径、设备与生成参数 |
| `elapsed_seconds` | 本次端到端耗时 |

校验不会悄悄篡改模型输出：若 JSON 无法解析或标签/CT 分数不符合约定，会保留原始输出并报告失败。答辩展示时应说明这是“显式容错和可追溯”，而不是自动伪造修复结果。

## 8. 验证记录

已经使用留出集病例 `train_2994_a_1` 完成端到端验证：

```text
CT：/root/summer_zhl/data/train_fixed/train_2994/train_2994_a/train_2994_a_1.nii.gz
结果：8 项 CT-CLIP 分数 + 可解析、Schema 合法的 Stage-2 JSON
端到端退出码：0
耗时：约 101 秒（GPU 空闲时）
```

本地保留的验证结果：`artifacts/demo/stage2_demo_train_2994_a_1.json`。

## 9. 常见问题排查

| 现象 | 原因与处理 |
| --- | --- |
| `CT volume not found` | 使用绝对路径；确认是 `.nii` 或 `.nii.gz`，且服务器上实际存在。 |
| `checkpoint/model/adapter not found` | 检查第 3 节资产路径，或用相应 `--...-dir` 参数覆盖。 |
| CUDA out of memory | 先 `nvidia-smi` 检查占用；优先设置 `CUDA_VISIBLE_DEVICES=1`；不要并发启动多个 Demo。 |
| CT-CLIP 尝试联网 | 确认 `models/cxrbert` 完整；入口已设置 `CTCLIP_TEXT_MODEL_DIR`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。 |
| Stage-2 JSON 校验失败 | 查看 `raw_stage2_output` 与 `validation.errors`；可重新运行该病例，不要手工修改后当作模型原始结果。 |
| Streamlit 无法访问 | 先在服务器本机确认 8501 端口监听，再检查 SSH 隧道或防火墙；不要直接暴露公网。 |

## 10. 安全边界与后续维护

1. 严禁将 Hugging Face token、服务器密码或任何密钥写入代码、结果包或文档；如曾在终端中使用过，应及时在相应平台轮换。
2. 不增加病例规模、不重训模型即可完成当前答辩管线。当前工作重点是稳定展示和结果可追溯。
3. 如果以后追求医学性能，应先取得放射科医师独立判读的影像金标准，再讨论 CT-CLIP 微调、外部测试、置信区间及统计检验。
4. 旧版 `demo/streamlit_app.py` 和 RAG 工作流仍保留作原型参考；答辩时应使用本文件定义的 `stage2_streamlit_app.py` 和 `run_stage2_agent.py`，以避免两条模型路径混淆。
