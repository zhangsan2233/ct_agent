# 人工反馈与候选校准流程

本模块保存反馈、审核状态和模型版本，但不会因为单次反馈自动修改推理结果、阈值或模型权重。

## API

1. 提交待审核反馈：`POST /api/cases/{case_id}/feedback`。
2. 查看队列：`GET /api/feedback?status=pending`。
3. 审核：`POST /api/feedback/{event_id}/review`，状态只能为 `approved` 或 `rejected`。

提交时必须带 `session_id`，系统会从已保存的病例上下文读取原模型标签并保存快照；未知标签、重复标签或不存在的病例上下文会被拒绝。

## 本地 dry-run

```bash
python scripts/run_feedback_calibration.py \
  --db artifacts/memory/agent_memory.sqlite3 \
  --out artifacts/feedback/candidate_calibration.json \
  --minimum-approved 50
```

该脚本仅统计审核通过的反馈、标签变更数量和是否达到服务器候选校准门槛；不会改变活动阈值、adapter 或完整模型。

首次部署时，先初始化审计库：

```bash
python scripts/initialize_feedback_store.py \
  --db artifacts/memory/agent_memory.sqlite3
```

## 构建候选 QLoRA 数据

```bash
python scripts/build_feedback_sft.py \
  --db artifacts/memory/agent_memory.sqlite3 \
  --out-dir artifacts/feedback/sft_candidate_YYYYMMDD
```

只会读取 `approved` 反馈，并从受控病例上下文恢复报告和完整 CT-CLIP 分数；缺少报告或 8 个 CT 分数的病例会被跳过。输出目录包含 `train.jsonl`、`valid.jsonl` 和不含报告正文的 `manifest.json`，必须保持在 Git 忽略目录中。

## 服务器阶段

当审核通过反馈达到门槛后，在受控服务器执行：候选阈值校准、固定回归集评估、人工批准发布。只有反馈覆盖、JSON 有效率和回归指标均满足门槛时，才可生成候选 QLoRA adapter；禁止自动发布或覆盖当前模型版本。
