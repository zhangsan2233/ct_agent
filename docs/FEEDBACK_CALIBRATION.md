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

服务器可每日调用 `scripts/run_feedback_maintenance.sh`。它仅刷新候选报告；即使达到门槛，也仍需固定回归集评估与人工批准，才可应用任何阈值或启动候选 QLoRA 训练。

若推理容器未运行 cron/systemd，可将 `scripts/run_feedback_maintenance_loop.sh` 作为后台进程启动；它每 24 小时调用同一安全脚本，并把标准输出重定向到 `logs/feedback_maintenance.log`。容器重启后应由部署脚本重新启动该循环。

## 构建候选 QLoRA 数据

```bash
python scripts/build_feedback_sft.py \
  --db artifacts/memory/agent_memory.sqlite3 \
  --out-dir artifacts/feedback/sft_candidate_YYYYMMDD \
  --modality ct_chest
```

只会读取 `approved` 反馈，并从受控病例上下文恢复报告和完整影像分数（CT：`ct_model`；CXR：`cxr_model`）；`model_version` 须带模态前缀。

## 服务器阶段

当审核通过反馈达到门槛后，在受控服务器执行：候选阈值校准、固定回归集评估、人工批准发布。只有反馈覆盖、JSON 有效率和回归指标均满足门槛时，才可生成候选 QLoRA adapter；禁止自动发布或覆盖当前模型版本。
