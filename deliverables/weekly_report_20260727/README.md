# ChestCT-Agent 本周交接包（2026-07-27）

本目录是不含受控医疗数据、原始报告、模型权重、密钥和运行日志的交接包。

- `WEEKLY_REPORT_20260727.md`：本周改动、实验过程、指标、结论与后续建议。
- `results/feedback_simulation_summary.json`：可机器读取的模拟反馈验收结果。
- `results/frozen_baseline_metrics.json`：原正式 Stage-2 adapter 的冻结集指标。
- `results/frozen_candidate_metrics.json`：反馈候选 adapter 的冻结集指标。
- `results/RESULTS_MANIFEST.json`：文件说明、数据边界和复现定位信息。

## 重要边界

本包中的“反馈”由隐藏的 CT-RATE 报告派生弱标签模拟产生，仅验证工程闭环，**不构成真实人工反馈或临床性能结论**。候选 adapter 在冻结集微平均 F1 上退化，未被部署；当前正式推理仍使用原 Stage-2 adapter。

完整代码仓库：<https://github.com/zhangsan2233/ct_agent>

受控资产的下载来源、放置路径和复现命令见仓库中的 `docs/ASSET_SETUP.md` 与 `reproducibility/REBUILD.md`。
