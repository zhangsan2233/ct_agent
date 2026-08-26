# 2D LLM 对照实验：负结果一页（答辩用）

**结论：主结果必须锁冻结 CT-CLIP；不要让 LLM 用 2D 切片覆盖 CT-CLIP 分数。**

## 实验 A：独立 2D 投票（基座 Qwen，无 Stage-2 adapter）

| 指标 | 数值 |
| --- | ---: |
| 可比槽位一致率 vs CT-CLIP | 70.0% |
| Cohen's κ | 0.36 |
| 弃权率 | 48% |
| vs 弱标签 κ | 0.15 |

## 实验 B：CLIP + 2D 审核（confirm/reject）

| 指标 | 数值 |
| --- | ---: |
| LLM 误判率（已判断槽） | 33.4% |
| CT-CLIP vs 弱标签准确率 | 68.75% |
| 采纳 LLM 翻转后 | 66.38% |
| **绝对变化** | **−2.38 pp** |
| 有益翻转 / 有害翻转 | 26 / 45 |

数据来源：`deliverables/llm_2d_agreement_20260823/metrics.json` 与 `deliverables/llm_2d_clip_audit_20260823/metrics.json`。弱标签非金标准；不构成临床结论。
