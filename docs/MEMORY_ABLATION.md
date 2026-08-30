# Feedback Memory Ablation

This experiment treats diagnostic Memory as an inspection instruction, not as patient
evidence. It compares the same frozen cases, images, Agent prompt, and imaging tools. The
only changed variable is whether an audited error Memory is retrieved for a second visual
review.

## Leakage controls

- Thresholds are selected on the 100-case feedback split only.
- The 50-case frozen split is not available to Memory generation or calibration.
- A Memory proposal needs an audited Memory ID, two current-case slice indices, visible
  evidence, and confidence of at least 0.75.
- A calibrated independent CT-CLIP score can veto a proposal in the opposite direction.
- Missing CT-CLIP probabilities are failures; they are never replaced with zero.

## PatchChestCT pilot result

The fixed 50-case, 9-label run contains 450 binary label decisions.

| Method | Micro-F1 | Macro-F1 | Wrong decisions |
| --- | ---: | ---: | ---: |
| Agent + tools | 0.4141 | 0.3012 | 150 |
| Agent + tools + Memory | 0.4173 | 0.3030 | 148 |

The calibrated Memory gate accepted two changes, both beneficial. Patient bootstrap
confidence intervals still include zero, so this is an exploratory engineering result rather
than evidence of statistically significant clinical improvement. Only two Memory groups were
usable; broader disease coverage requires more disjoint, reviewed feedback.

## Reproduction

First calibrate tool thresholds on feedback cases:

```bash
python scripts/calibrate_memory_tool_gate.py \
  --manifest feedback100.csv \
  --predictions ctclip_feedback_part0.jsonl ctclip_feedback_part1.jsonl \
  --labels consolidation \
  --out memory_tool_calibration.json
```

Then apply the locked calibration to the frozen A/B outputs:

```bash
python scripts/evaluate_memory_ablation.py \
  --baseline-predictions agent_tools_frozen50.jsonl \
  --memory-rechecks memory_rechecks_frozen50.jsonl \
  --calibration memory_tool_calibration.json \
  --labels arterial_wall_calcification atelectasis bronchiectasis consolidation \
    coronary_wall_calcification hiatal_hernia lung_opacity lymphadenopathy \
    pericardial_effusion \
  --out memory_ablation_metrics.json
```

Patient data, images, predictions, and Memory text artifacts remain local and are not committed.
