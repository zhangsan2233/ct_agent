"""Create paired held-out evaluation inputs for Stage-1/Stage-2 ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = [
    "arterial_wall_calcification", "atelectasis", "coronary_artery_wall_calcification",
    "emphysema", "lung_opacity", "lymphadenopathy", "pulmonary_fibrotic_sequela",
    "pulmonary_nodule",
]

STAGE1_SYSTEM = (
    "You are ChestCT-Agent, a research-only chest CT evidence integrator. Return JSON only. "
    "Ground every statement in supplied evidence. If CT evidence is unavailable or conflicts with report evidence, "
    "preserve uncertainty and require human review."
)
STAGE2_SYSTEM = (
    "You are ChestCT-Agent, a research-only chest CT evidence integrator. Return compact JSON only. "
    "Use only the supplied report impression and CT-CLIP scores; do not invent findings. Preserve uncertainty "
    "and require human review because labels are weak supervision."
)


def record(messages: list[dict], truth: dict[str, int], arm: str) -> dict:
    return {
        "messages": messages,
        "ground_truth": truth,
        "evaluation_labels": LABELS,
        "metadata": {"case_id": json.loads(messages[1]["content"])["case_id"], "arm": arm},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-valid", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    raw = [json.loads(line) for line in args.stage2_valid.read_text(encoding="utf-8").splitlines() if line.strip()]
    arms = {"stage1_report_only": [], "stage2_report_only": [], "stage2_with_ctclip": []}
    for item in raw:
        compact_input = json.loads(item["messages"][1]["content"])
        target = json.loads(item["messages"][2]["content"])
        truth = {label["name"]: int(label["status"] == "positive") for label in target["labels"]}
        report_scores = {label: 0.98 if truth[label] else 0.02 for label in LABELS}
        stage1_input = {
            "case_id": compact_input["case_id"],
            "question": "Please integrate the supplied evidence into the required JSON. Do not invent CT findings.",
            "report_text": compact_input["report_impression"],
            "report_model_scores": report_scores,
            "ct_model_scores": None,
            "ct_scores_available": False,
            "label_provenance": compact_input["label_provenance"],
        }
        stage2_without_ct = dict(compact_input)
        stage2_without_ct["ctclip_scores"] = None
        stage2_without_ct["ctclip_score_definition"] = "No CT-CLIP evidence supplied for this ablation arm"
        arms["stage1_report_only"].append(record(
            [{"role": "system", "content": STAGE1_SYSTEM},
             {"role": "user", "content": json.dumps(stage1_input, ensure_ascii=False)}], truth, "stage1_report_only"))
        arms["stage2_report_only"].append(record(
            [{"role": "system", "content": STAGE2_SYSTEM},
             {"role": "user", "content": json.dumps(stage2_without_ct, ensure_ascii=False)}], truth, "stage2_report_only"))
        arms["stage2_with_ctclip"].append(record(
            item["messages"][:2], truth, "stage2_with_ctclip"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in arms.items():
        (args.out_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    print(json.dumps({name: len(rows) for name, rows in arms.items()}))


if __name__ == "__main__":
    main()
