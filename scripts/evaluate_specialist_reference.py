from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


LABELS = [
    "pulmonary_nodule",
    "pleural_effusion",
    "pericardial_effusion",
]


def metrics(truth: list[int], predicted: list[int]) -> dict[str, float]:
    if not truth:
        return {}
    return {
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "accuracy": sum(a == b for a, b in zip(truth, predicted, strict=True)) / len(truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records", default="artifacts/evaluation/specialist_tool_records.jsonl"
    )
    parser.add_argument(
        "--reference", default="artifacts/evaluation/specialist_pilot_reference.csv"
    )
    parser.add_argument(
        "--out", default="artifacts/evaluation/specialist_report_audited_metrics.json"
    )
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in Path(args.records).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reference = pd.read_csv(args.reference).set_index("case_id")
    result = {
        "reference": "Manual audit of explicit positive/negative statements in CT-RATE reports.",
        "unknown_is_excluded": True,
        "per_label": {},
    }
    for label in LABELS:
        usable = [
            row
            for row in records
            if row["label"] == label
            and row["case_id"] in reference.index
            and reference.loc[row["case_id"], label] in {"positive", "negative"}
        ]
        truth = [int(reference.loc[row["case_id"], label] == "positive") for row in usable]
        ctclip = [int(row["ctclip_score"] >= 0.5) for row in usable]
        selective = [
            row for row in usable if row["tool_verdict"] in {"positive", "negative"}
        ]
        tool_truth = [
            int(reference.loc[row["case_id"], label] == "positive") for row in selective
        ]
        tool_prediction = [int(row["tool_verdict"] == "positive") for row in selective]
        result["per_label"][label] = {
            "audited_cases": len(usable),
            "positive_references": sum(truth),
            "ctclip_at_0_5": metrics(truth, ctclip),
            "tool_selective": {
                **metrics(tool_truth, tool_prediction),
                "coverage": len(selective) / len(usable),
                "decisions": len(selective),
            },
        }
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
