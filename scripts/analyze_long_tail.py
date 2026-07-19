import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.evaluation import patient_id_from_case_id
from chestct_agent.labels import LABEL_IDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CT-RATE long-tail labels and export weights.")
    parser.add_argument("--case-index", default="artifacts/prepared/case_index.csv")
    parser.add_argument("--out", default="artifacts/evaluation/long_tail_profile.json")
    parser.add_argument("--beta", type=float, default=0.9999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.case_index).fillna("")
    frame["patient_id"] = frame["case_id"].astype(str).map(patient_id_from_case_id)
    patient_labels = frame.groupby("patient_id")["labels"].apply(
        lambda values: {
            label
            for value in values
            for label in str(value).split(";")
            if label
        }
    )
    total_patients = len(patient_labels)
    rows: dict[str, dict[str, object]] = {}
    effective_values: dict[str, float] = {}
    for label in LABEL_IDS:
        positives = sum(label in labels for labels in patient_labels)
        negatives = total_patients - positives
        prevalence = positives / max(total_patients, 1)
        effective = (1.0 - args.beta) / max(1.0 - args.beta ** max(positives, 1), 1e-12)
        effective_values[label] = effective
        if prevalence >= 0.20:
            tier = "head"
        elif prevalence >= 0.05:
            tier = "mid"
        else:
            tier = "tail"
        rows[label] = {
            "positive_patients": positives,
            "negative_patients": negatives,
            "prevalence": prevalence,
            "tier": tier,
            "bce_pos_weight": min(negatives / max(positives, 1), 50.0),
        }

    mean_effective = sum(effective_values.values()) / len(effective_values)
    for label, value in effective_values.items():
        rows[label]["effective_number_weight"] = value / max(mean_effective, 1e-12)

    payload = {
        "source": "CT-RATE report-derived weak labels aggregated by patient",
        "patients": total_patients,
        "beta": args.beta,
        "recommended_training": {
            "loss": "BCEWithLogitsLoss with per-label bce_pos_weight; compare focal/asymmetric loss",
            "sampling": "patient-level sampling; do not duplicate reconstructions across splits",
            "reporting": "macro-F1, macro-AUPRC, per-label metrics, and head/mid/tail macro averages",
        },
        "labels": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tiers = pd.Series([item["tier"] for item in rows.values()]).value_counts().to_dict()
    print(f"Wrote {output}: patients={total_patients}, tiers={tiers}")


if __name__ == "__main__":
    main()
