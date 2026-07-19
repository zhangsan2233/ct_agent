import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.labels import LABEL_IDS, SOURCE_COLUMN_TO_ID
from chestct_agent.tools.evidence_extractor import extract_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rule evidence against CT-RATE weak labels.")
    parser.add_argument(
        "--reports", default="data/dataset/radiology_text_reports/validation_reports.csv"
    )
    parser.add_argument(
        "--labels", default="data/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="artifacts/evaluation/report_evidence_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = pd.read_csv(args.reports).fillna("")
    labels = pd.read_csv(args.labels).fillna(0)
    frame = reports.merge(labels, on="VolumeName", how="inner")
    if args.limit > 0:
        frame = frame.head(args.limit)
    label_columns = {source: target for source, target in SOURCE_COLUMN_TO_ID.items() if source in frame}
    y_true = np.zeros((len(frame), len(LABEL_IDS)), dtype=int)
    y_pred = np.zeros_like(y_true)
    negative_evidence = uncertain_evidence = historical_evidence = 0
    label_index = {label: index for index, label in enumerate(LABEL_IDS)}
    for row_index, row in frame.iterrows():
        report = f"Findings: {row.get('Findings_EN', '')}\nImpression: {row.get('Impressions_EN', '')}"
        evidence = extract_evidence(report, LABEL_IDS)
        for source, label in label_columns.items():
            column = label_index[label]
            y_true[row_index, column] = int(float(row[source]) > 0)
            y_pred[row_index, column] = int(
                any(item.polarity == "positive" for item in evidence[label])
            )
        all_items = [item for values in evidence.values() for item in values]
        negative_evidence += sum(item.polarity == "negative" for item in all_items)
        uncertain_evidence += sum(item.polarity == "uncertain" for item in all_items)
        historical_evidence += sum(item.polarity == "historical" for item in all_items)

    per_label = {}
    for label, column in label_index.items():
        per_label[label] = {
            "positive_count": int(y_true[:, column].sum()),
            "precision": float(
                precision_score(y_true[:, column], y_pred[:, column], zero_division=0)
            ),
            "recall": float(recall_score(y_true[:, column], y_pred[:, column], zero_division=0)),
            "f1": float(f1_score(y_true[:, column], y_pred[:, column], zero_division=0)),
        }
    result = {
        "cases": len(frame),
        "reference": "CT-RATE report-derived weak labels; not manually annotated evidence spans",
        "positive_evidence_proxy": {
            "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
            "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        },
        "polarity_counts": {
            "negative": negative_evidence,
            "uncertain": uncertain_evidence,
            "historical": historical_evidence,
        },
        "per_label": per_label,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_label"}, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
