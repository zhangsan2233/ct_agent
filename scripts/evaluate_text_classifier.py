import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from prepare_dataset import LABEL_NAME_MAP, _report_text_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate report text classifier on CT-RATE validation.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default="artifacts/text_classifier.joblib")
    parser.add_argument("--out", default="artifacts/evaluation/text_classifier_metrics.json")
    parser.add_argument("--threshold", type=float, default=0.35)
    return parser.parse_args()


def safe_metric(fn, y_true, y_score):
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    reports_path = data_dir / "dataset/radiology_text_reports/validation_reports.csv"
    labels_path = data_dir / "dataset/multi_abnormality_labels/valid_predicted_labels.csv"

    reports = pd.read_csv(reports_path)
    labels_df = pd.read_csv(labels_path)
    artifact = joblib.load(args.model)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    model_labels = artifact.get("labels") if isinstance(artifact, dict) else list(model.classes_)

    report_id = "VolumeName"
    label_id = "VolumeName"
    report_texts = _report_text_frame(reports, report_id)

    inverse_map = {v: k for k, v in LABEL_NAME_MAP.items()}
    y_columns = [inverse_map.get(label, label.replace("_", " ").title()) for label in model_labels]
    missing = [column for column in y_columns if column not in labels_df.columns]
    if missing:
        raise ValueError(f"Validation labels missing columns: {missing}")

    merged = report_texts.merge(
        labels_df[[label_id] + y_columns],
        left_on=report_id,
        right_on=label_id,
        how="inner",
    )
    y_true = merged[y_columns].to_numpy(dtype=int)
    y_score = np.asarray(model.predict_proba(merged["report_text"].fillna("")))
    y_pred = (y_score >= args.threshold).astype(int)

    per_label = {}
    for idx, label in enumerate(model_labels):
        per_label[label] = {
            "positive_count": int(y_true[:, idx].sum()),
            "f1": safe_metric(f1_score, y_true[:, idx], y_pred[:, idx]),
            "auroc": safe_metric(roc_auc_score, y_true[:, idx], y_score[:, idx]),
            "auprc": safe_metric(average_precision_score, y_true[:, idx], y_score[:, idx]),
        }

    metrics = {
        "num_cases": int(len(merged)),
        "threshold": args.threshold,
        "labels": model_labels,
        "micro_f1": safe_metric(lambda a, b: f1_score(a, b, average="micro", zero_division=0), y_true, y_pred),
        "macro_f1": safe_metric(lambda a, b: f1_score(a, b, average="macro", zero_division=0), y_true, y_pred),
        "micro_auroc": safe_metric(lambda a, b: roc_auc_score(a, b, average="micro"), y_true, y_score),
        "macro_auroc": safe_metric(lambda a, b: roc_auc_score(a, b, average="macro"), y_true, y_score),
        "micro_auprc": safe_metric(lambda a, b: average_precision_score(a, b, average="micro"), y_true, y_score),
        "macro_auprc": safe_metric(lambda a, b: average_precision_score(a, b, average="macro"), y_true, y_score),
        "per_label": per_label,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
