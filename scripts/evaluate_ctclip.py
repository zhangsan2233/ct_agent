import argparse
import json
from pathlib import Path
from statistics import mean, median
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CT-CLIP probabilities on a manifest.")
    parser.add_argument(
        "--manifest",
        default="artifacts/evaluation/multimodal_manifest.csv",
    )
    parser.add_argument(
        "--predictions",
        default="artifacts/evaluation/ctclip_predictions.jsonl",
    )
    parser.add_argument(
        "--out",
        default="artifacts/evaluation/ctclip_metrics.json",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model-name", default="CT-CLIP v2 zero-shot")
    return parser.parse_args()


def safe_metric(metric, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = float(metric(y_true, y_score))
        return result if np.isfinite(result) else None
    except ValueError:
        return None


def load_predictions(path: Path) -> dict[str, dict[str, object]]:
    predictions: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("case_id") and not item.get("error"):
            predictions[str(item["case_id"])] = item
    return predictions


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest).fillna("")
    predictions = load_predictions(Path(args.predictions))
    labels = sorted(
        {
            label
            for item in predictions.values()
            for label in dict(item["probabilities"])
        }
    )
    matched = manifest[manifest["case_id"].astype(str).isin(predictions)].copy()
    if matched.empty or not labels:
        raise SystemExit("No completed CT-CLIP predictions match the manifest.")

    y_true = np.zeros((len(matched), len(labels)), dtype=int)
    y_score = np.zeros_like(y_true, dtype=float)
    latencies: list[float] = []
    cache_hits = 0
    for row_index, (_, row) in enumerate(matched.iterrows()):
        case_id = str(row["case_id"])
        truth = {label for label in str(row["labels"]).split(";") if label}
        prediction = predictions[case_id]
        probabilities = dict(prediction["probabilities"])
        for column_index, label in enumerate(labels):
            y_true[row_index, column_index] = int(label in truth)
            y_score[row_index, column_index] = float(probabilities.get(label, 0.0))
        latencies.append(float(prediction.get("latency_ms", 0.0)))
        cache_hits += int(bool(prediction.get("cache_hit")))

    y_pred = (y_score >= args.threshold).astype(int)
    per_label: dict[str, dict[str, float | int | None]] = {}
    auroc_values: list[float] = []
    auprc_values: list[float] = []
    for index, label in enumerate(labels):
        positive_count = int(y_true[:, index].sum())
        auroc = (
            safe_metric(roc_auc_score, y_true[:, index], y_score[:, index])
            if 0 < positive_count < len(matched)
            else None
        )
        auprc = (
            safe_metric(average_precision_score, y_true[:, index], y_score[:, index])
            if positive_count > 0
            else None
        )
        if auroc is not None:
            auroc_values.append(auroc)
        if auprc is not None:
            auprc_values.append(auprc)
        per_label[label] = {
            "positive_count": positive_count,
            "precision": safe_metric(
                lambda actual, predicted: precision_score(
                    actual, predicted, zero_division=0
                ),
                y_true[:, index],
                y_pred[:, index],
            ),
            "recall": safe_metric(
                lambda actual, predicted: recall_score(
                    actual, predicted, zero_division=0
                ),
                y_true[:, index],
                y_pred[:, index],
            ),
            "f1": safe_metric(
                lambda actual, predicted: f1_score(
                    actual,
                    predicted,
                    zero_division=0,
                ),
                y_true[:, index],
                y_pred[:, index],
            ),
            "auroc": auroc,
            "auprc": auprc,
        }

    metrics = {
        "model": args.model_name,
        "manifest_cases": int(len(manifest)),
        "completed_cases": int(len(matched)),
        "coverage": float(len(matched) / len(manifest)),
        "threshold": args.threshold,
        "labels": labels,
        "micro_f1": safe_metric(
            lambda actual, predicted: f1_score(
                actual,
                predicted,
                average="micro",
                zero_division=0,
            ),
            y_true,
            y_pred,
        ),
        "micro_precision": safe_metric(
            lambda actual, predicted: precision_score(
                actual, predicted, average="micro", zero_division=0
            ),
            y_true,
            y_pred,
        ),
        "micro_recall": safe_metric(
            lambda actual, predicted: recall_score(
                actual, predicted, average="micro", zero_division=0
            ),
            y_true,
            y_pred,
        ),
        "macro_f1": safe_metric(
            lambda actual, predicted: f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0,
            ),
            y_true,
            y_pred,
        ),
        "micro_auroc": safe_metric(
            lambda actual, score: roc_auc_score(actual, score, average="micro"),
            y_true,
            y_score,
        ),
        "macro_auroc": mean(auroc_values) if auroc_values else None,
        "micro_auprc": safe_metric(
            lambda actual, score: average_precision_score(actual, score, average="micro"),
            y_true,
            y_score,
        ),
        "macro_auprc": mean(auprc_values) if auprc_values else None,
        "cache_hit_rate": cache_hits / len(matched),
        "latency_ms": {
            "mean": mean(latencies),
            "median": median(latencies),
            "p95": float(np.percentile(latencies, 95)),
        },
        "per_label": per_label,
        "label_source": "CT-RATE dataset-provided predicted labels (weak labels)",
    }
    output = json.dumps(metrics, ensure_ascii=False, indent=2)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
