"""Calculate binary CT-CLIP metrics and inspect report-derived label mismatches."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


LABELS = [
    "arterial_wall_calcification", "atelectasis",
    "coronary_artery_wall_calcification", "emphysema", "lung_opacity",
    "lymphadenopathy", "pulmonary_fibrotic_sequela", "pulmonary_nodule",
]


def auc_roc(y: list[int], scores: list[float]) -> float | None:
    positives, negatives = sum(y), len(y) - sum(y)
    if not positives or not negatives:
        return None
    ranked = sorted(zip(scores, y), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(y: list[int], scores: list[float]) -> float | None:
    positives = sum(y)
    if not positives:
        return None
    ordered = sorted(zip(scores, y), reverse=True)
    hits = 0
    score = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label:
            hits += 1
            score += hits / rank
    return score / positives


def binary_metrics(y: list[int], scores: list[float], threshold: float) -> dict[str, float | int | None]:
    predicted = [int(value >= threshold) for value in scores]
    tp = sum(a == 1 and b == 1 for a, b in zip(y, predicted))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, predicted))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, predicted))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(y), "positives": sum(y), "negatives": len(y) - sum(y),
        "auroc": auc_roc(y, scores), "auprc": average_precision(y, scores),
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "sensitivity": recall if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": precision, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def mean(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None and not math.isnan(value)]
    return sum(usable) / len(usable) if usable else None


def write_table(path: Path, rows: list[dict]) -> None:
    fields = ["kind", "label", "case_id", "study_id", "ground_truth", "probability", "report_impression"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--high-confidence", type=float, default=0.8)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Batch inference is resumable and may append records already written by a
    # previous interrupted run.  Keep the last result per case, so a resumed
    # job is evaluated as the manifest's cohort rather than as duplicated rows.
    records_by_case = {}
    failures_by_case = {}
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        case_id = row["case_id"]
        if row.get("error"):
            failures_by_case[case_id] = row
        else:
            records_by_case[case_id] = row
            failures_by_case.pop(case_id, None)
    records = list(records_by_case.values())
    failures = list(failures_by_case.values())
    if not records:
        raise SystemExit("No successful predictions available for evaluation.")
    metrics = {label: binary_metrics([int(row["ground_truth"][label]) for row in records], [float(row["probabilities"][label]) for row in records], args.threshold) for label in LABELS}
    flat_y = [int(row["ground_truth"][label]) for row in records for label in LABELS]
    flat_scores = [float(row["probabilities"][label]) for row in records for label in LABELS]
    output = {
        "successful_cases": len(records), "failed_cases": len(failures), "threshold": args.threshold,
        "per_label": metrics,
        "macro": {key: mean([item[key] for item in metrics.values()]) for key in ("auroc", "auprc", "f1", "sensitivity", "specificity")},
        "micro": binary_metrics(flat_y, flat_scores, args.threshold),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = []
    for row in records:
        for label in LABELS:
            truth, probability = int(row["ground_truth"][label]), float(row["probabilities"][label])
            kind = None
            if not truth and probability >= args.high_confidence:
                kind = "high_confidence_false_positive"
            elif truth and probability <= 1 - args.high_confidence:
                kind = "high_confidence_false_negative"
            elif int(probability >= args.threshold) != truth:
                kind = "report_ct_label_disagreement"
            if kind:
                errors.append({"kind": kind, "label": label, "case_id": row["case_id"], "study_id": row.get("study_id", ""), "ground_truth": truth, "probability": probability, "report_impression": row.get("report_impression", "")})
    errors.sort(key=lambda item: (item["kind"], -abs(item["probability"] - args.threshold)))
    write_table(args.out_dir / "error_cases.csv", errors)
    (args.out_dir / "failed_cases.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(args.out_dir / "metrics.json"), "error_cases": len(errors)}, indent=2))


if __name__ == "__main__":
    main()
