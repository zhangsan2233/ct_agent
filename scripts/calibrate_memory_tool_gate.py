"""Calibrate per-label tool thresholds on feedback cases, never on the frozen test set."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[str(item["case_id"])] = item
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum", type=float, default=0.2)
    parser.add_argument("--maximum", type=float, default=0.9)
    parser.add_argument("--step", type=float, default=0.005)
    args = parser.parse_args()

    predictions = {}
    for path in args.predictions:
        predictions.update(load_jsonl(path))
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))

    thresholds = {}
    details = {}
    candidates = np.arange(args.minimum, args.maximum + args.step / 2.0, args.step)
    for label in args.labels:
        rows = [row for row in manifest if row["case_id"] in predictions]
        truth = np.asarray([int(row[label]) for row in rows], dtype=int)
        scores = np.asarray(
            [float(predictions[row["case_id"]]["probabilities"][label]) for row in rows]
        )
        ranked = []
        for threshold in candidates:
            predicted = scores >= threshold
            ranked.append(
                (
                    float(f1_score(truth, predicted, zero_division=0)),
                    -abs(float(threshold) - 0.5),
                    float(threshold),
                    float(precision_score(truth, predicted, zero_division=0)),
                    float(recall_score(truth, predicted, zero_division=0)),
                )
            )
        f1, _, threshold, precision, recall = max(ranked)
        thresholds[label] = threshold
        details[label] = {
            "cases": len(rows),
            "positives": int(truth.sum()),
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": "maximize per-label F1 on feedback cases only",
        "thresholds": thresholds,
        "details": details,
        "notice": "Candidate gate only; lock it before evaluating a disjoint frozen set.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
