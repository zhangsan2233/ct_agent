from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_ct_attribution import load_attribution, mask_metrics


LABEL = "pulmonary_nodule"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CT-CLIP pulmonary-nodule generalization on LIDC-IDRI."
    )
    parser.add_argument("--manifest", default="artifacts/evaluation/lidc_external_manifest.csv")
    parser.add_argument("--predictions", default="artifacts/evaluation/lidc_predictions.jsonl")
    parser.add_argument("--calibration", default="artifacts/calibration/calibrators.joblib")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--out", default="artifacts/evaluation/lidc_external_metrics.json")
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict[str, object]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("case_id") and not item.get("error"):
            result[str(item["case_id"])] = item
    return result


def calibration(path: Path, scores: np.ndarray, threshold: float | None):
    if not path.exists():
        return scores, 0.5 if threshold is None else threshold, "none"
    artifact = joblib.load(path)
    item = artifact.get("sources", {}).get("ct", {}).get(LABEL, {})
    model = item.get("model")
    calibrated = scores
    if model is not None:
        calibrated = model.predict_proba(scores.reshape(-1, 1))[:, 1]
    deployed_threshold = float(item.get("positive_threshold", 0.5))
    return (
        np.asarray(calibrated, dtype=float),
        deployed_threshold if threshold is None else threshold,
        str(artifact.get("version", "unknown")),
    )


def point_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_intervals(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float, repeats: int, seed: int
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("auroc", "auprc", "precision", "sensitivity", "specificity", "f1")}
    for _ in range(repeats):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        actual, score = y_true[indices], y_score[indices]
        if len(np.unique(actual)) < 2:
            continue
        metrics = point_metrics(actual, score, threshold)
        for name in values:
            values[name].append(float(metrics[name]))
    return {
        name: {
            "lower": float(np.percentile(items, 2.5)),
            "upper": float(np.percentile(items, 97.5)),
        }
        for name, items in values.items()
        if items
    }


def localization_metrics(rows: pd.DataFrame, predictions: dict[str, dict[str, object]]) -> dict:
    results = []
    for _, row in rows[rows["ground_truth"].eq("positive")].iterrows():
        prediction = predictions[str(row["case_id"])]
        artifact_value = prediction.get("attribution_artifact")
        if not artifact_value:
            continue
        artifact = Path(str(artifact_value))
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        mask = PROJECT_ROOT / str(row["nodule_consensus2_mask_path"])
        if not artifact.exists() or not mask.exists():
            continue
        attribution, _, preprocess = load_attribution(artifact, LABEL)
        results.append(mask_metrics(attribution, preprocess, mask))
    return {
        "eligible_positive_cases": int(rows["ground_truth"].eq("positive").sum()),
        "evaluated_cases": len(results),
        "pointing_game_accuracy": (
            mean(float(item["pointing_game_hit"]) for item in results) if results else None
        ),
        "mean_mask_energy_ratio": (
            mean(float(item["mask_energy_ratio"]) for item in results) if results else None
        ),
        "note": (
            "Mask requires overlap from at least two expert readers; the heatmap explains "
            "the CT-CLIP score and is not segmentation."
        ),
    }


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest).fillna("")
    predictions = load_predictions(Path(args.predictions))
    matched = manifest[manifest["case_id"].astype(str).isin(predictions)].copy()
    if matched.empty:
        raise SystemExit("No completed LIDC-IDRI predictions match the manifest.")
    y_true = matched["ground_truth"].eq("positive").astype(int).to_numpy()
    raw_scores = np.asarray(
        [float(predictions[str(case)]["probabilities"][LABEL]) for case in matched["case_id"]]
    )
    scores, threshold, calibration_version = calibration(
        Path(args.calibration), raw_scores, args.threshold
    )
    metrics = point_metrics(y_true, scores, threshold)
    metrics.update(
        {
            "dataset": "LIDC-IDRI",
            "task": "scan-level pulmonary nodule presence",
            "cases_planned": int(len(manifest)),
            "cases_completed": int(len(matched)),
            "coverage": float(len(matched) / len(manifest)),
            "positive_cases": int(y_true.sum()),
            "negative_cases": int(len(y_true) - y_true.sum()),
            "calibration_version": calibration_version,
            "threshold_policy": "fixed before external evaluation; no LIDC tuning",
            "raw_score_fixed_0_5": point_metrics(y_true, raw_scores, 0.5),
            "ground_truth_policy": (
                "positive if >=3 readers marked a >=3mm nodule; negative if no reader did; "
                "1-2-reader cases excluded"
            ),
            "confidence_intervals_95": bootstrap_intervals(
                y_true, scores, threshold, args.bootstrap_repeats, args.seed
            ),
            "localization": localization_metrics(matched, predictions),
            "scope_warning": (
                "This benchmark validates pulmonary_nodule only, not the other 17 labels, "
                "report generation, RAG factuality, or clinical safety."
            ),
        }
    )
    output = json.dumps(metrics, ensure_ascii=False, indent=2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
