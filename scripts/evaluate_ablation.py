import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.config import Settings
from chestct_agent.evaluation import (
    aggregate_by_patient,
    bootstrap_macro_interval,
    classification_metrics,
)
from chestct_agent.labels import LABEL_IDS
from chestct_agent.schemas import ParsedReport
from chestct_agent.tools.text_classifier import TextClassifierTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patient-level CT/report/fusion ablation and calibration.")
    parser.add_argument("--manifest", default="artifacts/evaluation/multimodal_manifest.csv")
    parser.add_argument("--splits", default="artifacts/evaluation/patient_splits.csv")
    parser.add_argument("--ct-predictions", default="artifacts/evaluation/ctclip_predictions.jsonl")
    parser.add_argument("--calibrators", default="artifacts/calibration/calibrators.joblib")
    parser.add_argument("--out", default="artifacts/evaluation/ablation_patient_metrics.json")
    parser.add_argument(
        "--ct-threshold-method", choices=("f1", "precision"), default="precision"
    )
    parser.add_argument("--ct-target-precision", type=float, default=0.6)
    parser.add_argument("--ct-uncertain-target-precision", type=float, default=0.4)
    return parser.parse_args()


def _ct_scores(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("case_id") and item.get("probabilities"):
            result[str(item["case_id"])] = {
                str(label): float(score)
                for label, score in dict(item["probabilities"]).items()
            }
    return result


def _best_threshold(actual: np.ndarray, score: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 91)
    values = [f1_score(actual, score >= threshold, zero_division=0) for threshold in candidates]
    best = max(values)
    selected = float(candidates[next(index for index, value in enumerate(values) if value == best)])
    minority_count = min(int(actual.sum()), int((1 - actual).sum()))
    reliability = min(1.0, minority_count / 25.0)
    return float(0.5 + reliability * (selected - 0.5))


def _precision_threshold(
    actual: np.ndarray,
    score: np.ndarray,
    target_precision: float,
    minimum_predictions: int = 3,
) -> float:
    selected = 1.0001
    selected_recall = -1.0
    selected_precision = -1.0
    for threshold in np.unique(np.append(score, 1.0001)):
        predicted = score >= threshold
        if int(predicted.sum()) < minimum_predictions:
            continue
        precision = precision_score(actual, predicted, zero_division=0)
        recall = recall_score(actual, predicted, zero_division=0)
        if precision < target_precision:
            continue
        if recall > selected_recall or (
            recall == selected_recall and precision > selected_precision
        ):
            selected = float(threshold)
            selected_recall = float(recall)
            selected_precision = float(precision)
    return selected


def _fit_source(
    y_true: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    threshold_method: str = "f1",
    target_precision: float = 0.6,
    uncertain_target_precision: float = 0.4,
) -> tuple[dict[str, dict], np.ndarray, np.ndarray]:
    artifacts: dict[str, dict] = {}
    calibrated = scores.copy()
    thresholds = np.full(len(labels), 0.5, dtype=float)
    for index, label in enumerate(labels):
        actual = y_true[:, index]
        model = None
        if int(actual.sum()) >= 20 and int((1 - actual).sum()) >= 20:
            features = scores[:, index].reshape(-1, 1)
            minority_count = min(int(actual.sum()), int((1 - actual).sum()))
            folds = min(5, minority_count)
            candidate = LogisticRegression(max_iter=1000)
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260717)
            out_of_fold = cross_val_predict(
                candidate, features, actual, cv=splitter, method="predict_proba"
            )[:, 1]
            raw_brier = brier_score_loss(actual, scores[:, index])
            calibrated_brier = brier_score_loss(actual, out_of_fold)
            if calibrated_brier < raw_brier:
                model = candidate.fit(features, actual)
                calibrated[:, index] = out_of_fold
        f1_threshold = _best_threshold(actual, calibrated[:, index])
        if threshold_method == "precision":
            thresholds[index] = _precision_threshold(
                actual, calibrated[:, index], target_precision
            )
            uncertain_threshold = min(
                thresholds[index],
                _precision_threshold(
                    actual, calibrated[:, index], uncertain_target_precision
                ),
            )
        else:
            thresholds[index] = f1_threshold
            uncertain_threshold = max(0.05, thresholds[index] - 0.15)
        artifacts[label] = {
            "model": model,
            "positive_threshold": float(thresholds[index]),
            "uncertain_threshold": float(uncertain_threshold),
            "positive_count": int(actual.sum()),
        }
    return artifacts, calibrated, thresholds


def _apply_source(artifact: dict[str, dict], scores: np.ndarray, labels: list[str]):
    calibrated = scores.copy()
    thresholds = np.full(len(labels), 0.5, dtype=float)
    for index, label in enumerate(labels):
        item = artifact[label]
        if item["model"] is not None:
            calibrated[:, index] = item["model"].predict_proba(
                scores[:, index].reshape(-1, 1)
            )[:, 1]
        thresholds[index] = float(item["positive_threshold"])
    return calibrated, thresholds


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest).fillna("")
    splits = pd.read_csv(args.splits)
    frame = manifest.merge(splits, on="case_id", how="inner")
    ct_by_case = _ct_scores(Path(args.ct_predictions))
    frame = frame[frame["case_id"].astype(str).isin(ct_by_case)].reset_index(drop=True)
    labels = list(LABEL_IDS)

    no_calibration = Path(args.calibrators).with_suffix(".not-yet-created")
    classifier = TextClassifierTool(
        Settings(artifact_dir=Path("artifacts"), calibration_path=no_calibration)
    )
    y_true = np.zeros((len(frame), len(labels)), dtype=int)
    ct = np.zeros_like(y_true, dtype=float)
    report = np.zeros_like(y_true, dtype=float)
    for row_index, row in frame.iterrows():
        positives = {label for label in str(row["labels"]).split(";") if label}
        report_predictions = classifier.predict(
            ParsedReport(full_report=str(row["report_text"]))
        )
        report_map = {item.name: item.confidence for item in report_predictions}
        ct_map = ct_by_case[str(row["case_id"])]
        for column, label in enumerate(labels):
            y_true[row_index, column] = int(label in positives)
            report[row_index, column] = float(report_map.get(label, 0.0))
            ct[row_index, column] = float(ct_map.get(label, 0.0))
    fusion = report * 0.55 + ct * 0.45
    source_scores = {"report": report, "ct": ct, "fusion": fusion}

    calibration_mask = frame["evaluation_split"].eq("calibration").to_numpy()
    test_mask = frame["evaluation_split"].eq("test").to_numpy()
    artifact: dict[str, object] = {
        "version": "patient-platt-oof-shrunk-v2",
        "labels": labels,
        "label_source": "CT-RATE report-derived weak labels",
        "sources": {},
        "threshold_method": {
            "report": "OOF F1 optimum shrunk toward 0.5 by minority patient count / 25",
            "ct": (
                f"patient calibration precision >= {args.ct_target_precision:.2f}; "
                "uncertain boundary precision >= "
                f"{args.ct_uncertain_target_precision:.2f}"
                if args.ct_threshold_method == "precision"
                else "OOF F1 optimum shrunk toward 0.5 by minority patient count / 25"
            ),
            "fusion": "OOF F1 optimum shrunk toward 0.5 by minority patient count / 25",
        },
        "minimum_class_count_for_platt": 20,
    }
    output_metrics: dict[str, object] = {
        "label_source": "CT-RATE report-derived weak labels",
        "split": {
            "calibration_patients": int(frame.loc[calibration_mask, "patient_id"].nunique()),
            "test_patients": int(frame.loc[test_mask, "patient_id"].nunique()),
            "patient_overlap": int(
                len(
                    set(frame.loc[calibration_mask, "patient_id"])
                    & set(frame.loc[test_mask, "patient_id"])
                )
            ),
        },
        "sources": {},
    }
    test_patients = frame.loc[test_mask, "patient_id"].astype(str).tolist()
    calibration_patients = frame.loc[calibration_mask, "patient_id"].astype(str).tolist()
    for source, scores in source_scores.items():
        _, calibration_truth, calibration_scores = aggregate_by_patient(
            calibration_patients, y_true[calibration_mask], scores[calibration_mask]
        )
        source_artifact, _, _ = _fit_source(
            calibration_truth,
            calibration_scores,
            labels,
            threshold_method=(
                args.ct_threshold_method if source == "ct" else "f1"
            ),
            target_precision=args.ct_target_precision,
            uncertain_target_precision=args.ct_uncertain_target_precision,
        )
        artifact["sources"][source] = source_artifact
        calibrated_test, thresholds = _apply_source(
            source_artifact, scores[test_mask], labels
        )
        patient_ids, patient_truth, patient_scores = aggregate_by_patient(
            test_patients, y_true[test_mask], calibrated_test
        )
        raw_patient = aggregate_by_patient(test_patients, y_true[test_mask], scores[test_mask])[2]
        raw_metrics = classification_metrics(
            patient_truth, raw_patient, np.full(len(labels), 0.5), labels
        )
        calibrated_metrics = classification_metrics(
            patient_truth, patient_scores, thresholds, labels
        )
        calibrated_metrics["bootstrap_95_ci"] = {
            "macro_auroc": bootstrap_macro_interval(
                patient_truth, patient_scores, "auroc"
            ),
            "macro_auprc": bootstrap_macro_interval(
                patient_truth, patient_scores, "auprc"
            ),
        }
        output_metrics["sources"][source] = {
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
            "evaluated_patient_ids": patient_ids,
        }

    calibrator_path = Path(args.calibrators)
    calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, calibrator_path)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {calibrator_path}")
    print(f"Wrote {output_path}")
    for source, metrics in output_metrics["sources"].items():
        raw = metrics["raw"]
        calibrated = metrics["calibrated"]
        print(
            source,
            f"macro_f1 {raw['macro_f1']:.4f}->{calibrated['macro_f1']:.4f}",
            f"ECE {raw['macro_ece']:.4f}->{calibrated['macro_ece']:.4f}",
        )


if __name__ == "__main__":
    main()
