import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.evaluation import patient_id_from_case_id
from chestct_agent.labels import LABEL_IDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-free patient calibration/test splits.")
    parser.add_argument("--manifest", default="artifacts/evaluation/multimodal_manifest.csv")
    parser.add_argument("--out", default="artifacts/evaluation/patient_splits.csv")
    parser.add_argument("--test-fraction", type=float, default=0.4)
    parser.add_argument("--trials", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest).fillna("")
    frame["patient_id"] = frame["case_id"].map(patient_id_from_case_id)
    patient_labels: dict[str, set[str]] = {}
    for patient_id, group in frame.groupby("patient_id"):
        patient_labels[str(patient_id)] = {
            label
            for value in group["labels"]
            for label in str(value).split(";")
            if label
        }
    patients = np.asarray(sorted(patient_labels))
    matrix = np.asarray(
        [[int(label in patient_labels[patient]) for label in LABEL_IDS] for patient in patients],
        dtype=float,
    )
    target_test = max(1, int(round(len(patients) * args.test_fraction)))
    overall = matrix.mean(axis=0)
    best_score = float("inf")
    best_test: set[str] = set()
    rng = np.random.default_rng(20260717)
    for _ in range(args.trials):
        indices = rng.choice(len(patients), size=target_test, replace=False)
        test_mask = np.zeros(len(patients), dtype=bool)
        test_mask[indices] = True
        calibration = matrix[~test_mask]
        test = matrix[test_mask]
        missing = int((calibration.sum(axis=0) == 0).sum() + (test.sum(axis=0) == 0).sum())
        prevalence_error = float(np.abs(calibration.mean(axis=0) - overall).sum())
        prevalence_error += float(np.abs(test.mean(axis=0) - overall).sum())
        score = missing * 10.0 + prevalence_error
        if score < best_score:
            best_score = score
            best_test = set(patients[test_mask].tolist())

    frame["evaluation_split"] = frame["patient_id"].map(
        lambda patient: "test" if patient in best_test else "calibration"
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[["case_id", "patient_id", "evaluation_split"]].to_csv(output, index=False)
    summary = frame.groupby("evaluation_split")["patient_id"].nunique().to_dict()
    print(f"Wrote {output}: {summary}; objective={best_score:.4f}")


if __name__ == "__main__":
    main()
