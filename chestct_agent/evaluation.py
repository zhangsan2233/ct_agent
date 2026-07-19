import re
import warnings

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PATIENT_PATTERN = re.compile(r"^(train|valid)_(\d+)", re.IGNORECASE)


def patient_id_from_case_id(case_id: str) -> str:
    normalized = str(case_id).removesuffix(".nii.gz").removesuffix(".nii")
    match = PATIENT_PATTERN.match(normalized)
    if match:
        return f"{match.group(1).lower()}_{int(match.group(2))}"
    return normalized


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (y_score >= lower) & (
            y_score <= upper if index == bins - 1 else y_score < upper
        )
        if not mask.any():
            continue
        result += float(mask.mean()) * abs(
            float(y_true[mask].mean()) - float(y_score[mask].mean())
        )
    return result


def aggregate_by_patient(
    patient_ids: list[str], y_true: np.ndarray, y_score: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    unique = sorted(set(patient_ids))
    truth = np.zeros((len(unique), y_true.shape[1]), dtype=int)
    scores = np.zeros((len(unique), y_score.shape[1]), dtype=float)
    for row, patient_id in enumerate(unique):
        indices = [index for index, value in enumerate(patient_ids) if value == patient_id]
        truth[row] = y_true[indices].max(axis=0)
        scores[row] = y_score[indices].max(axis=0)
    return unique, truth, scores


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    labels: list[str],
) -> dict[str, object]:
    y_pred = (y_score >= thresholds.reshape(1, -1)).astype(int)
    per_label: dict[str, dict[str, float | int | None]] = {}
    aurocs: list[float] = []
    auprcs: list[float] = []
    briers: list[float] = []
    eces: list[float] = []
    for index, label in enumerate(labels):
        actual = y_true[:, index]
        score = y_score[:, index]
        positive_count = int(actual.sum())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auroc = (
                float(roc_auc_score(actual, score))
                if 0 < positive_count < len(actual)
                else None
            )
            auprc = (
                float(average_precision_score(actual, score))
                if positive_count > 0
                else None
            )
        brier = float(brier_score_loss(actual, score))
        ece = expected_calibration_error(actual, score)
        if auroc is not None:
            aurocs.append(auroc)
        if auprc is not None:
            auprcs.append(auprc)
        briers.append(brier)
        eces.append(ece)
        per_label[label] = {
            "positive_count": positive_count,
            "prevalence": float(actual.mean()),
            "threshold": float(thresholds[index]),
            "precision": float(
                precision_score(actual, y_pred[:, index], zero_division=0)
            ),
            "recall": float(recall_score(actual, y_pred[:, index], zero_division=0)),
            "f1": float(f1_score(actual, y_pred[:, index], zero_division=0)),
            "auroc": auroc,
            "auprc": auprc,
            "brier": brier,
            "ece": ece,
        }
    return {
        "patients": int(len(y_true)),
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
        "macro_brier": float(np.mean(briers)),
        "macro_ece": float(np.mean(eces)),
        "per_label": per_label,
    }


def bootstrap_macro_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str,
    repeats: int = 300,
    seed: int = 20260717,
) -> dict[str, float | None]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repeats):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        actual = y_true[indices]
        score = y_score[indices]
        label_values: list[float] = []
        for column in range(actual.shape[1]):
            positives = int(actual[:, column].sum())
            try:
                if metric == "auroc" and 0 < positives < len(actual):
                    label_values.append(float(roc_auc_score(actual[:, column], score[:, column])))
                elif metric == "auprc" and positives > 0:
                    label_values.append(
                        float(average_precision_score(actual[:, column], score[:, column]))
                    )
            except ValueError:
                continue
        if label_values:
            values.append(float(np.mean(label_values)))
    if not values:
        return {"lower": None, "upper": None}
    return {
        "lower": float(np.percentile(values, 2.5)),
        "upper": float(np.percentile(values, 97.5)),
    }
