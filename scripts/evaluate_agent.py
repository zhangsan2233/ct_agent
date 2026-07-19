import argparse
import json
from pathlib import Path
from statistics import mean, median
import sys
import warnings

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.schemas import AnalyzeResponse


LABEL_NAME_MAP = {
    "Medical material": "medical_material",
    "Arterial wall calcification": "arterial_wall_calcification",
    "Cardiomegaly": "cardiomegaly",
    "Pericardial effusion": "pericardial_effusion",
    "Coronary artery wall calcification": "coronary_artery_wall_calcification",
    "Hiatal hernia": "hiatal_hernia",
    "Lymphadenopathy": "lymphadenopathy",
    "Emphysema": "emphysema",
    "Atelectasis": "atelectasis",
    "Lung nodule": "pulmonary_nodule",
    "Lung opacity": "lung_opacity",
    "Pulmonary fibrotic sequela": "pulmonary_fibrotic_sequela",
    "Pleural effusion": "pleural_effusion",
    "Mosaic attenuation pattern": "mosaic_attenuation_pattern",
    "Peribronchial thickening": "peribronchial_thickening",
    "Consolidation": "consolidation",
    "Bronchiectasis": "bronchiectasis",
    "Interlobular septal thickening": "interlobular_septal_thickening",
}
ALLOWED_LABELS = set(LABEL_NAME_MAP.values()) | {"ground_glass_opacity", "pneumothorax"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ChestCT-Agent JSONL predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth")
    parser.add_argument("--out")
    parser.add_argument("--uncertain-as-positive", action="store_true")
    return parser.parse_args()


def _normalize_case_id(value: object) -> str:
    case_id = str(value).strip()
    return case_id.removesuffix(".nii.gz").removesuffix(".nii")


def _safe_metric(metric, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = float(metric(y_true, y_score))
        return result if np.isfinite(result) else None
    except ValueError:
        return None


def _load_ground_truth(path: Path) -> tuple[dict[str, set[str]], list[str]]:
    frame = pd.read_csv(path)
    id_column = next(
        (column for column in ("case_id", "VolumeName", "volume_name") if column in frame.columns),
        frame.columns[0],
    )
    cases: dict[str, set[str]] = {}
    if "labels" in frame.columns:
        labels = sorted(
            {
                label
                for value in frame["labels"].fillna("")
                for label in str(value).split(";")
                if label
            }
        )
        for _, row in frame.iterrows():
            cases[_normalize_case_id(row[id_column])] = {
                label for label in str(row["labels"]).split(";") if label and label != "nan"
            }
        return cases, labels

    label_columns = [column for column in LABEL_NAME_MAP if column in frame.columns]
    for _, row in frame.iterrows():
        cases[_normalize_case_id(row[id_column])] = {
            LABEL_NAME_MAP[column] for column in label_columns if float(row[column]) > 0
        }
    return cases, [LABEL_NAME_MAP[column] for column in label_columns]


def _classification_metrics(
    responses: dict[str, AnalyzeResponse],
    truth: dict[str, set[str]],
    labels: list[str],
    uncertain_as_positive: bool,
) -> dict[str, object]:
    case_ids = sorted(set(responses) & set(truth))
    if not case_ids or not labels:
        return {"matched_cases": 0}

    y_true = np.zeros((len(case_ids), len(labels)), dtype=int)
    y_pred = np.zeros_like(y_true)
    y_score = np.zeros_like(y_true, dtype=float)
    positive_statuses = {"positive", "uncertain"} if uncertain_as_positive else {"positive"}

    for row_index, case_id in enumerate(case_ids):
        output_by_label = {item.name: item for item in responses[case_id].labels}
        for column_index, label in enumerate(labels):
            y_true[row_index, column_index] = int(label in truth[case_id])
            output = output_by_label.get(label)
            if output is not None:
                y_pred[row_index, column_index] = int(output.status in positive_statuses)
                y_score[row_index, column_index] = output.confidence

    per_label: dict[str, object] = {}
    for index, label in enumerate(labels):
        positive_count = int(y_true[:, index].sum())
        per_label[label] = {
            "positive_count": positive_count,
            "f1": _safe_metric(
                lambda actual, predicted: f1_score(
                    actual, predicted, zero_division=0
                ),
                y_true[:, index],
                y_pred[:, index],
            ),
            "auroc": (
                _safe_metric(roc_auc_score, y_true[:, index], y_score[:, index])
                if 0 < positive_count < len(case_ids)
                else None
            ),
            "auprc": (
                _safe_metric(average_precision_score, y_true[:, index], y_score[:, index])
                if positive_count > 0
                else None
            ),
        }

    auroc_columns = [
        index for index in range(len(labels)) if 0 < int(y_true[:, index].sum()) < len(case_ids)
    ]
    auprc_columns = [index for index in range(len(labels)) if int(y_true[:, index].sum()) > 0]

    return {
        "matched_cases": len(case_ids),
        "uncertain_as_positive": uncertain_as_positive,
        "micro_f1": _safe_metric(
            lambda actual, predicted: f1_score(actual, predicted, average="micro", zero_division=0),
            y_true,
            y_pred,
        ),
        "macro_f1": _safe_metric(
            lambda actual, predicted: f1_score(actual, predicted, average="macro", zero_division=0),
            y_true,
            y_pred,
        ),
        "micro_auroc": _safe_metric(
            lambda actual, score: roc_auc_score(actual, score, average="micro"),
            y_true,
            y_score,
        ),
        "macro_auroc": _safe_metric(
            lambda actual, score: roc_auc_score(actual, score, average="macro"),
            y_true[:, auroc_columns],
            y_score[:, auroc_columns],
        ) if auroc_columns else None,
        "micro_auprc": _safe_metric(
            lambda actual, score: average_precision_score(actual, score, average="micro"),
            y_true,
            y_score,
        ),
        "macro_auprc": _safe_metric(
            lambda actual, score: average_precision_score(actual, score, average="macro"),
            y_true[:, auprc_columns],
            y_score[:, auprc_columns],
        ) if auprc_columns else None,
        "per_label": per_label,
    }


def main() -> None:
    args = parse_args()
    path = Path(args.predictions)
    raw_items = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not raw_items:
        raise SystemExit("No predictions found.")

    responses: dict[str, AnalyzeResponse] = {}
    schema_valid = 0
    disclaimer_count = 0
    route_complete = 0
    hallucinated_labels = 0
    conclusion_count = 0
    direct_evidence_count = 0
    localized_evidence_count = 0
    rag_source_count = 0
    rag_source_aligned = 0
    self_retrieval_count = 0
    llm_calls = 0
    llm_fallbacks = 0
    retrieval_attempts: list[int] = []
    latencies: list[float] = []

    for raw in raw_items:
        try:
            response = AnalyzeResponse.model_validate(raw)
        except ValidationError:
            continue
        schema_valid += 1
        case_id = _normalize_case_id(response.case_id)
        responses[case_id] = response
        disclaimer_count += int(bool(response.disclaimer))
        trace = set(response.tool_trace)
        expected_trace = {
            "medical_rag_tool",
            "retrieval_grader",
            "consistency_checker_tool",
            "json_validator_tool",
        }
        if response.execution.input_mode in {"report_only", "report_and_ct"}:
            expected_trace |= {"report_parser_tool", "text_classifier_tool"}
        if response.execution.input_mode in {"ct_only", "report_and_ct"}:
            expected_trace |= {"ct_preprocess_tool", "ct_classifier_tool"}
        route_complete += int(expected_trace <= trace)

        llm_calls += response.execution.llm_calls
        llm_fallbacks += response.execution.llm_fallbacks
        retrieval_attempts.append(response.execution.retrieval_attempts)
        if response.execution.total_latency_ms > 0:
            latencies.append(response.execution.total_latency_ms)
        self_retrieval_count += sum(
            _normalize_case_id(item.case_id) == case_id for item in response.similar_cases
        )

        for label in response.labels:
            hallucinated_labels += int(label.name not in ALLOWED_LABELS)
            if label.status == "negative":
                continue
            conclusion_count += 1
            has_model_or_report_evidence = (
                any(score > 0 for score in label.source_scores.values())
                or bool(label.evidence_from_report)
            )
            direct_evidence_count += int(has_model_or_report_evidence)
            localized_evidence_count += int(label.evidence_from_image.localized)
            for source in label.rag_sources:
                rag_source_count += 1
                rag_source_aligned += int(
                    source.endswith(f":{label.name}")
                    or f":{label.name}:" in source
                )

    total = len(raw_items)
    metrics: dict[str, object] = {
        "total_records": total,
        "json_schema_valid_rate": schema_valid / total,
        "disclaimer_rate": disclaimer_count / total,
        "tool_route_complete_rate": route_complete / total,
        "hallucinated_label_count": hallucinated_labels,
        "conclusion_count": conclusion_count,
        "model_or_report_evidence_rate": (
            direct_evidence_count / conclusion_count if conclusion_count else 1.0
        ),
        "localized_image_evidence_rate": (
            localized_evidence_count / conclusion_count if conclusion_count else 0.0
        ),
        "rag_source_alignment_rate": (
            rag_source_aligned / rag_source_count if rag_source_count else None
        ),
        "similar_case_self_retrieval_count": self_retrieval_count,
        "llm_fallback_rate": llm_fallbacks / llm_calls if llm_calls else 0.0,
        "mean_retrieval_attempts": mean(retrieval_attempts) if retrieval_attempts else 0.0,
    }
    if latencies:
        metrics["latency_ms"] = {
            "mean": mean(latencies),
            "median": median(latencies),
            "p95": float(np.percentile(latencies, 95)),
        }

    if args.ground_truth:
        truth, labels = _load_ground_truth(Path(args.ground_truth))
        metrics["classification"] = _classification_metrics(
            responses,
            truth,
            labels,
            args.uncertain_as_positive,
        )

    output = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
