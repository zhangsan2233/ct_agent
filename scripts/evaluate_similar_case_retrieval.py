import argparse
import json
from pathlib import Path
import random
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.config import get_settings
from chestct_agent.labels import LABEL_SPECS, SOURCE_COLUMN_TO_ID
from chestct_agent.tools.similar_cases import SimilarCaseRetrieverTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CT-RATE report-to-case retrieval without validation-label leakage."
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--reports",
        default="data/dataset/radiology_text_reports/validation_reports.csv",
    )
    parser.add_argument(
        "--labels",
        default="data/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
    )
    parser.add_argument(
        "--out",
        default="artifacts/evaluation/similar_case_metrics.json",
    )
    return parser.parse_args()


def _validation_rows(reports_path: Path, labels_path: Path) -> pd.DataFrame:
    reports = pd.read_csv(reports_path)
    labels = pd.read_csv(labels_path)
    report_text = (
        "Findings: "
        + reports["Findings_EN"].fillna("").astype(str)
        + " Impression: "
        + reports["Impressions_EN"].fillna("").astype(str)
    )
    report_rows = pd.DataFrame(
        {"case_id": reports["VolumeName"].astype(str), "report_text": report_text}
    )
    canonical_columns = [
        spec.source_column for spec in LABEL_SPECS if spec.source_column in labels.columns
    ]
    label_rows = labels[["VolumeName", *canonical_columns]].copy()
    label_rows["labels"] = label_rows[canonical_columns].apply(
        lambda row: {
            SOURCE_COLUMN_TO_ID[column]
            for column in canonical_columns
            if float(row[column]) > 0
        },
        axis=1,
    )
    merged = report_rows.merge(label_rows[["VolumeName", "labels"]], left_on="case_id", right_on="VolumeName")
    return merged[
        merged["report_text"].str.strip().ne("") & merged["labels"].map(bool)
    ].reset_index(drop=True)


def _case_metrics(query_labels: set[str], candidates: list[set[str]]) -> dict[str, float]:
    union = set().union(*candidates) if candidates else set()
    top_one = candidates[0] if candidates else set()
    return {
        "label_recall": len(query_labels & union) / len(query_labels),
        "any_label_match": float(bool(query_labels & union)),
        "top1_jaccard": len(query_labels & top_one) / max(1, len(query_labels | top_one)),
    }


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.top_k < 1:
        raise ValueError("--limit and --top-k must be positive")
    rng = random.Random(args.seed)
    validation = _validation_rows(Path(args.reports), Path(args.labels))
    sample_size = min(args.limit, len(validation))
    validation = validation.sample(n=sample_size, random_state=args.seed)

    retriever = SimilarCaseRetrieverTool(get_settings())
    labels_by_case = {
        str(row.case_id): {label for label in str(row.labels).split(";") if label}
        for row in retriever.index.itertuples(index=False)
    }
    patient_to_cases: dict[str, list[str]] = {}
    for case_id, patient_id in zip(
        retriever.index["case_id"].astype(str), retriever.patient_ids, strict=True
    ):
        patient_to_cases.setdefault(patient_id, []).append(case_id)
    training_patients = sorted(patient_to_cases)

    retrieval_rows: list[dict[str, float]] = []
    random_rows: list[dict[str, float]] = []
    unique_patient_rates: list[float] = []
    failures: list[dict[str, object]] = []
    for row in validation.itertuples(index=False):
        query_labels = set(row.labels)
        results = retriever.retrieve(
            str(row.report_text),
            labels=[],
            top_k=args.top_k,
            query_case_id=str(row.case_id),
        )
        candidate_labels = [labels_by_case.get(item.case_id, set()) for item in results]
        metrics = _case_metrics(query_labels, candidate_labels)
        retrieval_rows.append(metrics)
        unique_patient_rates.append(
            len({item.patient_id for item in results}) / max(1, len(results))
        )
        if not metrics["any_label_match"]:
            failures.append(
                {
                    "case_id": str(row.case_id),
                    "query_labels": sorted(query_labels),
                    "retrieved_cases": [item.case_id for item in results],
                }
            )

        random_patients = rng.sample(training_patients, k=min(args.top_k, len(training_patients)))
        random_labels = [
            labels_by_case.get(rng.choice(patient_to_cases[patient_id]), set())
            for patient_id in random_patients
        ]
        random_rows.append(_case_metrics(query_labels, random_labels))

    retrieval = _average(retrieval_rows)
    baseline = _average(random_rows)
    result = {
        "dataset": "CT-RATE validation reports queried against training reports",
        "queries": sample_size,
        "top_k": args.top_k,
        "query_inputs": "report text only; validation labels withheld from retriever",
        "reference": "CT-RATE report-derived weak labels",
        "retrieval": retrieval,
        "random_patient_baseline": baseline,
        "absolute_improvement": {
            key: retrieval[key] - baseline[key] for key in retrieval
        },
        "unique_patient_rate": sum(unique_patient_rates) / len(unique_patient_rates),
        "failure_count": len(failures),
        "failure_examples": failures[:20],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "failure_examples"}, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
