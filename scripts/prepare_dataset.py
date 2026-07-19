import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.labels import LABEL_SPECS, SOURCE_COLUMN_TO_ID


ID_CANDIDATES = ["case_id", "volume_name", "VolumeName", "study_id", "StudyInstanceUID", "id"]
REPORT_CANDIDATES = [
    "report_text",
    "Report",
    "report",
    "findings",
    "Findings",
    "Findings_EN",
    "impression",
    "Impression",
    "Impressions_EN",
]

LABEL_NAME_MAP = SOURCE_COLUMN_TO_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CT-RATE reports/labels into a compact case index.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="artifacts/prepared")
    parser.add_argument(
        "--top-labels",
        type=int,
        default=0,
        help="Keep the N most frequent labels. Zero keeps the canonical 18 labels.",
    )
    return parser.parse_args()


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _read_first_existing(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("None of the expected files exist: " + ", ".join(map(str, paths)))


def _report_text_frame(reports: pd.DataFrame, report_id: str) -> pd.DataFrame:
    if {"Findings_EN", "Impressions_EN"} <= set(reports.columns):
        text = (
            "Findings: "
            + reports["Findings_EN"].fillna("").astype(str)
            + " Impression: "
            + reports["Impressions_EN"].fillna("").astype(str)
        )
        return pd.DataFrame({report_id: reports[report_id].astype(str), "report_text": text})

    report_column = _first_existing_column(reports, REPORT_CANDIDATES)
    if report_column is None:
        text_columns = [column for column in reports.columns if reports[column].dtype == object]
        if not text_columns:
            raise ValueError("No report text column found.")
        report_column = text_columns[-1]
    return pd.DataFrame(
        {
            report_id: reports[report_id].astype(str),
            "report_text": reports[report_column].fillna("").astype(str),
        }
    )


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = _read_first_existing(
        [
            data_dir / "dataset/radiology_text_reports/train_reports.csv",
            data_dir / "radiology_text_reports/train_reports.csv",
        ]
    )
    labels = _read_first_existing(
        [
            data_dir / "dataset/multi_abnormality_labels/train_predicted_labels.csv",
            data_dir / "multi_abnormality_labels/train_predicted_labels.csv",
        ]
    )

    report_id = _first_existing_column(reports, ID_CANDIDATES) or reports.columns[0]
    label_id = _first_existing_column(labels, ID_CANDIDATES) or labels.columns[0]
    report_texts = _report_text_frame(reports, report_id)

    numeric_label_columns = [
        column
        for column in labels.columns
        if column != label_id and pd.api.types.is_numeric_dtype(labels[column])
    ]
    canonical_columns = [
        spec.source_column for spec in LABEL_SPECS if spec.source_column in numeric_label_columns
    ]
    if len(canonical_columns) != len(LABEL_SPECS):
        missing = sorted(set(SOURCE_COLUMN_TO_ID) - set(canonical_columns))
        raise ValueError(f"CT-RATE label file is missing canonical columns: {missing}")
    if args.top_labels > 0:
        top_label_columns = (
            labels[canonical_columns]
            .sum()
            .sort_values(ascending=False)
            .head(args.top_labels)
            .index.tolist()
        )
    else:
        top_label_columns = canonical_columns

    label_rows = labels[[label_id] + top_label_columns].copy()
    label_rows["labels"] = label_rows[top_label_columns].apply(
        lambda row: ";".join(
            [LABEL_NAME_MAP.get(label, label.lower().replace(" ", "_")) for label in top_label_columns if row[label] > 0]
        ),
        axis=1,
    )
    merged = report_texts.merge(
        label_rows[[label_id, "labels"]],
        left_on=report_id,
        right_on=label_id,
        how="inner",
    )
    case_index = pd.DataFrame(
        {
            "case_id": merged[report_id].astype(str),
            "report_text": merged["report_text"].fillna("").astype(str),
            "labels": merged["labels"].fillna("").astype(str),
        }
    )
    case_index.to_csv(out_dir / "case_index.csv", index=False)
    pd.Series(
        [LABEL_NAME_MAP.get(label, label.lower().replace(" ", "_")) for label in top_label_columns],
        name="label",
    ).to_csv(out_dir / "top_labels.csv", index=False)
    print(f"Wrote {out_dir / 'case_index.csv'}")
    print(f"Wrote {out_dir / 'top_labels.csv'}")


if __name__ == "__main__":
    main()
