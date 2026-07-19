import argparse
import csv
from pathlib import Path


REPORT_FIELDS = (
    ("Clinical Information", "ClinicalInformation_EN"),
    ("Technique", "Technique_EN"),
    ("Findings", "Findings_EN"),
    ("Impression", "Impressions_EN"),
)


def load_reports(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["VolumeName"].strip(): row
            for row in csv.DictReader(handle)
            if row.get("VolumeName", "").strip()
        }


def render_report(row: dict[str, str]) -> str:
    sections = []
    for title, field in REPORT_FIELDS:
        value = row.get(field, "").strip() or "Not given."
        sections.append(f"{title}: {value}")
    return "\n\n".join(sections) + "\n"


def export_reports(ct_root: Path, reports_csv: Path, overwrite: bool) -> tuple[int, int]:
    reports = load_reports(reports_csv)
    written = 0
    missing = 0
    for ct_path in sorted(ct_root.rglob("*.nii.gz")):
        row = reports.get(ct_path.name)
        if row is None:
            missing += 1
            continue
        output_path = ct_path.with_name(
            ct_path.name.removesuffix(".nii.gz") + "_report.txt"
        )
        if output_path.exists() and not overwrite:
            continue
        output_path.write_text(render_report(row), encoding="utf-8")
        written += 1
    return written, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CT-RATE validation CSV reports beside downloaded CT volumes."
    )
    parser.add_argument(
        "--ct-root",
        type=Path,
        default=Path("data/dataset/valid_fixed"),
    )
    parser.add_argument(
        "--reports-csv",
        type=Path,
        default=Path("data/dataset/radiology_text_reports/validation_reports.csv"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    written, missing = export_reports(args.ct_root, args.reports_csv, args.overwrite)
    print(f"written={written} missing={missing}")


if __name__ == "__main__":
    main()
