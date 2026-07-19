import argparse
from pathlib import Path

import pandas as pd

from prepare_dataset import LABEL_NAME_MAP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manifest for locally available CT-RATE volumes.")
    parser.add_argument("--volumes-dir", default="data/dataset/valid_fixed")
    parser.add_argument(
        "--reports",
        default="data/dataset/radiology_text_reports/validation_reports.csv",
    )
    parser.add_argument(
        "--labels",
        default="data/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
    )
    parser.add_argument("--out", default="artifacts/evaluation/multimodal_manifest.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--absolute-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    reports = pd.read_csv(args.reports)
    labels = pd.read_csv(args.labels)
    reports_by_name = reports.set_index("VolumeName")
    labels_by_name = labels.set_index("VolumeName")
    volume_paths = sorted(Path(args.volumes_dir).rglob("*.nii.gz"))
    if args.limit > 0:
        volume_paths = volume_paths[: args.limit]

    rows: list[dict[str, object]] = []
    for volume_path in volume_paths:
        volume_name = volume_path.name
        if volume_name not in reports_by_name.index or volume_name not in labels_by_name.index:
            continue
        report = reports_by_name.loc[volume_name]
        label_row = labels_by_name.loc[volume_name]
        positive_labels = [
            normalized
            for source, normalized in LABEL_NAME_MAP.items()
            if source in label_row.index and float(label_row[source]) > 0
        ]
        resolved = volume_path.resolve()
        stored_path = str(resolved if args.absolute_paths else resolved.relative_to(project_root))
        rows.append(
            {
                "case_id": volume_name.removesuffix(".nii.gz"),
                "volume_name": volume_name,
                "ct_volume_path": stored_path,
                "report_text": (
                    f"Findings: {report.get('Findings_EN', '')}\n"
                    f"Impression: {report.get('Impressions_EN', '')}"
                ),
                "labels": ";".join(positive_labels),
                "split": "validation",
            }
        )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} cases to {output_path}")


if __name__ == "__main__":
    main()
