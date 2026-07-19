"""Create a deterministic, multi-label-stratified CT-RATE validation manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import random


TARGET_LABELS = [
    "arterial_wall_calcification",
    "atelectasis",
    "coronary_artery_wall_calcification",
    "emphysema",
    "lung_opacity",
    "lymphadenopathy",
    "pulmonary_fibrotic_sequela",
    "pulmonary_nodule",
]

CSV_COLUMNS = {
    "arterial_wall_calcification": "Arterial wall calcification",
    "atelectasis": "Atelectasis",
    "coronary_artery_wall_calcification": "Coronary artery wall calcification",
    "emphysema": "Emphysema",
    "lung_opacity": "Lung opacity",
    "lymphadenopathy": "Lymphadenopathy",
    "pulmonary_fibrotic_sequela": "Pulmonary fibrotic sequela",
    "pulmonary_nodule": "Lung nodule",
}


def remote_volume_path(volume_name: str, remote_data_root: str) -> str:
    stem = volume_name.removesuffix(".nii.gz")
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "valid":
        raise ValueError(f"Unexpected CT-RATE validation volume name: {volume_name}")
    group = "_".join(parts[:2])
    subgroup = "_".join(parts[:3])
    return f"{remote_data_root.rstrip('/')}/{group}/{subgroup}/{volume_name}"


def load_rows(labels_path: Path, reports_path: Path, remote_data_root: str) -> list[dict[str, str | int]]:
    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    with reports_path.open(encoding="utf-8-sig", newline="") as handle:
        reports = {row["VolumeName"]: row for row in csv.DictReader(handle)}

    rows: list[dict[str, str | int]] = []
    for row in labels:
        volume_name = row["VolumeName"]
        report = reports.get(volume_name, {})
        item: dict[str, str | int] = {
            "case_id": volume_name.removesuffix(".nii.gz"),
            "study_id": volume_name.removesuffix(".nii.gz").rsplit("_", 1)[0],
            "volume_name": volume_name,
            "ct_volume_path": remote_volume_path(volume_name, remote_data_root),
            "report_impression": report.get("Impressions_EN", "").strip(),
        }
        for label, source_column in CSV_COLUMNS.items():
            item[label] = int(float(row[source_column]))
        rows.append(item)
    return rows


def select_rows(rows: list[dict[str, str | int]], sample_size: int, seed: int) -> list[dict[str, str | int]]:
    if sample_size > len(rows):
        raise ValueError(f"Requested {sample_size} samples from only {len(rows)} rows.")
    rng = random.Random(seed)
    # CT-RATE may include multiple reconstructions of the same study (for
    # example valid_213_a_1 and valid_213_a_2). Keep one deterministically.
    by_study: dict[str, dict[str, str | int]] = {}
    for row in sorted(rows, key=lambda item: str(item["volume_name"])):
        by_study.setdefault(str(row["study_id"]), row)
    candidates = list(by_study.values())
    if sample_size > len(candidates):
        raise ValueError(f"Requested {sample_size} unique studies from only {len(candidates)} studies.")
    rng.shuffle(candidates)
    positive_counts = Counter({label: sum(int(row[label]) for row in candidates) for label in TARGET_LABELS})
    # Request an equal positive quota per class, then greedily select records that
    # cover the largest unmet quota. Rarer labels are weighted more heavily.
    quota = max(1, min(15, sample_size // len(TARGET_LABELS)))
    selected: list[dict[str, str | int]] = []
    covered = Counter()
    remaining = candidates.copy()
    while remaining and len(selected) < sample_size:
        def score(row: dict[str, str | int]) -> tuple[float, float]:
            unmet = [
                label for label in TARGET_LABELS
                if int(row[label]) and covered[label] < quota
            ]
            rarity = sum(1 / positive_counts[label] for label in unmet)
            return (float(len(unmet)), rarity)

        best = max(remaining, key=score)
        if score(best) == (0.0, 0):
            break
        selected.append(best)
        for label in TARGET_LABELS:
            covered[label] += int(best[label])
        remaining.remove(best)

    # Fill with a reproducible mixture of target-negative and remaining records.
    negatives = [row for row in remaining if not any(int(row[label]) for label in TARGET_LABELS)]
    rng.shuffle(negatives)
    for row in negatives:
        if len(selected) >= sample_size:
            break
        selected.append(row)
        remaining.remove(row)
    rng.shuffle(remaining)
    selected.extend(remaining[: sample_size - len(selected)])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/dataset/multi_abnormality_labels/valid_predicted_labels.csv")
    parser.add_argument("--reports", default="data/dataset/radiology_text_reports/validation_reports.csv")
    parser.add_argument("--out", default="artifacts/ctclip_validation/test_manifest.csv")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--remote-data-root", default="/root/summer_zhl/data/valid_fixed")
    args = parser.parse_args()

    rows = load_rows(Path(args.labels), Path(args.reports), args.remote_data_root)
    selected = select_rows(rows, args.sample_size, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "study_id", "volume_name", "ct_volume_path", *TARGET_LABELS, "report_impression"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Wrote {out} with {len(selected)} cases.")
    print("Target positive counts:")
    for label in TARGET_LABELS:
        print(f"  {label}: {sum(int(row[label]) for row in selected)}")


if __name__ == "__main__":
    main()
