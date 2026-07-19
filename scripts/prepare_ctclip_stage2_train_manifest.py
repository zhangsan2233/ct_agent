"""Create a reproducible CT-RATE training subset manifest for Stage-2 fusion.

Each row includes its expected server location and gated Hugging Face download
URL.  The script deliberately does not download volumes.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random


LABELS = [
    "arterial_wall_calcification", "atelectasis", "coronary_artery_wall_calcification",
    "emphysema", "lung_opacity", "lymphadenopathy", "pulmonary_fibrotic_sequela", "pulmonary_nodule",
]
COLUMNS = {
    "arterial_wall_calcification": "Arterial wall calcification", "atelectasis": "Atelectasis",
    "coronary_artery_wall_calcification": "Coronary artery wall calcification", "emphysema": "Emphysema",
    "lung_opacity": "Lung opacity", "lymphadenopathy": "Lymphadenopathy",
    "pulmonary_fibrotic_sequela": "Pulmonary fibrotic sequela", "pulmonary_nodule": "Lung nodule",
}


def paths(volume_name: str, remote_data_root: str) -> tuple[str, str, str]:
    stem = volume_name.removesuffix(".nii.gz")
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "train":
        raise ValueError(f"Unexpected training volume name: {volume_name}")
    group, study = "_".join(parts[:2]), "_".join(parts[:3])
    relative = f"dataset/train_fixed/{group}/{study}/{volume_name}"
    return f"{remote_data_root.rstrip('/')}/{group}/{study}/{volume_name}", relative, (
        "https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/resolve/main/" + relative + "?download=true"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/dataset/multi_abnormality_labels/train_predicted_labels.csv", type=Path)
    parser.add_argument("--reports", default="data/dataset/radiology_text_reports/train_reports.csv", type=Path)
    parser.add_argument("--out", default="artifacts/ctclip_stage2/train_manifest_1000.csv", type=Path)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--remote-data-root", default="/root/summer_zhl/data/train_fixed")
    args = parser.parse_args()
    with args.reports.open(encoding="utf-8-sig", newline="") as handle:
        reports = {row["VolumeName"]: row.get("Impressions_EN", "").strip() for row in csv.DictReader(handle)}
    with args.labels.open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    # One reconstruction per study prevents patient/study duplicates in Stage 2.
    candidates: dict[str, dict] = {}
    for row in sorted(source, key=lambda item: item["VolumeName"]):
        name = row["VolumeName"]
        candidates.setdefault(name.removesuffix(".nii.gz").rsplit("_", 1)[0], row)
    rows = list(candidates.values())
    if args.sample_size > len(rows):
        raise SystemExit(f"Requested {args.sample_size} studies but only {len(rows)} are available.")
    rng = random.Random(args.seed)
    positives = [row for row in rows if any(int(float(row[COLUMNS[label]])) for label in LABELS)]
    negatives = [row for row in rows if row not in positives]
    rng.shuffle(positives); rng.shuffle(negatives)
    selected = positives[: min(len(positives), args.sample_size * 3 // 4)]
    selected.extend(negatives[: args.sample_size - len(selected)])
    if len(selected) < args.sample_size:
        remaining = [row for row in rows if row not in selected]
        rng.shuffle(remaining); selected.extend(remaining[: args.sample_size - len(selected)])
    rng.shuffle(selected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "study_id", "volume_name", "ct_volume_path", "hf_relative_path", "download_url", *LABELS, "report_impression"]
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in selected:
            name = row["VolumeName"]
            remote, relative, url = paths(name, args.remote_data_root)
            item = {"case_id": name.removesuffix(".nii.gz"), "study_id": name.removesuffix(".nii.gz").rsplit("_", 1)[0],
                    "volume_name": name, "ct_volume_path": remote, "hf_relative_path": relative, "download_url": url,
                    "report_impression": reports.get(name, "")}
            item.update({label: int(float(row[COLUMNS[label]])) for label in LABELS})
            writer.writerow(item)
    print(f"Wrote {args.out} with {len(selected)} unique studies.")


if __name__ == "__main__":
    main()
