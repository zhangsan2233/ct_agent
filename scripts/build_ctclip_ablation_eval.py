"""Build paired report-only and report+CT-CLIP evaluation prompts.

The 100 existing CT-CLIP validation predictions are joined with CT-RATE
validation reports and report-derived scores.  Both arms share every field
except ``ct_model_scores`` so their generated outputs can be compared fairly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = [
    "arterial_wall_calcification", "atelectasis", "coronary_artery_wall_calcification",
    "emphysema", "lung_opacity", "lymphadenopathy", "pulmonary_fibrotic_sequela", "lung_nodule",
]
ALL_LABELS = [
    "medical_material", "arterial_wall_calcification", "cardiomegaly", "pericardial_effusion",
    "coronary_artery_wall_calcification", "hiatal_hernia", "lymphadenopathy", "emphysema",
    "atelectasis", "lung_nodule", "lung_opacity", "pulmonary_fibrotic_sequela", "pleural_effusion",
    "mosaic_attenuation_pattern", "peribronchial_thickening", "consolidation", "bronchiectasis",
    "interlobular_septal_thickening",
]


def snake(name: str) -> str:
    return name.lower().replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctclip", default="artifacts/ctclip_validation/batch/predictions.jsonl", type=Path)
    parser.add_argument("--reports", default="data/dataset/radiology_text_reports/validation_reports.csv", type=Path)
    parser.add_argument("--report-labels", default="data/dataset/multi_abnormality_labels/valid_predicted_labels.csv", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    with args.reports.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            reports[row["VolumeName"].removesuffix(".nii.gz")] = "\n".join(
                part.strip() for part in (row.get("Findings_EN", ""), row.get("Impressions_EN", "")) if part.strip()
            )
    score_rows = {}
    with args.report_labels.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            score_rows[row["VolumeName"].removesuffix(".nii.gz")] = {
                snake(key): 0.98 if int(float(value)) else 0.02
                for key, value in row.items() if key != "VolumeName"
            }
    ct_rows = [json.loads(line) for line in args.ctclip.read_text(encoding="utf-8").splitlines()]
    system = ("You are ChestCT-Agent, a research-only chest CT evidence integrator. Return JSON only. "
              "Ground every statement in supplied evidence. If CT evidence is unavailable or conflicts with report evidence, "
              "preserve uncertainty and require human review.")
    arms = {"report_only": [], "report_plus_ctclip": []}
    for row in ct_rows:
        if row.get("error") or row["case_id"] not in reports or row["case_id"] not in score_rows:
            continue
        probabilities = dict(row["probabilities"])
        probabilities["lung_nodule"] = probabilities.pop("pulmonary_nodule")
        truth = dict(row["ground_truth"])
        truth["lung_nodule"] = truth.pop("pulmonary_nodule")
        common = {
            "case_id": row["case_id"],
            "question": "Please integrate the supplied evidence into the required JSON. Do not invent CT findings.",
            "report_text": reports[row["case_id"]], "report_model_scores": score_rows[row["case_id"]],
            "label_provenance": "CT-RATE report-derived weak labels; CT-CLIP comparison uses supplied validation labels.",
        }
        for name, use_ct in (("report_only", False), ("report_plus_ctclip", True)):
            evidence = {**common, "ct_model_scores": probabilities if use_ct else None, "ct_scores_available": use_ct}
            arms[name].append({
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}],
                "ground_truth": truth, "evaluation_labels": LABELS,
                "metadata": {"case_id": row["case_id"], "arm": name},
            })
    for name, rows in arms.items():
        (args.out_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    print(json.dumps({name: len(rows) for name, rows in arms.items()}))


if __name__ == "__main__":
    main()
