"""Build a patient-disjoint weak-supervision SFT dataset for ChestCT-Agent.

Targets are derived from CT-RATE report-derived labels.  This teaches an LLM to
consume tool evidence and emit the Agent JSON schema; it is *not* a replacement
for radiologist-adjudicated diagnostic supervision.  CT-CLIP scores can be
merged later with --ct-predictions after batch inference is available.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path


LABEL_ALIASES = {
    "medical_material": "medical material",
    "arterial_wall_calcification": "arterial wall calcification",
    "cardiomegaly": "cardiomegaly",
    "pericardial_effusion": "pericardial effusion",
    "coronary_artery_wall_calcification": "coronary artery wall calcification",
    "hiatal_hernia": "hiatal hernia",
    "lymphadenopathy": "lymphadenopathy",
    "emphysema": "emphysema",
    "atelectasis": "atelectasis",
    "lung_nodule": "lung nodule",
    "lung_opacity": "lung opacity",
    "pulmonary_fibrotic_sequela": "pulmonary fibrotic sequela",
    "pleural_effusion": "pleural effusion",
    "mosaic_attenuation_pattern": "mosaic attenuation pattern",
    "peribronchial_thickening": "peribronchial thickening",
    "consolidation": "consolidation",
    "bronchiectasis": "bronchiectasis",
    "interlobular_septal_thickening": "interlobular septal thickening",
}
ZH_LABELS = {
    "medical_material": "医疗材料", "arterial_wall_calcification": "动脉壁钙化",
    "cardiomegaly": "心脏增大", "pericardial_effusion": "心包积液",
    "coronary_artery_wall_calcification": "冠状动脉壁钙化", "hiatal_hernia": "食管裂孔疝",
    "lymphadenopathy": "淋巴结肿大", "emphysema": "肺气肿", "atelectasis": "肺不张",
    "lung_nodule": "肺结节", "lung_opacity": "肺部密度增高/阴影",
    "pulmonary_fibrotic_sequela": "肺纤维化后遗改变", "pleural_effusion": "胸腔积液",
    "mosaic_attenuation_pattern": "马赛克灌注", "peribronchial_thickening": "支气管周围增厚",
    "consolidation": "实变", "bronchiectasis": "支气管扩张",
    "interlobular_septal_thickening": "小叶间隔增厚",
}


def snake(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def study_id(volume: str) -> str:
    return volume.removesuffix(".nii.gz").rsplit("_", 1)[0]


def patient_id(volume: str) -> str:
    parts = volume.removesuffix(".nii.gz").split("_")
    return "_".join(parts[:2])


def sentences_with_term(report: str, term: str) -> list[str]:
    tokens = [part.strip() for part in re.split(r"(?<=[.!?])\s+", report) if part.strip()]
    words = term.split()
    matches = [sentence for sentence in tokens if any(word in sentence.lower() for word in words)]
    return matches[:2] or ([tokens[-1]] if tokens else [])


def read_ct_predictions(path: Path | None) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    predictions: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row.get("error") and row.get("case_id") and row.get("probabilities"):
            predictions[row["case_id"]] = {snake(key): float(value) for key, value in row["probabilities"].items()}
    return predictions


def make_example(row: dict[str, str], labels: list[str], ct_scores: dict[str, dict[str, float]]) -> dict:
    volume = row["VolumeName"]
    case_id = volume.removesuffix(".nii.gz")
    report = "\n".join(part.strip() for part in (row.get("Findings_EN", ""), row.get("Impressions_EN", "")) if part.strip())
    report = report[:6000]
    weak_scores = {label: 0.98 if int(float(row[label])) else 0.02 for label in labels}
    visual_scores = ct_scores.get(case_id)
    input_evidence = {
        "case_id": case_id,
        "question": "Please integrate the supplied evidence into the required JSON. Do not invent CT findings.",
        "report_text": report,
        "report_model_scores": weak_scores,
        "ct_model_scores": visual_scores,
        "ct_scores_available": visual_scores is not None,
        "label_provenance": "report-derived weak labels; not radiologist-adjudicated CT ground truth",
    }
    outputs = []
    positives = []
    for label in labels:
        positive = int(float(row[label])) == 1
        evidence = sentences_with_term(report, LABEL_ALIASES[label]) if positive else []
        if positive:
            positives.append(ZH_LABELS[label])
        source_scores = {"report_model": weak_scores[label]}
        if visual_scores and label in visual_scores:
            source_scores["ct_model"] = visual_scores[label]
        outputs.append({
            "name": label,
            "status": "positive" if positive else "negative",
            "confidence": weak_scores[label],
            "source_scores": source_scores,
            "evidence_from_report": evidence,
            "rag_support": False,
            "need_human_review": True,
        })
    conclusion = "、".join(positives) if positives else "未见本任务标签对应的报告弱监督阳性"
    target = {
        "case_id": case_id,
        "labels": outputs,
        "explanation_zh": f"本结果基于英文报告及报告模型弱标签整合。提示：{conclusion}。未提供或未定位的影像证据不得补充推断，建议结合原始 CT 由专业人员复核。",
        "warnings": ["训练标签为报告自动生成的弱标签，不能视为影像诊断金标准。"],
        "disclaimer": "仅用于课程设计和科研演示，不用于临床诊断。",
    }
    system = (
        "You are ChestCT-Agent, a research-only chest CT evidence integrator. "
        "Return JSON only. Ground every statement in supplied evidence. "
        "If CT evidence is unavailable or conflicts with report evidence, preserve uncertainty and require human review."
    )
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(input_evidence, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ], "metadata": {"case_id": case_id, "study_id": study_id(volume), "patient_id": patient_id(volume), "weak_supervision": True}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="data/dataset/radiology_text_reports/train_reports.csv", type=Path)
    parser.add_argument("--labels", default="data/dataset/multi_abnormality_labels/train_predicted_labels.csv", type=Path)
    parser.add_argument("--ct-predictions", type=Path)
    parser.add_argument("--out-dir", default="artifacts/llm_sft", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1")
    with args.reports.open(encoding="utf-8-sig", newline="") as handle:
        reports = {row["VolumeName"]: row for row in csv.DictReader(handle)}
    with args.labels.open(encoding="utf-8-sig", newline="") as handle:
        label_rows = list(csv.DictReader(handle))
    labels = [snake(column) for column in label_rows[0] if column not in {"VolumeName", "Medical material"}]
    # Preserve all 18 CT-RATE labels, including Medical material.
    labels.insert(0, "medical_material")
    original_columns = {snake(column): column for column in label_rows[0] if column != "VolumeName"}
    canonical_rows = []
    seen_studies = set()
    for source in sorted(label_rows, key=lambda item: item["VolumeName"]):
        volume = source["VolumeName"]
        if volume not in reports or study_id(volume) in seen_studies:
            continue
        seen_studies.add(study_id(volume))
        row = {"VolumeName": volume, **reports[volume]}
        row.update({label: source[original_columns[label]] for label in labels})
        canonical_rows.append(row)
    rng = random.Random(args.seed)
    patients = sorted({patient_id(row["VolumeName"]) for row in canonical_rows})
    rng.shuffle(patients)
    val_patients = set(patients[:round(len(patients) * args.val_fraction)])
    ct_scores = read_ct_predictions(args.ct_predictions)
    examples = [make_example(row, labels, ct_scores) for row in canonical_rows]
    train = [item for item in examples if item["metadata"]["patient_id"] not in val_patients]
    valid = [item for item in examples if item["metadata"]["patient_id"] in val_patients]
    if args.max_examples:
        train, valid = train[:args.max_examples], valid[:max(1, round(args.max_examples * args.val_fraction))]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        (args.out_dir / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "train_examples": len(train), "valid_examples": len(valid), "labels": labels,
        "train_patients": len({row["metadata"]["patient_id"] for row in train}),
        "valid_patients": len({row["metadata"]["patient_id"] for row in valid}),
        "patient_overlap": sorted({row["metadata"]["patient_id"] for row in train} & {row["metadata"]["patient_id"] for row in valid}),
        "ct_scores_merged": sum(item["messages"][1]["content"].find('"ct_scores_available": true') >= 0 for item in examples),
        "weak_supervision_notice": "Labels are CT-RATE report-derived predicted labels, not CT gold-standard annotations.",
    }
    manifest["dataset_sha256"] = hashlib.sha256((args.out_dir / "train.jsonl").read_bytes() + (args.out_dir / "valid.jsonl").read_bytes()).hexdigest()
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
