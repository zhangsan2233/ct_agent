"""Build compact, patient-disjoint Stage-2 SFT records from CT-CLIP outputs.

The supervision labels originate from CT-RATE's report-derived labels.  They
are useful for teaching the adapter how to consume CT-CLIP evidence, but are
not radiologist-adjudicated image diagnosis ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


LABELS = [
    "arterial_wall_calcification",
    "atelectasis",
    "coronary_artery_wall_calcification",
    "emphysema",
    "lung_opacity",
    "lymphadenopathy",
    "pulmonary_fibrotic_sequela",
    "pulmonary_nodule",
]

SYSTEM = (
    "You are ChestCT-Agent, a research-only chest CT evidence integrator. "
    "Return compact JSON only. Use only the supplied report impression and "
    "CT-CLIP scores; do not invent findings. Preserve uncertainty and require "
    "human review because labels are weak supervision."
)


def patient_id(case_id: str) -> str:
    """Map train_1234_a_1 and train_1234_b_1 to one patient/study group."""
    parts = case_id.split("_")
    return "_".join(parts[:2])


def compact_scores(scores: dict[str, float]) -> dict[str, float]:
    return {label: round(float(scores[label]), 4) for label in LABELS}


def make_example(row: dict) -> dict:
    case_id = str(row["case_id"])
    scores = compact_scores(row["probabilities"])
    truth = {label: int(row["ground_truth"][label]) for label in LABELS}
    user_payload = {
        "case_id": case_id,
        "task": "Integrate report and CT-CLIP evidence into the required compact JSON.",
        "report_impression": str(row.get("report_impression", ""))[:3500],
        "ctclip_scores": scores,
        "ctclip_score_definition": "probability of the finding being present",
        "label_provenance": "report-derived weak labels, not radiologist-adjudicated CT ground truth",
    }
    target = {
        "case_id": case_id,
        "labels": [
            {
                "name": label,
                "status": "positive" if truth[label] else "negative",
                "confidence": 0.98 if truth[label] else 0.02,
                "ctclip_score": scores[label],
            }
            for label in LABELS
        ],
        "need_human_review": True,
        "disclaimer": "Research-only weak-supervision output; not for clinical diagnosis.",
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "metadata": {
            "case_id": case_id,
            "patient_id": patient_id(case_id),
            "weak_supervision": True,
            "ctclip_available": True,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1")

    raw_rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in raw_rows if not row.get("error") and row.get("probabilities")]
    if not rows:
        raise SystemExit("No successful CT-CLIP predictions found.")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise SystemExit("Duplicate case_id in predictions.")
    missing = [label for row in rows for label in LABELS if label not in row["probabilities"] or label not in row["ground_truth"]]
    if missing:
        raise SystemExit(f"Prediction rows lack required labels, first: {missing[0]}")

    groups = sorted({patient_id(str(row["case_id"])) for row in rows})
    random.Random(args.seed).shuffle(groups)
    valid_groups = set(groups[:max(1, round(len(groups) * args.val_fraction))])
    examples = [make_example(row) for row in rows]
    train = [row for row in examples if row["metadata"]["patient_id"] not in valid_groups]
    valid = [row for row in examples if row["metadata"]["patient_id"] in valid_groups]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)
    manifest = {
        "source_predictions": str(args.predictions),
        "successful_predictions": len(rows),
        "train_examples": len(train),
        "valid_examples": len(valid),
        "labels": LABELS,
        "train_patients": len({row["metadata"]["patient_id"] for row in train}),
        "valid_patients": len(valid_groups),
        "patient_overlap": sorted(
            {row["metadata"]["patient_id"] for row in train}
            & {row["metadata"]["patient_id"] for row in valid}
        ),
        "weak_supervision_notice": "CT-RATE report-derived labels; not CT image gold-standard annotations.",
    }
    digest = (args.out_dir / "train.jsonl").read_bytes() + (args.out_dir / "valid.jsonl").read_bytes()
    manifest["dataset_sha256"] = hashlib.sha256(digest).hexdigest()
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
