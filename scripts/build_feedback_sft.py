"""Build Stage-2 SFT JSONL from approved, versioned human feedback in SQLite.

Run only in a controlled environment: the output may contain report text and must
remain ignored by Git.  This script never includes pending or rejected feedback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys

# Permit ``python scripts/build_feedback_sft.py`` from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.stage2_pipeline import LABELS, SYSTEM_PROMPT


def patient_id(case_id: str) -> str:
    parts = case_id.removesuffix(".nii.gz").split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else case_id


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1")
    if not args.db.is_file():
        raise SystemExit(f"Feedback database not found: {args.db}")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.out_dir}")

    with sqlite3.connect(args.db) as connection:
        events = connection.execute(
            """SELECT id, created_at, session_id, case_id, model_version, label, corrected_status,
                      response_snapshot_json
            FROM feedback_events WHERE status='approved' ORDER BY created_at, id"""
        ).fetchall()
        contexts = {
            (session_id, case_id): (request_json, response_json)
            for session_id, case_id, request_json, response_json in connection.execute(
                "SELECT session_id, case_id, request_json, response_json FROM case_contexts"
            )
        }

    grouped: dict[tuple[str, str], list[tuple]] = {}
    for event in events:
        grouped.setdefault((event[2], event[3]), []).append(event)
    skipped: dict[str, int] = {"missing_context": 0, "missing_report": 0, "missing_ct_scores": 0}
    examples: list[dict] = []
    for key, rows in grouped.items():
        context = contexts.get(key)
        if context is None:
            skipped["missing_context"] += 1
            continue
        request, _current_response = (json.loads(item) for item in context)
        report = str(request.get("report_text", "")).strip()
        if not report:
            skipped["missing_report"] += 1
            continue
        # Use the immutable submission snapshot as the baseline.  The mutable
        # case context may later contain a displayed correction and must not
        # silently become the training target without its review event.
        response = json.loads(rows[0][7])
        by_label = {item.get("name"): item for item in response.get("labels", [])}
        scores = {
            label: by_label.get(label, {}).get("source_scores", {}).get("ct_model")
            for label in LABELS
        }
        if any(not isinstance(value, (int, float)) for value in scores.values()):
            skipped["missing_ct_scores"] += 1
            continue
        corrections = {row[5]: row[6] for row in rows}
        target_labels = []
        for label in LABELS:
            original = by_label[label]
            target_labels.append(
                {
                    "name": label,
                    "status": corrections.get(label, original["status"]),
                    "confidence": float(original.get("confidence", 0.5)),
                    "ctclip_score": round(float(scores[label]), 4),
                }
            )
        case_id = key[1]
        payload = {
            "case_id": case_id,
            "task": "Integrate report and CT-CLIP evidence into the required compact JSON.",
            "report_impression": report[:3500],
            "ctclip_scores": {label: round(float(scores[label]), 4) for label in LABELS},
            "ctclip_score_definition": "probability of the finding being present",
            "label_provenance": "human-reviewed feedback; still research-only and not CT image gold-standard annotation",
        }
        target = {
            "case_id": case_id,
            "labels": target_labels,
            "need_human_review": True,
            "disclaimer": "Research-only feedback-supervised output; not for clinical diagnosis.",
        }
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                ],
                "metadata": {
                    "case_id": case_id,
                    "patient_id": patient_id(case_id),
                    "feedback_event_ids": [row[0] for row in rows],
                    "source_model_versions": sorted({row[4] for row in rows}),
                    "human_reviewed": True,
                },
            }
        )
    patients = sorted({item["metadata"]["patient_id"] for item in examples})
    random.Random(args.seed).shuffle(patients)
    valid_patients = set(patients[:max(1, round(len(patients) * args.val_fraction))]) if patients else set()
    train = [item for item in examples if item["metadata"]["patient_id"] not in valid_patients]
    valid = [item for item in examples if item["metadata"]["patient_id"] in valid_patients]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)
    digest = (args.out_dir / "train.jsonl").read_bytes() + (args.out_dir / "valid.jsonl").read_bytes()
    manifest = {
        "approved_feedback_events": len(events),
        "examples": len(examples),
        "train_examples": len(train),
        "valid_examples": len(valid),
        "train_patients": len({item["metadata"]["patient_id"] for item in train}),
        "valid_patients": len(valid_patients),
        "patient_overlap": sorted({item["metadata"]["patient_id"] for item in train} & valid_patients),
        "skipped_cases": skipped,
        "seed": args.seed,
        "dataset_sha256": hashlib.sha256(digest).hexdigest(),
        "notice": "Controlled local artifact. Do not commit report text, feedback SFT JSONL, or patient data.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
