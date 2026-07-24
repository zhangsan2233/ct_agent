"""Create a clearly labelled, non-clinical feedback simulation from Stage-2 results.

The hidden CT-RATE report-derived weak labels are used only to exercise the
feedback state machine.  They are not radiologist feedback or image ground truth.
The selected feedback patients are disjoint from the frozen evaluation patients.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.config import Settings
from chestct_agent.feedback import FeedbackItem, FeedbackSubmission
from chestct_agent.memory import AgentMemory
from chestct_agent.schemas import AnalyzeRequest, AnalyzeResponse, LabelOutput
from chestct_agent.stage2_pipeline import LABELS


def patient_id(case_id: str) -> str:
    parts = case_id.removesuffix(".nii.gz").split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else case_id


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_case_id(row: dict) -> str | None:
    """Stage-2 inputs store case_id in metadata; generations store it at root."""
    value = row.get("case_id")
    if isinstance(value, str) and value:
        return value
    metadata = row.get("metadata")
    value = metadata.get("case_id") if isinstance(metadata, dict) else None
    return value if isinstance(value, str) and value else None


def choose(value_seed: str, fraction: float) -> bool:
    """Stable sampling independent of input row ordering."""
    number = int(hashlib.sha256(value_seed.encode("utf-8")).hexdigest()[:16], 16)
    return number / 2**64 < fraction


def make_response(case_id: str, prediction: dict, scores: dict[str, float]) -> AnalyzeResponse:
    output_labels = {item["name"]: item for item in prediction["labels"]}
    return AnalyzeResponse(
        case_id=case_id,
        labels=[
            LabelOutput(
                name=label,
                status=output_labels[label]["status"],
                confidence=float(output_labels[label].get("confidence", 0.5)),
                source="ct",
                source_scores={"ct_model": float(scores[label])},
                need_human_review=True,
            )
            for label in LABELS
        ],
        disclaimer="Research-only simulated-feedback result; not for clinical diagnosis.",
    )


def truth_from_input(item: dict) -> dict[str, int]:
    """Support both evaluator rows and compact SFT rows with target JSON."""
    truth = item.get("ground_truth")
    if isinstance(truth, dict):
        return {label: int(truth[label]) for label in LABELS if label in truth}
    messages = item.get("messages")
    if isinstance(messages, list) and len(messages) >= 3:
        target = messages[2].get("content") if isinstance(messages[2], dict) else None
        if isinstance(target, str):
            try:
                labels = json.loads(target).get("labels", [])
                by_label = {row.get("name"): row.get("status") for row in labels}
                return {label: int(by_label[label] == "positive") for label in LABELS if label in by_label}
            except (json.JSONDecodeError, AttributeError):
                pass
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Stage-2 evaluation input JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="Stage-2 prediction JSONL")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Non-sensitive simulation manifest")
    parser.add_argument("--feedback-fraction", type=float, default=0.5)
    parser.add_argument("--submit-fraction", type=float, default=0.6)
    parser.add_argument("--reject-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    if not 0 < args.feedback_fraction < 1:
        raise SystemExit("--feedback-fraction must be between 0 and 1")
    if not 0 < args.submit_fraction <= 1 or not 0 <= args.reject_fraction < 1:
        raise SystemExit("Invalid submit/reject fraction")
    if args.db.exists():
        raise SystemExit(f"Refusing to mix simulation with an existing database: {args.db}")
    if args.out.exists():
        raise SystemExit(f"Output already exists: {args.out}")

    inputs = {
        case_id: row
        for row in load_jsonl(args.inputs)
        if (case_id := row_case_id(row)) is not None
    }
    predictions = {
        row["case_id"]: row["prediction"]
        for row in load_jsonl(args.predictions)
        if row.get("json_valid") and isinstance(row.get("prediction"), dict)
    }
    usable = sorted(set(inputs) & set(predictions))
    if not usable:
        raise SystemExit("No matching valid input/prediction pairs.")
    patients = sorted({patient_id(case_id) for case_id in usable})
    shuffled = patients[:]
    random.Random(args.seed).shuffle(shuffled)
    feedback_patients = set(shuffled[:max(1, round(len(shuffled) * args.feedback_fraction))])
    frozen_patients = set(patients) - feedback_patients
    if not frozen_patients:
        raise SystemExit("Feedback split consumed all patients; no frozen evaluation set remains.")

    memory = AgentMemory(Settings(memory_db_path=args.db))
    session_id = f"simulated-feedback-{args.seed}"
    submitted = approved = rejected = 0
    changed_by_label: Counter[str] = Counter()
    feedback_cases: list[str] = []
    for case_id in usable:
        if patient_id(case_id) not in feedback_patients:
            continue
        item = inputs[case_id]
        payload = json.loads(item["messages"][1]["content"])
        scores = payload.get("ctclip_scores", {})
        truth = truth_from_input(item)
        prediction = predictions[case_id]
        if set(scores) < set(LABELS) or set(truth) < set(LABELS):
            continue
        response = make_response(case_id, prediction, scores)
        request = AnalyzeRequest(
            case_id=case_id,
            session_id=session_id,
            report_text=str(payload.get("report_impression", "")),
            question="Simulated feedback workflow validation.",
        )
        memory.record(request, response, plan=None)
        predicted = {label.name: label.status for label in response.labels}
        corrections = [
            FeedbackItem(
                label=label,
                corrected_status="positive" if int(truth[label]) else "negative",
                reason="SIMULATED_REVIEW: hidden CT-RATE report-derived weak label; not clinical feedback.",
            )
            for label in LABELS
            if predicted[label] != ("positive" if int(truth[label]) else "negative")
            and choose(f"submit:{args.seed}:{case_id}:{label}", args.submit_fraction)
        ]
        if not corrections:
            continue
        feedback_cases.append(case_id)
        events = memory.submit_feedback(
            case_id,
            FeedbackSubmission(
                session_id=session_id,
                reviewer="simulated-weak-label-generator",
                reviewer_role="administrator",
                model_version="stage2-ctclip-500-2ep-simulation",
                items=corrections,
            ),
        )
        submitted += len(events)
        for event, correction in zip(events, corrections, strict=True):
            status = "rejected" if choose(
                f"reject:{args.seed}:{case_id}:{correction.label}", args.reject_fraction
            ) else "approved"
            memory.review_feedback(
                event["id"], status, "simulated-reviewer", "Non-clinical workflow test.")
            if status == "approved":
                approved += 1
                changed_by_label[correction.label] += 1
            else:
                rejected += 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "simulation_only": True,
        "notice": "CT-RATE report-derived weak labels simulate workflow feedback; they are not clinician feedback or CT-image ground truth.",
        "source_inputs": str(args.inputs),
        "source_predictions": str(args.predictions),
        "usable_cases": len(usable),
        "feedback_patients": len(feedback_patients),
        "frozen_evaluation_patients": len(frozen_patients),
        "feedback_cases_with_submissions": len(feedback_cases),
        "frozen_case_ids": [case_id for case_id in usable if patient_id(case_id) in frozen_patients],
        "submitted_events": submitted,
        "approved_events": approved,
        "rejected_events": rejected,
        "approved_changes_by_label": dict(sorted(changed_by_label.items())),
        "seed": args.seed,
        "patient_overlap": sorted(feedback_patients & frozen_patients),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
