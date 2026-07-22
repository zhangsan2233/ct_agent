"""Regression test for the approved-feedback to Stage-2 SFT export boundary."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from chestct_agent.config import Settings
from chestct_agent.feedback import FeedbackItem, FeedbackSubmission
from chestct_agent.memory import AgentMemory
from chestct_agent.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ExecutionMetadata,
    HumanApproval,
    LabelOutput,
)
from chestct_agent.stage2_pipeline import LABELS


def test_builds_stage2_sft_only_from_approved_feedback(tmp_path: Path) -> None:
    memory = AgentMemory(Settings(memory_db_path=str(tmp_path / "memory.sqlite3")))
    case_id, session_id = "train_1234_a_1", "feedback-test"
    response = AnalyzeResponse(
        case_id=case_id,
        labels=[
            LabelOutput(
                name=label,
                status="negative",
                confidence=0.2,
                source="ct",
                source_scores={"ct_model": 0.1 + number / 100},
            )
            for number, label in enumerate(LABELS)
        ],
        disclaimer="Research-only.",
        execution=ExecutionMetadata(input_mode="report_and_ct"),
        approval=HumanApproval(required=True, status="pending"),
    )
    request = AnalyzeRequest(
        case_id=case_id,
        session_id=session_id,
        report_text="Small pulmonary nodule is described.",
        ct_volume_path="/controlled/input.nii.gz",
    )
    memory.record(request, response, plan=None)
    event = memory.submit_feedback(
        case_id,
        FeedbackSubmission(
            session_id=session_id,
            reviewer="clinician-a",
            reviewer_role="clinician",
            model_version="stage2-test",
            items=[
                FeedbackItem(
                    label="pulmonary_nodule",
                    corrected_status="positive",
                    reason="Reviewed finding.",
                )
            ],
        ),
    )[0]
    memory.review_feedback(event["id"], "approved", "reviewer-b", "confirmed")

    out_dir = tmp_path / "sft"
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_feedback_sft.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--db", str(memory.path), "--out-dir", str(out_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"examples": 1' in completed.stdout
    records = [
        json.loads(line)
        for path in (out_dir / "train.jsonl", out_dir / "valid.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    target = json.loads(records[0]["messages"][2]["content"])
    statuses = {item["name"]: item["status"] for item in target["labels"]}
    assert statuses["pulmonary_nodule"] == "positive"
    assert records[0]["metadata"]["human_reviewed"] is True
