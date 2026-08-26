"""Bridge Stage-2 modality results into AgentMemory for feedback workflows."""
from __future__ import annotations

from typing import Any

from chestct_agent.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ExecutionMetadata,
    HumanApproval,
    LabelOutput,
)
from chestct_agent.stage2_pipeline import LABELS


def _score_source_key(modality: str) -> str:
    return "ct_model" if modality == "ct_chest" else "cxr_model"


def stage2_result_to_response(result: dict[str, Any], modality: str) -> AnalyzeResponse:
    scores = result.get("ctclip_scores") or result.get("image_scores") or {}
    stage2_json = result.get("stage2_json") or {}
    json_labels = {
        item.get("name"): item
        for item in (stage2_json.get("labels") or [])
        if isinstance(item, dict)
    }
    score_key = _score_source_key(modality)
    labels: list[LabelOutput] = []
    for name in LABELS:
        item = json_labels.get(name, {})
        labels.append(
            LabelOutput(
                name=name,
                status=item.get("status", "uncertain"),
                confidence=float(item.get("confidence", 0.0)),
                source="ct" if modality == "ct_chest" else "fusion",
                source_scores={score_key: float(scores.get(name, 0.0))},
                need_human_review=True,
            )
        )
    adapter_dir = (result.get("provenance") or {}).get("adapter_dir", "unknown")
    return AnalyzeResponse(
        case_id=result.get("input", {}).get("case_id", "unknown"),
        labels=labels,
        explanation_zh=result.get("report_zh", result.get("summary_zh", "")),
        disclaimer=result.get("warning", ""),
        execution=ExecutionMetadata(input_mode="report_and_ct"),
        approval=HumanApproval(required=True, status="pending"),
    )


def stage2_result_to_request(result: dict[str, Any], modality: str, session_id: str) -> AnalyzeRequest:
    payload = result.get("input") or {}
    image_path = payload.get("ct_path") or payload.get("image_path")
    return AnalyzeRequest(
        case_id=str(payload.get("case_id", "unknown")),
        session_id=session_id,
        report_text=str(payload.get("report_text", "")),
        ct_volume_path=str(image_path) if modality == "ct_chest" and image_path else None,
        question="请分析该检查中的胸部异常和证据。",
    )
