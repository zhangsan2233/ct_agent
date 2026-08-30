"""Safety gates for feedback-derived diagnostic memory proposals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import json
from typing import Literal


BinaryStatus = Literal["positive", "negative"]


@dataclass(frozen=True)
class MemoryGateDecision:
    accepted: bool
    final_status: BinaryStatus
    reasons: tuple[str, ...]


def load_tool_thresholds(path: Path) -> dict[str, float]:
    """Load finite per-label thresholds from a feedback-only calibration artifact."""

    value = json.loads(path.read_text(encoding="utf-8"))
    raw = value.get("thresholds") if isinstance(value, dict) else None
    if not isinstance(raw, dict):
        label = str(value.get("label") or "") if isinstance(value, dict) else ""
        raw = {label: value.get("threshold")} if label else {}

    thresholds: dict[str, float] = {}
    for label, threshold in raw.items():
        try:
            number = float(threshold)
        except (TypeError, ValueError):
            continue
        if str(label) and math.isfinite(number) and 0.0 <= number <= 1.0:
            thresholds[str(label)] = number
    return thresholds


def gate_memory_change(
    *,
    before_status: BinaryStatus,
    proposed_status: BinaryStatus,
    confidence: float,
    memory_ids: list[str],
    supporting_slice_indices: list[int],
    visible_evidence: str,
    tool_score: float | None = None,
    tool_threshold: float | None = None,
    minimum_confidence: float = 0.75,
    minimum_slices: int = 2,
) -> MemoryGateDecision:
    """Accept a memory proposal only when current evidence and tools corroborate it.

    A memory is an inspection instruction, not patient evidence. When a calibrated
    independent tool threshold is available, memory cannot override the tool in the
    opposite direction.
    """

    reasons: list[str] = []
    if proposed_status == before_status:
        reasons.append("unchanged")
    if not memory_ids:
        reasons.append("no_matching_audited_memory")
    if len(set(supporting_slice_indices)) < minimum_slices:
        reasons.append("fewer_than_required_current_slices")
    if not math.isfinite(confidence) or confidence < minimum_confidence:
        reasons.append("confidence_below_gate")
    if not visible_evidence.strip():
        reasons.append("missing_current_image_evidence")

    if tool_score is not None or tool_threshold is not None:
        if tool_score is None or tool_threshold is None:
            reasons.append("incomplete_tool_calibration")
        elif not (
            math.isfinite(tool_score)
            and math.isfinite(tool_threshold)
            and 0.0 <= tool_score <= 1.0
            and 0.0 <= tool_threshold <= 1.0
        ):
            reasons.append("invalid_tool_calibration")
        else:
            tool_positive = tool_score >= tool_threshold
            proposal_positive = proposed_status == "positive"
            if tool_positive != proposal_positive:
                reasons.append("independent_tool_corroboration_veto")

    accepted = not reasons
    return MemoryGateDecision(
        accepted=accepted,
        final_status=proposed_status if accepted else before_status,
        reasons=tuple(reasons),
    )
