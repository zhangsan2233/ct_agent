import json

from chestct_agent.memory_gate import gate_memory_change, load_tool_thresholds


def proposal(**overrides):
    values = {
        "before_status": "positive",
        "proposed_status": "negative",
        "confidence": 0.95,
        "memory_ids": ["consolidation:FP"],
        "supporting_slice_indices": [10, 11],
        "visible_evidence": "Volume loss is visible on two adjacent slices.",
        "tool_score": 0.30,
        "tool_threshold": 0.475,
    }
    values.update(overrides)
    return gate_memory_change(**values)


def test_accepts_current_case_change_when_memory_and_tool_agree():
    decision = proposal()

    assert decision.accepted
    assert decision.final_status == "negative"
    assert decision.reasons == ()


def test_vetoes_memory_downgrade_when_independent_tool_remains_positive():
    decision = proposal(tool_score=0.82)

    assert not decision.accepted
    assert decision.final_status == "positive"
    assert "independent_tool_corroboration_veto" in decision.reasons


def test_requires_audited_memory_and_two_current_slices():
    decision = proposal(memory_ids=[], supporting_slice_indices=[10])

    assert not decision.accepted
    assert "no_matching_audited_memory" in decision.reasons
    assert "fewer_than_required_current_slices" in decision.reasons


def test_loads_only_finite_unit_interval_thresholds(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "thresholds": {
                    "consolidation": 0.475,
                    "bad_high": 1.2,
                    "bad_text": "not-a-number",
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_tool_thresholds(path) == {"consolidation": 0.475}
