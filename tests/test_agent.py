import pytest

from chestct_agent.agent.graph import ChestCtAgent
from chestct_agent.schemas import AgentState, AnalyzeRequest, LabelPrediction
from chestct_agent.tools.consistency_checker import fuse_predictions


@pytest.mark.asyncio
async def test_agent_returns_structured_output():
    request = AnalyzeRequest(
        case_id="test_case",
        report_text="Findings: Linear atelectasis is present in both lung parenchyma. No pneumothorax.",
        question="What abnormalities are present?",
    )
    response = await ChestCtAgent().run(AgentState(request=request))
    assert response.case_id == "test_case"
    assert response.disclaimer
    assert response.labels
    assert any(label.name == "atelectasis" for label in response.labels)
    assert "medical_rag_tool" in response.tool_trace


def test_fusion_keeps_ct_only_positive_label():
    report = [
        LabelPrediction(name="atelectasis", status="negative", confidence=0.1, source="report")
    ]
    ct = [
        LabelPrediction(name="pulmonary_nodule", status="positive", confidence=0.8, source="ct")
    ]
    fused, warnings = fuse_predictions(report, ct)
    by_label = {item.name: item for item in fused}
    assert by_label["pulmonary_nodule"].status == "positive"
    assert warnings == []
