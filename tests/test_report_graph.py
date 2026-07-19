from chestct_agent.agent.planner import ToolPolicy
from chestct_agent.schemas import AnalyzeRequest
from chestct_agent.tools.report_graph import (
    build_rule_fallback_graph,
    parse_radgraph_annotations,
    report_graph_to_evidence,
)


def sample_annotations() -> dict:
    return {
        "0": {
            "text": (
                "Small bilateral pleural effusions are present . No pulmonary nodule . "
                "Right lower lobe opacity may represent atelectasis ."
            ),
            "entities": {
                "1": {
                    "tokens": "bilateral pleural",
                    "label": "Anatomy::definitely present",
                    "start_ix": 1,
                    "end_ix": 2,
                    "relations": [],
                },
                "2": {
                    "tokens": "effusions",
                    "label": "Observation::definitely present",
                    "start_ix": 3,
                    "end_ix": 3,
                    "relations": [["located_at", "1"]],
                },
                "3": {
                    "tokens": "pulmonary",
                    "label": "Anatomy::definitely present",
                    "start_ix": 8,
                    "end_ix": 8,
                    "relations": [],
                },
                "4": {
                    "tokens": "nodule",
                    "label": "Observation::definitely absent",
                    "start_ix": 9,
                    "end_ix": 9,
                    "relations": [["located_at", "3"]],
                },
                "5": {
                    "tokens": "opacity",
                    "label": "Observation::definitely present",
                    "start_ix": 14,
                    "end_ix": 14,
                    "relations": [["located_at", "7"], ["suggestive_of", "6"]],
                },
                "6": {
                    "tokens": "atelectasis",
                    "label": "Observation::uncertain",
                    "start_ix": 17,
                    "end_ix": 17,
                    "relations": [],
                },
                "7": {
                    "tokens": "Right lower lobe",
                    "label": "Anatomy::definitely present",
                    "start_ix": 11,
                    "end_ix": 13,
                    "relations": [],
                },
            },
        }
    }


def test_radgraph_annotations_preserve_entities_relations_and_assertions():
    report = (
        "Small bilateral pleural effusions are present. No pulmonary nodule. "
        "Right lower lobe opacity may represent atelectasis."
    )
    graph = parse_radgraph_annotations(
        sample_annotations(), report, "modern-radgraph-xl"
    )
    by_text = {node.text: node for node in graph.nodes}

    assert graph.degraded is False
    assert len(graph.nodes) == 7
    assert len(graph.edges) == 4
    assert by_text["effusions"].canonical_label == "pleural_effusion"
    assert by_text["nodule"].canonical_label == "pulmonary_nodule"
    assert by_text["nodule"].assertion == "definitely_absent"
    assert by_text["opacity"].canonical_label == "lung_opacity"
    assert by_text["atelectasis"].assertion == "uncertain"
    assert any(edge.relation == "suggestive_of" for edge in graph.edges)


def test_radgraph_observations_become_typed_report_evidence():
    report = "No pulmonary nodule. Right lower lobe opacity may represent atelectasis."
    graph = parse_radgraph_annotations(
        sample_annotations(), report, "modern-radgraph-xl"
    )
    evidence = report_graph_to_evidence(graph)

    assert evidence["pulmonary_nodule"][0].polarity == "negative"
    assert evidence["pulmonary_nodule"][0].source == "radgraph_xl"
    assert evidence["atelectasis"][0].polarity == "uncertain"


def test_rule_graph_is_explicitly_marked_as_degraded():
    graph = build_rule_fallback_graph("No pleural effusion.", "model unavailable")

    assert graph.backend == "rule_fallback"
    assert graph.degraded is True
    assert graph.warning == "model unavailable"
    assert graph.edges == []


def test_report_graph_tool_is_mandatory_only_for_report_input():
    report_request = AnalyzeRequest(case_id="report", report_text="No pleural effusion.")
    ct_request = AnalyzeRequest(case_id="ct", ct_volume_path="case.nii.gz")

    assert "report_graph_tool" in ToolPolicy.fallback_tools(report_request)
    assert "report_graph_tool" not in ToolPolicy.fallback_tools(ct_request)
