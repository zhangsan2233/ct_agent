from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    requires_report: bool = False
    requires_ct: bool = False
    risk: Literal["low", "medium", "high"] = "low"
    optional: bool = False
    timeout_seconds: float = 60.0


TOOL_REGISTRY: dict[str, ToolSpec] = {
    item.name: item
    for item in (
        ToolSpec("report_parser_tool", "Parse findings and impression.", requires_report=True),
        ToolSpec("text_classifier_tool", "Predict 18 labels from report text.", requires_report=True),
        ToolSpec(
            "report_graph_tool",
            "Extract RadGraph-XL anatomy/observation entities and clinical relations.",
            requires_report=True,
            risk="medium",
            timeout_seconds=180.0,
        ),
        ToolSpec(
            "ct_classifier_tool",
            "Run CT-CLIP 18-label volume classification.",
            requires_ct=True,
            risk="medium",
            timeout_seconds=600.0,
        ),
        ToolSpec(
            "organ_segmentation_tool",
            "Load aligned RadGenome anatomy or region masks.",
            requires_ct=True,
            optional=True,
            timeout_seconds=120.0,
        ),
        ToolSpec(
            "lesion_grounding_tool",
            "Ground findings to anatomy masks or explicitly marked weak evidence.",
            requires_ct=True,
            risk="medium",
            optional=True,
            timeout_seconds=120.0,
        ),
        ToolSpec("medical_rag_tool", "Retrieve medical knowledge with sources.", optional=True),
        ToolSpec(
            "similar_case_retriever_tool",
            "Retrieve patient-deduplicated CT-RATE training cases from report text, predicted CT conditions, and grounded anatomy.",
            optional=True,
        ),
        ToolSpec("evidence_extractor_tool", "Extract positive and negative report evidence."),
        ToolSpec("consistency_checker_tool", "Check CT/report conflicts.", risk="medium"),
        ToolSpec("structured_output_generator", "Build the typed JSON response.", risk="medium"),
        ToolSpec("json_validator_tool", "Validate and repair the response contract."),
        ToolSpec("human_approval_gate", "Escalate high-risk or unsupported results.", risk="high"),
        ToolSpec(
            "human_correction_tool",
            "Apply a complete doctor-reviewed 18-label correction event.",
            risk="high",
            optional=True,
        ),
        ToolSpec(
            "dataset_oracle_tool",
            "Reveal CT-RATE weak labels inside a leakage-marked training sandbox.",
            risk="high",
            optional=True,
        ),
        ToolSpec("explanation_generator", "Generate a grounded Chinese explanation."),
    )
}
