import argparse
import asyncio
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.agent.planner import DynamicToolPlanner, ToolPolicy
from chestct_agent.config import Settings, get_settings
from chestct_agent.labels import LABEL_SPECS
from chestct_agent.llm import QwenClient
from chestct_agent.schemas import AnalyzeRequest
from chestct_agent.tools.evidence_extractor import extract_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tool planning and report evidence polarity.")
    parser.add_argument("--out", default="artifacts/evaluation/agent_capabilities.json")
    parser.add_argument("--use-llm", action="store_true")
    return parser.parse_args()


def planning_cases() -> list[tuple[AnalyzeRequest, set[str], set[str]]]:
    report = "Findings: A pulmonary nodule is present. No pleural effusion."
    ct_path = "data/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    return [
        (
            AnalyzeRequest(case_id="report", report_text=report, question="有哪些异常？"),
            {"report_parser_tool", "text_classifier_tool", "evidence_extractor_tool"},
            {"ct_classifier_tool", "organ_segmentation_tool", "lesion_grounding_tool"},
        ),
        (
            AnalyzeRequest(case_id="ct", ct_volume_path=ct_path, question="有哪些异常？"),
            {"ct_classifier_tool"},
            {
                "report_parser_tool",
                "text_classifier_tool",
                "evidence_extractor_tool",
                "organ_segmentation_tool",
                "lesion_grounding_tool",
            },
        ),
        (
            AnalyzeRequest(
                case_id="multimodal",
                report_text=report,
                ct_volume_path=ct_path,
                question="异常位于哪些区域？请检索医学知识和相似病例。",
            ),
            {
                "report_parser_tool",
                "text_classifier_tool",
                "ct_classifier_tool",
                "organ_segmentation_tool",
                "lesion_grounding_tool",
                "evidence_extractor_tool",
                "medical_rag_tool",
                "similar_case_retriever_tool",
            },
            set(),
        ),
    ]


async def evaluate_planning(use_llm: bool) -> dict[str, object]:
    settings = (
        get_settings().model_copy(update={"agent_dynamic_planning": True})
        if use_llm
        else Settings(agent_dynamic_planning=False, openai_compatible_api_key="replace-me")
    )
    planner = DynamicToolPlanner(settings, QwenClient(settings))
    true_positive = false_positive = false_negative = invalid = 0
    rows = []
    cases = planning_cases()
    plans = await asyncio.gather(*(planner.plan(request) for request, _, _ in cases))
    for (request, required, forbidden), plan in zip(cases, plans, strict=True):
        selected = {step.tool for step in plan.steps}
        true_positive += len(selected & required)
        false_negative += len(required - selected)
        false_positive += len(selected & forbidden)
        invalid += len(selected - set(ToolPolicy.allowed(request)))
        rows.append(
            {
                "case_id": request.case_id,
                "generated_by": plan.generated_by,
                "selected": sorted(selected),
                "missing_required": sorted(required - selected),
                "selected_forbidden": sorted(selected & forbidden),
            }
        )
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "cases": len(rows),
        "required_tool_precision": precision,
        "required_tool_recall": recall,
        "invalid_tool_count": invalid,
        "details": rows,
    }


def evaluate_evidence() -> dict[str, object]:
    examples: list[tuple[str, str, str]] = []
    for spec in LABEL_SPECS:
        term = spec.terms[0]
        examples.append((spec.id, f"There is {term}.", "positive"))
        examples.append((spec.id, f"No {term} is present.", "negative"))
    for spec in LABEL_SPECS[:6]:
        examples.append((spec.id, f"Possible {spec.terms[0]} is present.", "uncertain"))
        examples.append((spec.id, f"History of {spec.terms[0]}.", "historical"))

    correct = 0
    missing = 0
    details = []
    for label, sentence, expected in examples:
        found = extract_evidence(sentence, [label])[label]
        predicted = found[0].polarity if found else "missing"
        correct += int(predicted == expected)
        missing += int(not found)
        if predicted != expected:
            details.append(
                {"label": label, "sentence": sentence, "expected": expected, "predicted": predicted}
            )
    return {
        "examples": len(examples),
        "polarity_accuracy": correct / len(examples),
        "missing_evidence_rate": missing / len(examples),
        "failures": details,
    }


async def main() -> None:
    args = parse_args()
    result = {
        "planning": await evaluate_planning(args.use_llm),
        "report_evidence": evaluate_evidence(),
        "notes": [
            "Planning metrics test tool selection contracts, not diagnostic accuracy.",
            "Evidence examples are deterministic unit-evaluation sentences, not a clinical corpus.",
        ],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
