import json

from chestct_agent.config import Settings
from chestct_agent.llm import QwenClient
from chestct_agent.schemas import AnalyzeRequest, ToolPlan, ToolPlanStep
from chestct_agent.tools.registry import TOOL_REGISTRY


class ToolPolicy:
    """Deterministic allow-list and mandatory steps for medical tool planning."""

    @staticmethod
    def requests_grounding(request: AnalyzeRequest) -> bool:
        question = request.question.lower()
        return any(
            term in question
            for term in (
                "where",
                "location",
                "slice",
                "mask",
                "bbox",
                "bounding box",
                "定位",
                "位置",
                "区域",
                "图像证据",
            )
        )

    @staticmethod
    def requests_knowledge(request: AnalyzeRequest) -> bool:
        question = request.question.lower()
        return any(
            term in question
            for term in ("definition", "knowledge", "explain", "医学知识", "定义", "解释")
        )

    @staticmethod
    def requests_similar_cases(request: AnalyzeRequest) -> bool:
        question = request.question.lower()
        return any(term in question for term in ("similar case", "similar cases", "相似病例"))

    @staticmethod
    def allowed(request: AnalyzeRequest) -> list[str]:
        allowed = {
            "medical_rag_tool",
            "similar_case_retriever_tool",
            "consistency_checker_tool",
            "structured_output_generator",
            "json_validator_tool",
            "human_approval_gate",
            "explanation_generator",
        }
        if request.report_text.strip():
            allowed |= {
                "report_parser_tool",
                "text_classifier_tool",
                "report_graph_tool",
                "evidence_extractor_tool",
            }
        if request.ct_volume_path:
            allowed |= {
                "ct_classifier_tool",
                "nodule_segmentation_tool",
                "effusion_segmentation_tool",
                "qwen_slice_vqa_tool",
                "ct_attribution_tool",
            }
            if ToolPolicy.requests_grounding(request):
                allowed |= {"organ_segmentation_tool", "lesion_grounding_tool"}
        return [name for name in TOOL_REGISTRY if name in allowed]

    @staticmethod
    def fallback_tools(request: AnalyzeRequest) -> list[str]:
        tools: list[str] = []
        if request.report_text.strip():
            tools += [
                "report_parser_tool",
                "text_classifier_tool",
                "report_graph_tool",
                "evidence_extractor_tool",
            ]
        if request.ct_volume_path:
            tools += [
                "ct_classifier_tool",
                "nodule_segmentation_tool",
                "effusion_segmentation_tool",
                "qwen_slice_vqa_tool",
                "ct_attribution_tool",
            ]
            if ToolPolicy.requests_grounding(request):
                tools += ["organ_segmentation_tool", "lesion_grounding_tool"]
        tools += ["medical_rag_tool"]
        if request.report_text.strip() or request.ct_volume_path:
            tools += ["similar_case_retriever_tool"]
        tools += [
            "consistency_checker_tool",
            "structured_output_generator",
            "json_validator_tool",
            "human_approval_gate",
            "explanation_generator",
        ]
        return tools

    @classmethod
    def sanitize(cls, request: AnalyzeRequest, proposed: list[str]) -> list[str]:
        allowed = set(cls.allowed(request))
        result = [name for name in proposed if name in allowed]
        if cls.requests_grounding(request):
            for name in ("organ_segmentation_tool", "lesion_grounding_tool"):
                if name in allowed and name not in result:
                    result.append(name)
        if (
            "lesion_grounding_tool" in result
            and "organ_segmentation_tool" in allowed
            and "organ_segmentation_tool" not in result
        ):
            result.append("organ_segmentation_tool")
        if (
            cls.requests_knowledge(request)
            and "medical_rag_tool" in allowed
            and "medical_rag_tool" not in result
        ):
            result.append("medical_rag_tool")
        if (
            cls.requests_similar_cases(request)
            and "similar_case_retriever_tool" in allowed
            and "similar_case_retriever_tool" not in result
        ):
            result.append("similar_case_retriever_tool")
        mandatory = cls.fallback_tools(request)
        for name in mandatory:
            spec = TOOL_REGISTRY[name]
            if not spec.optional and name not in result:
                result.append(name)
        order = {name: index for index, name in enumerate(TOOL_REGISTRY)}
        return sorted(set(result), key=order.get)


class DynamicToolPlanner:
    def __init__(self, settings: Settings, qwen: QwenClient):
        self.settings = settings
        self.qwen = qwen

    async def plan(self, request: AnalyzeRequest) -> ToolPlan:
        fallback_tools = ToolPolicy.fallback_tools(request)
        fallback = {
            "objective": request.question,
            "tools": fallback_tools,
        }
        generated_by = "policy_fallback"
        fallback_reason: str | None = "dynamic_planning_disabled"
        proposed = fallback_tools
        if self.settings.agent_dynamic_planning:
            call = await self.qwen.chat_json(
                system=(
                    "Select only necessary tools for a chest CT coursework agent. "
                    "Return JSON with objective and tools. Never invent a tool."
                ),
                user=json.dumps(
                    {
                        "question": request.question,
                        "has_report": bool(request.report_text.strip()),
                        "has_ct": bool(request.ct_volume_path),
                        "allowed_tools": ToolPolicy.allowed(request),
                    },
                    ensure_ascii=False,
                ),
                fallback=fallback,
            )
            proposed = [str(name) for name in call.value.get("tools", fallback_tools)]
            generated_by = "llm" if call.used_remote else "policy_fallback"
            fallback_reason = call.fallback_reason
        tools = ToolPolicy.sanitize(request, proposed)
        return ToolPlan(
            objective=request.question,
            generated_by=generated_by,
            fallback_reason=fallback_reason,
            steps=[
                ToolPlanStep(
                    tool=name,
                    reason=TOOL_REGISTRY[name].description,
                    required=not TOOL_REGISTRY[name].optional,
                )
                for name in tools
            ],
        )
