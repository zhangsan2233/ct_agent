from collections.abc import Awaitable, Callable

from chestct_agent.config import Settings, get_settings
from chestct_agent.llm import QwenClient
from chestct_agent.schemas import AgentState, AnalyzeResponse, LabelOutput
from chestct_agent.tools.consistency_checker import fuse_predictions
from chestct_agent.tools.ct_classifier import CtClassifierTool
from chestct_agent.tools.ct_preprocess import CtPreprocessTool
from chestct_agent.tools.evidence_extractor import extract_evidence
from chestct_agent.tools.json_validator import validate_response
from chestct_agent.tools.rag import MedicalRagTool, grade_retrieval
from chestct_agent.tools.report_parser import parse_report
from chestct_agent.tools.similar_cases import SimilarCaseRetrieverTool
from chestct_agent.tools.text_classifier import TextClassifierTool
from chestct_agent.tools.visual_evidence import build_visual_evidence


NodeFn = Callable[[dict], Awaitable[dict]]


class ChestCtAgent:
    """Controlled Agentic RAG workflow for ChestCT-Agent."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.qwen = QwenClient(self.settings)
        self.text_classifier = TextClassifierTool(self.settings)
        self.ct_preprocess = CtPreprocessTool(self.settings)
        self.ct_classifier = CtClassifierTool(self.settings)
        self.medical_rag = MedicalRagTool()
        self.similar_cases = SimilarCaseRetrieverTool(self.settings)
        self.graph = self._compile_graph()

    def _compile_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        workflow = StateGraph(dict)
        workflow.add_node("parse_input", self.parse_input)
        workflow.add_node("parse_report", self.parse_report)
        workflow.add_node("run_text_classifier", self.run_text_classifier)
        workflow.add_node("run_ct_classifier", self.run_ct_classifier)
        workflow.add_node("plan_rag_queries", self.plan_rag_queries)
        workflow.add_node("retrieve_medical_knowledge", self.retrieve_medical_knowledge)
        workflow.add_node("retrieve_similar_cases", self.retrieve_similar_cases)
        workflow.add_node("grade_retrieval", self.grade_retrieval)
        workflow.add_node("rewrite_query_if_needed", self.rewrite_query_if_needed)
        workflow.add_node("extract_evidence", self.extract_evidence)
        workflow.add_node("check_consistency", self.check_consistency)
        workflow.add_node("generate_json", self.generate_json)
        workflow.add_node("validate_output", self.validate_output)
        workflow.add_node("generate_chinese_explanation", self.generate_chinese_explanation)

        workflow.set_entry_point("parse_input")
        workflow.add_edge("parse_input", "parse_report")
        workflow.add_edge("parse_report", "run_text_classifier")
        workflow.add_edge("run_text_classifier", "run_ct_classifier")
        workflow.add_edge("run_ct_classifier", "plan_rag_queries")
        workflow.add_edge("plan_rag_queries", "retrieve_medical_knowledge")
        workflow.add_edge("retrieve_medical_knowledge", "retrieve_similar_cases")
        workflow.add_edge("retrieve_similar_cases", "grade_retrieval")
        workflow.add_edge("grade_retrieval", "rewrite_query_if_needed")
        workflow.add_edge("rewrite_query_if_needed", "extract_evidence")
        workflow.add_edge("extract_evidence", "check_consistency")
        workflow.add_edge("check_consistency", "generate_json")
        workflow.add_edge("generate_json", "validate_output")
        workflow.add_edge("validate_output", "generate_chinese_explanation")
        workflow.add_edge("generate_chinese_explanation", END)
        return workflow.compile()

    async def run(self, state: AgentState) -> AnalyzeResponse:
        initial = state.model_dump(mode="python")
        if self.graph is not None:
            result = await self.graph.ainvoke(initial)
            final_state = AgentState.model_validate(result)
        else:
            final_state = AgentState.model_validate(initial)
            for node in self._sequential_nodes():
                final_state = AgentState.model_validate(await node(final_state.model_dump(mode="python")))

        if final_state.final_response is None:
            raise RuntimeError("Agent workflow completed without final response.")
        return final_state.final_response

    def _sequential_nodes(self) -> list[NodeFn]:
        return [
            self.parse_input,
            self.parse_report,
            self.run_text_classifier,
            self.run_ct_classifier,
            self.plan_rag_queries,
            self.retrieve_medical_knowledge,
            self.retrieve_similar_cases,
            self.grade_retrieval,
            self.rewrite_query_if_needed,
            self.extract_evidence,
            self.check_consistency,
            self.generate_json,
            self.validate_output,
            self.generate_chinese_explanation,
        ]

    async def parse_input(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        state.tool_trace.append("parse_input")
        return state.model_dump(mode="python")

    async def parse_report(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        state.parsed_report = parse_report(state.request.report_text)
        state.tool_trace.append("report_parser_tool")
        return state.model_dump(mode="python")

    async def run_text_classifier(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        if state.parsed_report is None:
            state.parsed_report = parse_report(state.request.report_text)
        state.report_predictions = self.text_classifier.predict(state.parsed_report)
        state.tool_trace.append("text_classifier_tool")
        return state.model_dump(mode="python")

    async def run_ct_classifier(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        rendered = self.ct_preprocess.render_preview_slices(
            state.request.case_id, state.request.ct_volume_path
        )
        preview_images = state.request.ct_preview_images or rendered
        state.ct_preview_images = preview_images
        state.ct_predictions, ct_warnings = self.ct_classifier.predict(
            state.request.ct_volume_path, preview_images
        )
        state.consistency_warnings.extend(ct_warnings)
        state.image_evidence_by_label = build_visual_evidence(state.ct_predictions, preview_images)
        state.tool_trace.append("ct_preprocess_tool")
        state.tool_trace.append("ct_classifier_tool")
        state.tool_trace.append("visual_evidence_tool")
        return state.model_dump(mode="python")

    async def plan_rag_queries(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        positive_labels = [
            item.name
            for item in state.report_predictions + state.ct_predictions
            if item.status in {"positive", "uncertain"} and item.confidence >= 0.3
        ]
        fallback = {"queries": sorted(set(positive_labels + [state.request.question]))}
        result = await self.qwen.chat_json(
            system="You plan concise retrieval queries for a radiology Agentic RAG system.",
            user=(
                "Create retrieval queries for medical definitions, imaging findings, and report terms. "
                f"Question: {state.request.question}\nReport: {state.request.report_text[:2000]}\n"
                f"Candidate labels: {positive_labels}"
            ),
            fallback=fallback,
        )
        queries = result.get("queries", fallback["queries"])
        state.rag_queries = [str(query) for query in queries if str(query).strip()]
        state.tool_trace.append("agentic_rag_query_planner")
        return state.model_dump(mode="python")

    async def retrieve_medical_knowledge(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        state.retrieved_docs = self.medical_rag.retrieve(state.rag_queries, top_k=5)
        state.tool_trace.append("medical_rag_tool")
        return state.model_dump(mode="python")

    async def retrieve_similar_cases(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        positive_labels = [
            item.name
            for item in state.report_predictions
            if item.status in {"positive", "uncertain"} and item.confidence >= 0.3
        ]
        top_k = state.request.top_k_similar or self.settings.top_k_similar
        state.similar_cases = self.similar_cases.retrieve(
            state.request.report_text, sorted(set(positive_labels)), top_k=top_k
        )
        state.tool_trace.append("similar_case_retriever_tool")
        return state.model_dump(mode="python")

    async def grade_retrieval(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        state.retrieval_sufficient = grade_retrieval(state.retrieved_docs)
        state.tool_trace.append("retrieval_grader")
        return state.model_dump(mode="python")

    async def rewrite_query_if_needed(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        if not state.retrieval_sufficient:
            positive_labels = [
                item.name.replace("_", " ")
                for item in state.report_predictions
                if item.status in {"positive", "uncertain"}
            ]
            state.rag_queries = sorted(set(state.rag_queries + positive_labels))
            state.retrieved_docs = self.medical_rag.retrieve(state.rag_queries, top_k=5)
        state.tool_trace.append("query_rewriter")
        return state.model_dump(mode="python")

    async def extract_evidence(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        labels = [
            item.name
            for item in state.report_predictions
            if item.status in {"positive", "uncertain"} and item.confidence >= 0.3
        ]
        state.evidence_by_label = extract_evidence(state.request.report_text, labels)
        state.tool_trace.append("evidence_extractor_tool")
        return state.model_dump(mode="python")

    async def check_consistency(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        state.fusion_predictions, warnings = fuse_predictions(
            state.report_predictions, state.ct_predictions
        )
        state.consistency_warnings.extend(warnings)
        state.tool_trace.append("consistency_checker_tool")
        return state.model_dump(mode="python")

    async def generate_json(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        doc_labels = {doc.metadata.get("label") for doc in state.retrieved_docs}
        label_outputs: list[LabelOutput] = []
        for prediction in state.fusion_predictions:
            if prediction.status == "negative" and prediction.confidence < self.settings.min_label_confidence:
                continue
            report_score = next(
                (item.confidence for item in state.report_predictions if item.name == prediction.name),
                0.0,
            )
            ct_score = next(
                (item.confidence for item in state.ct_predictions if item.name == prediction.name),
                0.0,
            )
            label_outputs.append(
                LabelOutput(
                    name=prediction.name,
                    status=prediction.status,
                    confidence=prediction.confidence,
                    source_scores={"ct_model": ct_score, "report_model": report_score},
                    evidence_from_report=state.evidence_by_label.get(prediction.name, []),
                    evidence_from_image=state.image_evidence_by_label.get(prediction.name, {}),
                    rag_support=prediction.name in doc_labels,
                    need_human_review=True,
                )
            )

        response = AnalyzeResponse(
            case_id=state.request.case_id,
            labels=label_outputs,
            ct_preview_images=state.ct_preview_images,
            similar_cases=state.similar_cases,
            explanation_zh="",
            disclaimer=self.settings.disclaimer,
            tool_trace=state.tool_trace.copy(),
            warnings=state.consistency_warnings.copy(),
        )
        state.draft_response = response
        state.tool_trace.append("structured_output_generator")
        return state.model_dump(mode="python")

    async def validate_output(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        if state.draft_response is None:
            raise RuntimeError("Missing draft response before validation.")
        validated, warnings = validate_response(state.draft_response)
        validated.warnings.extend(warnings)
        validated.tool_trace = state.tool_trace.copy()
        state.final_response = validated
        state.tool_trace.append("json_validator_tool")
        return state.model_dump(mode="python")

    async def generate_chinese_explanation(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        if state.final_response is None:
            raise RuntimeError("Missing final response before explanation.")
        labels = [
            {"name": item.name, "status": item.status, "confidence": item.confidence}
            for item in state.final_response.labels
        ]
        fallback = (
            "系统已整合报告证据、CT 工具输出、检索结果和相似病例。"
            "当前结论仅用于课程设计和科研演示，必须由人工复核原始 CT 与报告。"
        )
        explanation = await self.qwen.chat_text(
            system=(
                "You are a medical imaging assistant for coursework only. "
                "Write a concise Chinese explanation grounded only in provided evidence."
            ),
            user=(
                f"Labels: {labels}\n"
                f"Evidence: {state.evidence_by_label}\n"
                f"Warnings: {state.final_response.warnings}\n"
                f"Disclaimer: {state.final_response.disclaimer}"
            ),
            fallback=fallback,
        )
        state.final_response.explanation_zh = explanation
        state.final_response.tool_trace = state.tool_trace.copy()
        state.tool_trace.append("explanation_generator")
        return state.model_dump(mode="python")
