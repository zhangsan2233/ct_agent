import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import time

from chestct_agent.agent.planner import DynamicToolPlanner
from chestct_agent.calibration import CalibrationStore
from chestct_agent.config import Settings, get_settings
from chestct_agent.conversation import CaseConversationAgent
from chestct_agent.corrections import (
    apply_corrections,
    dataset_correction_request,
    load_ct_rate_reference_labels,
)
from chestct_agent.knowledge import LABEL_KNOWLEDGE, LABEL_ZH, STATUS_ZH
from chestct_agent.labels import LABEL_BY_ID, LABEL_IDS
from chestct_agent.llm import QwenClient
from chestct_agent.memory import AgentMemory
from chestct_agent.schemas import (
    AgentState,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    CorrectionRequest,
    ExecutionEvent,
    ExecutionMetadata,
    HumanApproval,
    LabelOutput,
    ModelReasoningReport,
    ModelReasoningStep,
    RagTrace,
    RetrievalAttemptTrace,
)
from chestct_agent.tools.consistency_checker import apply_credibility_gate, fuse_predictions
from chestct_agent.tools.ct_classifier import CtClassifierTool
from chestct_agent.tools.ct_preprocess import CtPreprocessTool
from chestct_agent.tools.evidence_extractor import extract_evidence
from chestct_agent.tools.json_validator import validate_response
from chestct_agent.tools.lesion_grounding import LESION_MASK_ALIASES, ground_findings
from chestct_agent.tools.organ_segmentation import OrganSegmentationTool
from chestct_agent.tools.rag import MedicalRagTool, grade_retrieval
from chestct_agent.tools.report_parser import parse_report
from chestct_agent.tools.report_graph import ReportGraphTool, report_graph_to_evidence
from chestct_agent.tools.similar_cases import SimilarCaseRetrieverTool
from chestct_agent.tools.text_classifier import TextClassifierTool
from chestct_agent.tools.registry import TOOL_REGISTRY
from chestct_agent.tools.visual_evidence import build_visual_evidence


EventCallback = Callable[[ExecutionEvent], Awaitable[None]]
_EVENT_CALLBACK: ContextVar[EventCallback | None] = ContextVar(
    "chestct_agent_event_callback", default=None
)


class ChestCtAgent:
    """Controlled Agentic RAG workflow for ChestCT-Agent."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.qwen = QwenClient(self.settings)
        self.planner = DynamicToolPlanner(self.settings, self.qwen)
        self.memory = AgentMemory(self.settings)
        self.text_classifier = TextClassifierTool(self.settings)
        self.report_graph = ReportGraphTool(self.settings)
        self.ct_preprocess = CtPreprocessTool(self.settings)
        self.ct_classifier = CtClassifierTool(self.settings)
        self.fusion_calibration = CalibrationStore(self.settings)
        self.medical_rag = MedicalRagTool(self.settings)
        self.similar_cases = SimilarCaseRetrieverTool(self.settings)
        self.organ_segmentation = OrganSegmentationTool(self.settings)
        self.conversation = CaseConversationAgent(
            self.qwen, self.memory, self.medical_rag
        )
        self.graph = self._compile_graph()

    def _compile_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        workflow = StateGraph(dict)
        nodes = {
            "parse_input": self.parse_input,
            "plan_tools": self.plan_tools,
            "parse_report": self.parse_report,
            "run_text_classifier": self.run_text_classifier,
            "run_report_graph": self.run_report_graph,
            "run_ct_classifier": self.run_ct_classifier,
            "run_organ_segmentation": self.run_organ_segmentation,
            "run_lesion_grounding": self.run_lesion_grounding,
            "plan_rag_queries": self.plan_rag_queries,
            "retrieve_medical_knowledge": self.retrieve_medical_knowledge,
            "retrieve_similar_cases": self.retrieve_similar_cases,
            "grade_retrieval": self.grade_retrieval,
            "rewrite_query_if_needed": self.rewrite_query_if_needed,
            "extract_evidence": self.extract_evidence,
            "check_consistency": self.check_consistency,
            "generate_json": self.generate_json,
            "validate_output": self.validate_output,
            "human_approval": self.human_approval,
            "generate_chinese_explanation": self.generate_chinese_explanation,
        }
        for name, node in nodes.items():
            workflow.add_node(name, self._timed_node(name, node))

        workflow.set_entry_point("parse_input")
        workflow.add_edge("parse_input", "plan_tools")
        workflow.add_conditional_edges(
            "plan_tools",
            self.route_after_plan,
            {
                "report": "parse_report",
                "ct": "run_ct_classifier",
                "rag": "plan_rag_queries",
                "similar": "retrieve_similar_cases",
                "evidence": "extract_evidence",
            },
        )
        workflow.add_edge("parse_report", "run_text_classifier")
        workflow.add_edge("run_text_classifier", "run_report_graph")
        workflow.add_conditional_edges(
            "run_report_graph",
            self.route_after_text,
            {
                "ct": "run_ct_classifier",
                "rag": "plan_rag_queries",
                "similar": "retrieve_similar_cases",
                "evidence": "extract_evidence",
            },
        )
        workflow.add_conditional_edges(
            "run_ct_classifier",
            self.route_after_ct,
            {
                "organ": "run_organ_segmentation",
                "grounding": "run_lesion_grounding",
                "rag": "plan_rag_queries",
                "similar": "retrieve_similar_cases",
                "evidence": "extract_evidence",
            },
        )
        workflow.add_conditional_edges(
            "run_organ_segmentation",
            self.route_after_organ_segmentation,
            {
                "grounding": "run_lesion_grounding",
                "rag": "plan_rag_queries",
                "similar": "retrieve_similar_cases",
                "evidence": "extract_evidence",
            },
        )
        workflow.add_conditional_edges(
            "run_lesion_grounding",
            self.route_after_analysis,
            {
                "rag": "plan_rag_queries",
                "similar": "retrieve_similar_cases",
                "evidence": "extract_evidence",
            },
        )
        workflow.add_edge("plan_rag_queries", "retrieve_medical_knowledge")
        workflow.add_edge("retrieve_medical_knowledge", "grade_retrieval")
        workflow.add_conditional_edges(
            "grade_retrieval",
            self.route_after_retrieval_grade,
            {
                "rewrite": "rewrite_query_if_needed",
                "similar": "retrieve_similar_cases",
                "evidence": "extract_evidence",
            },
        )
        workflow.add_edge("rewrite_query_if_needed", "retrieve_medical_knowledge")
        workflow.add_edge("retrieve_similar_cases", "extract_evidence")
        workflow.add_edge("extract_evidence", "check_consistency")
        workflow.add_edge("check_consistency", "generate_json")
        workflow.add_edge("generate_json", "validate_output")
        workflow.add_edge("validate_output", "human_approval")
        workflow.add_edge("human_approval", "generate_chinese_explanation")
        workflow.add_edge("generate_chinese_explanation", END)
        return workflow.compile()

    async def run(
        self,
        state: AgentState,
        event_callback: EventCallback | None = None,
    ) -> AnalyzeResponse:
        token = _EVENT_CALLBACK.set(event_callback)
        try:
            return await self._run_with_context(state)
        finally:
            _EVENT_CALLBACK.reset(token)

    async def chat(
        self,
        request: ChatRequest,
        event_callback: EventCallback | None = None,
    ) -> ChatResponse:
        # Tests and embedding applications may replace memory after construction.
        self.conversation.memory = self.memory
        return await self.conversation.answer(request, event_callback=event_callback)

    async def correct_case(
        self, case_id: str, request: CorrectionRequest
    ) -> AnalyzeResponse:
        context = self.memory.get_case_context(request.session_id, case_id)
        if context is None:
            raise LookupError("未找到该会话中的病例结果，无法应用纠错。")
        _, response = context
        before = response.model_copy(deep=True)
        corrected, event = apply_corrections(response, request)
        changed = [
            item for item in event.items if item.before_status != item.after_status
        ]
        positive = [item.name_zh for item in corrected.labels if item.status == "positive"]
        uncertain = [item.name_zh for item in corrected.labels if item.status == "uncertain"]
        source_name = "医生逐标签复核" if request.source == "human" else "隐藏弱标签沙箱"
        fallback = (
            f"{source_name}已反馈并修改{len(changed)}项。纠错后的主要发现："
            f"{'、'.join(positive) or '无'}；仍需复核：{'、'.join(uncertain) or '无'}。"
        )
        call = await self.qwen.chat_text(
            system=(
                "You rewrite a Chinese chest CT result after receiving external factual feedback. "
                "Use the corrected statuses exactly. Explain which labels changed and identify "
                "whether feedback came from a doctor or a dataset weak-label sandbox. Never claim "
                "dataset weak labels are clinical ground truth. Do not add diagnoses or evidence."
            ),
            user=json.dumps(
                {
                    "feedback_source": request.source,
                    "reviewer": request.reviewer,
                    "changes": [item.model_dump(mode="json") for item in changed],
                    "positive_labels": positive,
                    "uncertain_labels": uncertain,
                    "limitations": corrected.approval.reasons,
                },
                ensure_ascii=False,
            ),
            fallback=fallback,
            max_tokens=600,
        )
        corrected.explanation_zh = call.value.strip() or fallback
        corrected.model_reasoning.steps.append(
            ModelReasoningStep(
                order=len(corrected.model_reasoning.steps) + 1,
                stage="外部事实纠错",
                decision=f"根据{source_name}修改{len(changed)}个标签并重写结论。",
                evidence=[
                    f"{item.label}:{item.before_status}->{item.after_status}"
                    for item in changed[:12]
                ] or ["复核者确认原状态，无标签状态变化。"],
                uncertainty=(
                    "CT-RATE反馈为报告派生弱标签，不是临床金标准。"
                    if request.source == "dataset_weak_label"
                    else ""
                ),
            )
        )
        corrected.model_reasoning.summary_zh = corrected.explanation_zh
        corrected.model_reasoning.raw_response_zh = corrected.explanation_zh
        corrected.model_reasoning.generated_by = (
            "qwen" if call.used_remote else "deterministic_fallback"
        )
        corrected.execution.llm_calls += 1
        corrected.execution.llm_fallbacks += int(not call.used_remote)
        if call.fallback_reason:
            corrected.execution.llm_fallback_reasons.append(
                f"correction_explanation:{call.fallback_reason}"
            )
        corrected.execution_events.append(
            ExecutionEvent(
                sequence=len(corrected.execution_events) + 1,
                node="apply_external_correction",
                tool=(
                    "human_correction_tool"
                    if request.source == "human"
                    else "dataset_oracle_tool"
                ),
                status="success",
                summary=f"接收{source_name}，修改{len(changed)}个标签并重新生成结论",
                decision_summary="只根据外部反馈修改对应标签，不由Qwen自行判定真值。",
                decision_basis=[
                    f"反馈来源={request.source}",
                    f"复核者={request.reviewer}",
                    f"修改标签={len(changed)}",
                ],
                key_metrics={"submitted": len(event.items), "changed": len(changed)},
            )
        )
        corrected.tool_trace.append(
            "human_correction_tool"
            if request.source == "human"
            else "dataset_oracle_tool"
        )
        self.memory.record_correction(
            request.session_id, case_id, event, before, corrected
        )
        return corrected

    async def correct_case_with_dataset(
        self, case_id: str, session_id: str
    ) -> AnalyzeResponse:
        context = self.memory.get_case_context(session_id, case_id)
        if context is None:
            raise LookupError("未找到该会话中的开发病例结果。")
        analyze_request, response = context
        if not analyze_request.ct_volume_path:
            raise LookupError("训练沙箱只允许使用本地CT-RATE开发病例。")
        ct_path = Path(analyze_request.ct_volume_path).resolve()
        valid_root = (self.settings.data_dir / "dataset" / "valid_fixed").resolve()
        if valid_root not in ct_path.parents:
            raise LookupError("训练沙箱只允许使用本地CT-RATE开发病例。")
        reference = load_ct_rate_reference_labels(self.settings.data_dir, case_id)
        if reference is None:
            raise LookupError("当前病例没有可用的CT-RATE弱标签。")
        request = dataset_correction_request(response, session_id, reference)
        return await self.correct_case(case_id, request)

    async def _run_with_context(self, state: AgentState) -> AnalyzeResponse:
        started = time.perf_counter()
        state.max_retrieval_attempts = self.settings.rag_max_attempts
        initial = state.model_dump(mode="python")
        if self.graph is not None:
            result = await self.graph.ainvoke(initial)
            final_state = AgentState.model_validate(result)
        else:
            final_state = await self._run_without_langgraph(initial)

        if final_state.final_response is None:
            raise RuntimeError("Agent workflow completed without final response.")
        final_state.final_response.execution.total_latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )
        final_state.final_response.execution.node_timings_ms = final_state.node_timings_ms
        final_state.final_response.execution.planned_tools = (
            [step.tool for step in final_state.tool_plan.steps] if final_state.tool_plan else []
        )
        failed_tools = list(dict.fromkeys(final_state.failed_tools))
        final_state.final_response.execution.failed_tools = failed_tools
        final_state.final_response.execution.recovered_failures = final_state.recovered_failures
        final_state.final_response.execution.degraded = bool(
            failed_tools and final_state.recovered_failures < len(failed_tools)
        )
        final_state.final_response.agent_plan = final_state.tool_plan
        final_state.final_response.execution_events = final_state.execution_events.copy()
        final_state.final_response.rag_trace = RagTrace(
            query_history=final_state.rag_query_history,
            attempts=final_state.retrieval_history,
            final_sufficient=final_state.retrieval_sufficient,
        )
        try:
            self.memory.record(state.request, final_state.final_response, final_state.tool_plan)
        except Exception as exc:
            final_state.final_response.warnings.append(
                f"审计记忆写入失败：{type(exc).__name__}。本次分析结果未受影响。"
            )
        return final_state.final_response

    def _timed_node(self, name: str, node):
        async def wrapped(data: dict) -> dict:
            started = time.perf_counter()
            attempts = 0
            call_count = 0
            degraded = False
            error_type: str | None = None
            callback = _EVENT_CALLBACK.get()
            if callback is not None:
                try:
                    await callback(
                        ExecutionEvent(
                            sequence=len(AgentState.model_validate(data).execution_events) + 1,
                            node=name,
                            tool=self._node_tool_name(name),
                            status="running",
                            summary=self._event_start_summary(name),
                        )
                    )
                except Exception:
                    pass
            while True:
                call_count += 1
                try:
                    result = await node(data)
                    if attempts:
                        recovered_state = AgentState.model_validate(result)
                        recovered_state.recovered_failures += 1
                        result = recovered_state.model_dump(mode="python")
                    break
                except Exception as exc:
                    attempts += 1
                    error_type = type(exc).__name__
                    state = AgentState.model_validate(data)
                    tool_name = self._node_tool_name(name)
                    if tool_name not in state.failed_tools:
                        state.failed_tools.append(tool_name)
                    if attempts <= self.settings.tool_max_retries:
                        data = state.model_dump(mode="python")
                        await asyncio.sleep(min(0.25 * attempts, 1.0))
                        continue
                    spec = TOOL_REGISTRY.get(tool_name)
                    if spec is None or not spec.optional:
                        raise
                    state.consistency_warnings.append(
                        f"可选工具 {tool_name} 失败并已降级：{type(exc).__name__}"
                    )
                    degraded = True
                    result = state.model_dump(mode="python")
                    break
            state = AgentState.model_validate(result)
            elapsed_ms = (time.perf_counter() - started) * 1000
            state.node_timings_ms[name] = round(
                state.node_timings_ms.get(name, 0.0) + elapsed_ms,
                2,
            )
            status = "degraded" if degraded else "recovered" if attempts else "success"
            decision_summary, decision_basis, key_metrics = self._event_decision_audit(
                name, state
            )
            event = ExecutionEvent(
                sequence=len(state.execution_events) + 1,
                node=name,
                tool=self._node_tool_name(name),
                status=status,
                duration_ms=round(elapsed_ms, 2),
                attempts=call_count,
                summary=self._event_summary(name, state),
                decision_summary=decision_summary,
                decision_basis=decision_basis,
                key_metrics=key_metrics,
                error_type=error_type,
            )
            state.execution_events.append(event)
            if callback is not None:
                try:
                    await callback(event)
                except Exception:
                    # A disconnected observer must not invalidate a completed medical tool.
                    pass
            return state.model_dump(mode="python")

        return wrapped

    @staticmethod
    def _event_start_summary(node_name: str) -> str:
        summaries = {
            "parse_input": "正在校验输入并识别报告/CT模式",
            "plan_tools": "正在根据任务动态选择本轮工具",
            "parse_report": "正在拆分报告 Findings 和 Impression",
            "run_text_classifier": "正在执行18类报告异常分类",
            "run_report_graph": "正在抽取RadGraph-XL实体节点与临床关系",
            "run_ct_classifier": "正在预处理3D CT并运行CT-CLIP分类",
            "run_organ_segmentation": "正在读取并对齐器官分割mask",
            "run_lesion_grounding": "正在定位异常切片和解剖区域",
            "plan_rag_queries": "正在生成医学知识检索问题",
            "retrieve_medical_knowledge": "正在执行BM25、Dense和Reranker检索",
            "grade_retrieval": "正在判断检索证据是否充分",
            "rewrite_query_if_needed": "正在改写低质量检索问题",
            "retrieve_similar_cases": "正在检索非当前病例的相似检查",
            "extract_evidence": "正在抽取阳性、阴性和不确定报告证据",
            "check_consistency": "正在融合CT、报告和检索证据并检查冲突",
            "generate_json": "正在生成18类结构化结果",
            "validate_output": "正在校验JSON结构和字段约束",
            "human_approval": "正在判断是否需要医生人工审批",
            "generate_chinese_explanation": "正在生成病例中文结论和Qwen分析说明",
        }
        return summaries.get(node_name, f"正在执行 {node_name}")

    @staticmethod
    def _event_summary(node_name: str, state: AgentState) -> str:
        positive_report = sum(item.status == "positive" for item in state.report_predictions)
        positive_ct = sum(item.status == "positive" for item in state.ct_predictions)
        summaries = {
            "parse_input": "输入校验完成",
            "plan_tools": (
                f"规划 {len(state.tool_plan.steps) if state.tool_plan else 0} 个工具；"
                f"规划器={state.tool_plan.generated_by if state.tool_plan else 'none'}"
            ),
            "parse_report": (
                f"Findings {len(state.parsed_report.findings) if state.parsed_report else 0} 字符；"
                f"Impression {len(state.parsed_report.impression) if state.parsed_report else 0} 字符"
            ),
            "run_text_classifier": f"18 类报告分类；阳性 {positive_report} 类",
            "run_report_graph": (
                f"生成 {len(state.report_graph.nodes)} 个实体、"
                f"{len(state.report_graph.edges)} 条关系；"
                f"后端={state.report_graph.backend}"
            ),
            "run_ct_classifier": (
                f"18 类 CT 分类；阳性 {positive_ct} 类；"
                f"缓存={'命中' if state.ct_cache_hit else '未命中'}"
            ),
            "run_organ_segmentation": f"获得 {len(state.anatomy_masks)} 个可用 mask",
            "run_lesion_grounding": f"生成 {len(state.region_findings)} 条区域级发现",
            "plan_rag_queries": f"生成 {len(state.rag_queries)} 条检索问题",
            "retrieve_medical_knowledge": (
                f"召回 {len(state.retrieved_docs)} 篇知识文档；后端={state.rag_backend}"
            ),
            "grade_retrieval": (
                "检索证据充分" if state.retrieval_sufficient else "检索证据不足，准备改写"
            ),
            "rewrite_query_if_needed": f"改写为 {len(state.rag_queries)} 条检索问题",
            "retrieve_similar_cases": (
                f"召回 {len(state.similar_cases)} 个患者级去重相似病例；"
                f"策略={state.similar_cases[0].retrieval_strategy if state.similar_cases else 'none'}"
            ),
            "extract_evidence": (
                f"为 {sum(bool(items) for items in state.evidence_by_label.values())} 类找到报告证据"
            ),
            "check_consistency": (
                f"融合 {len(state.fusion_predictions)} 类结果；当前警告 {len(state.consistency_warnings)} 条"
            ),
            "generate_json": "生成 18 类结构化医学结果",
            "validate_output": "Pydantic 结构与字段约束校验完成",
            "human_approval": (
                "触发人工审批" if state.approval.required else "无需强制人工审批"
            ),
            "generate_chinese_explanation": "基于结构化证据生成中文结论和公开分析说明",
        }
        return summaries.get(node_name, "节点执行完成")

    def _event_decision_audit(
        self, node_name: str, state: AgentState
    ) -> tuple[str, list[str], dict[str, object]]:
        """Build an auditable rationale from persisted inputs, outputs, and rules."""

        def status_counts(predictions) -> dict[str, int]:
            return {
                status: sum(item.status == status for item in predictions)
                for status in ("positive", "uncertain", "negative")
            }

        def candidate_text(predictions, limit: int = 8) -> str:
            candidates = sorted(
                (
                    item
                    for item in predictions
                    if item.status in {"positive", "uncertain"}
                ),
                key=lambda item: item.confidence,
                reverse=True,
            )[:limit]
            if not candidates:
                return "无阳性或不确定候选"
            return "、".join(
                f"{LABEL_ZH.get(item.name, item.name)}={STATUS_ZH[item.status]}"
                f"({item.confidence:.2f})"
                for item in candidates
            )

        has_report = bool(state.request.report_text.strip())
        has_ct = bool(state.request.ct_volume_path)
        summary = self._event_summary(node_name, state)

        if node_name == "parse_input":
            mode = "CT+报告" if has_ct and has_report else "仅CT" if has_ct else "仅报告"
            ct_identity = (
                f"CT文件={state.ct_input_name or '未知'}；"
                f"SHA-256={state.ct_input_sha256[:16] if state.ct_input_sha256 else '未计算'}；"
                f"大小={state.ct_input_size_bytes if state.ct_input_size_bytes is not None else '未知'}字节。"
                if has_ct
                else "未提供CT文件。"
            )
            return (
                f"输入被识别为{mode}，后续只允许调用与现有资料匹配的工具。",
                [
                    f"CT输入={'有' if has_ct else '无'}；报告输入={'有' if has_report else '无'}。",
                    ct_identity,
                    f"用户任务：{state.request.question[:240]}",
                ],
                {
                    "has_ct": has_ct,
                    "has_report": has_report,
                    "input_mode": mode,
                    "ct_input_name": state.ct_input_name,
                    "ct_input_sha256_prefix": (
                        state.ct_input_sha256[:16] if state.ct_input_sha256 else None
                    ),
                    "ct_input_size_bytes": state.ct_input_size_bytes,
                },
            )

        if node_name == "plan_tools":
            steps = state.tool_plan.steps if state.tool_plan else []
            selected = [step.tool for step in steps]
            omitted_reasons = []
            if not has_report:
                omitted_reasons.append("未提供报告，因此跳过报告解析、报告分类和RadGraph。")
            if not has_ct:
                omitted_reasons.append("未提供CT，因此跳过CT分类、分割和病灶定位。")
            basis = [
                f"规划器={state.tool_plan.generated_by if state.tool_plan else 'none'}；"
                f"选择工具：{'、'.join(selected)}。"
            ] + omitted_reasons
            return (
                f"从允许列表中选择{len(selected)}个工具，并由确定性策略补齐必需节点。",
                basis,
                {
                    "planner": state.tool_plan.generated_by if state.tool_plan else "none",
                    "selected_tools": len(selected),
                    "llm_fallback": bool(
                        state.tool_plan and state.tool_plan.generated_by != "llm"
                    ),
                },
            )

        if node_name == "parse_report":
            findings_len = len(state.parsed_report.findings) if state.parsed_report else 0
            impression_len = len(state.parsed_report.impression) if state.parsed_report else 0
            return (
                "报告已按Findings和Impression分段，供分类、RadGraph和证据抽取复用。",
                [f"Findings={findings_len}字符；Impression={impression_len}字符。"],
                {"findings_chars": findings_len, "impression_chars": impression_len},
            )

        if node_name == "run_text_classifier":
            counts = status_counts(state.report_predictions)
            return (
                "报告候选由否定感知的18类分类结果产生，尚未直接作为最终结论。",
                [f"报告候选：{candidate_text(state.report_predictions)}。"],
                counts,
            )

        if node_name == "run_report_graph":
            assertions = {
                assertion: sum(node.assertion == assertion for node in state.report_graph.nodes)
                for assertion in ("definitely_present", "definitely_absent", "uncertain")
            }
            return (
                "用实体断言和临床关系约束报告证据的极性与解剖位置。",
                [
                    f"后端={state.report_graph.backend}；降级={state.report_graph.degraded}。",
                    f"实体={len(state.report_graph.nodes)}；关系={len(state.report_graph.edges)}。",
                ],
                assertions,
            )

        if node_name == "run_ct_classifier":
            counts = status_counts(state.ct_predictions)
            return (
                "CT模型输出先作为候选，继续与报告、定位和检索证据融合。",
                [
                    f"CT候选：{candidate_text(state.ct_predictions)}。",
                    f"缓存={'命中' if state.ct_cache_hit else '未命中'}；"
                    f"质量门控={'触发' if state.ct_quality_degraded else '通过'}。",
                ],
                {**counts, "cache_hit": state.ct_cache_hit, "quality_degraded": state.ct_quality_degraded},
            )

        if node_name == "run_organ_segmentation":
            requested_regions = sorted(
                {
                    region
                    for prediction in state.report_predictions + state.ct_predictions
                    if prediction.status in {"positive", "uncertain"}
                    for region in LABEL_BY_ID[prediction.name].anatomy_regions
                }
            )
            available = len(state.anatomy_masks)
            decision = (
                "获得真实对齐mask，可供区域定位使用。"
                if available
                else "未获得真实对齐mask，后续不得把病例级预览冒充病灶定位。"
            )
            return (
                decision,
                [
                    "候选解剖区域：" + ("、".join(requested_regions) or "无"),
                    f"可用mask={available}；对齐验证通过="
                    f"{sum(item.alignment_verified for item in state.anatomy_masks)}。",
                ],
                {"requested_regions": len(requested_regions), "available_masks": available},
            )

        if node_name == "run_lesion_grounding":
            grounded = len(state.region_findings)
            return (
                "已输出区域级发现。" if grounded else "缺少可验证mask，因此没有生成伪定位结果。",
                [
                    f"输入mask={len(state.anatomy_masks)}；区域级发现={grounded}。",
                    "只有真实mask或明确弱定位证据才能写入区域级输出。",
                ],
                {"masks": len(state.anatomy_masks), "region_findings": grounded},
            )

        if node_name in {"plan_rag_queries", "rewrite_query_if_needed"}:
            source = "Qwen远程规划" if not state.llm_fallback_reasons else "Qwen规划或确定性降级"
            return (
                f"根据用户问题和阳性/不确定候选生成{len(state.rag_queries)}条检索查询。",
                [f"查询：{'；'.join(state.rag_queries[:10])}", f"生成来源：{source}。"],
                {"query_count": len(state.rag_queries), "llm_fallbacks": state.llm_fallbacks},
            )

        if node_name == "retrieve_medical_knowledge":
            top_docs = sorted(state.retrieved_docs, key=lambda item: item.score, reverse=True)[:5]
            return (
                f"混合检索召回{len(state.retrieved_docs)}篇文档，交给独立规则判断是否充分。",
                [
                    f"后端={state.rag_backend}。",
                    "Top结果：" + (
                        "；".join(f"{doc.title}({doc.score:.3f})" for doc in top_docs)
                        or "无"
                    ),
                ],
                {
                    "documents": len(state.retrieved_docs),
                    "top_score": round(top_docs[0].score, 4) if top_docs else 0.0,
                    "attempt": state.retrieval_attempts,
                },
            )

        if node_name == "grade_retrieval":
            expected = {
                item.name
                for item in state.report_predictions + state.ct_predictions
                if item.status in {"positive", "uncertain"} and item.confidence >= 0.3
            }
            relevant = [doc for doc in state.retrieved_docs if doc.score >= 0.08]
            covered = {
                str(doc.metadata.get("label"))
                for doc in relevant
                if doc.metadata.get("label")
            }
            required = 0 if not expected else 1 if len(expected) <= 2 else max(2, (len(expected) + 1) // 2)
            return (
                "检索证据达到充分条件，停止改写。"
                if state.retrieval_sufficient
                else "检索证据未达到充分条件，且仍有轮次时改写查询。",
                [
                    "相关文档阈值=0.08。",
                    f"期望覆盖={sorted(expected)}；实际覆盖={sorted(expected & covered)}；"
                    f"要求至少覆盖={required}。",
                ],
                {
                    "relevant_documents": len(relevant),
                    "expected_labels": len(expected),
                    "covered_labels": len(expected & covered),
                    "required_coverage": required,
                    "sufficient": state.retrieval_sufficient,
                },
            )

        if node_name == "retrieve_similar_cases":
            strategy = (
                state.similar_cases[0].retrieval_strategy
                if state.similar_cases
                else "none"
            )
            reason = (
                "按报告、CT预测条件和解剖区域的可用组合返回训练集病例。"
                if state.similar_cases
                else "索引中没有满足条件且排除当前病例后的候选。"
            )
            return (
                reason,
                [
                    f"查询病例={state.request.case_id}；返回数量={len(state.similar_cases)}。",
                    f"检索策略={strategy}；报告输入={'有' if has_report else '无'}；"
                    f"CT候选标签={sum(item.status in {'positive', 'uncertain'} and item.confidence >= 0.3 for item in state.ct_predictions)}。",
                    "当前患者及其其他重建版本始终排除，返回结果也按患者去重。",
                    "候选标签来自模型预测；CT-RATE弱标签只参与候选重排和结果解释，不作为当前病例真值。",
                ],
                {
                    "similar_cases": len(state.similar_cases),
                    "has_report_query": has_report,
                    "has_ct_condition_query": bool(state.ct_predictions),
                    "retrieval_strategy": strategy,
                    "unique_patients": len({item.patient_id for item in state.similar_cases}),
                },
            )

        if node_name == "extract_evidence":
            supported = sum(bool(items) for items in state.evidence_by_label.values())
            return (
                "没有报告输入，因此不生成报告证据。"
                if not has_report
                else f"为{supported}类保留了带极性和确定性的原文证据。",
                [
                    f"报告输入={'有' if has_report else '无'}；有证据标签={supported}。",
                    "RadGraph证据优先，并与规则证据去重。",
                ],
                {"labels_with_report_evidence": supported},
            )

        if node_name == "check_consistency":
            counts = status_counts(state.fusion_predictions)
            return (
                "按逐标签校准阈值融合CT、报告和证据；冲突项进入不确定或人工复核。",
                [
                    f"默认阈值：阳性={self.settings.positive_label_threshold:.2f}；"
                    f"不确定={self.settings.min_label_confidence:.2f}；"
                    f"强阴性={self.settings.strong_negative_threshold:.2f}。",
                    f"融合候选：{candidate_text(state.fusion_predictions)}。",
                    f"一致性警告={len(state.consistency_warnings)}。",
                ],
                {**counts, "warnings": len(state.consistency_warnings)},
            )

        if node_name == "generate_json":
            counts = status_counts(state.fusion_predictions)
            return (
                "把融合后的18类状态、来源分数和证据引用写入固定Schema。",
                [
                    f"最终状态统计：{counts}。",
                    f"区域级发现={len(state.region_findings)}；相似病例={len(state.similar_cases)}。",
                ],
                counts,
            )

        if node_name == "validate_output":
            return (
                "结构化结果通过Pydantic类型、范围和唯一性约束后才允许返回。",
                [
                    f"JSON校验={'通过' if state.final_response is not None else '未通过'}。",
                    f"输出警告={len(state.final_response.warnings) if state.final_response else 0}。",
                ],
                {"valid": state.final_response is not None},
            )

        if node_name == "human_approval":
            return (
                "命中至少一条审批规则，结果标记为待医生复核。"
                if state.approval.required
                else "未命中强制审批规则。",
                state.approval.reasons or ["无强制审批原因。"],
                {"required": state.approval.required, "reason_count": len(state.approval.reasons)},
            )

        if node_name == "generate_chinese_explanation":
            labels = state.final_response.labels if state.final_response else []
            positive = [item.name_zh for item in labels if item.status == "positive"]
            uncertain = [item.name_zh for item in labels if item.status == "uncertain"]
            reasoning = (
                state.final_response.model_reasoning if state.final_response else None
            )
            source = reasoning.generated_by if reasoning else "not_used"
            return (
                "Qwen基于融合后的标签和证据生成公开分析说明。"
                if source == "qwen"
                else "Qwen调用不可用，使用确定性证据摘要且明确标记为回退。",
                [
                    "阳性：" + ("、".join(positive) or "无"),
                    "不确定：" + ("、".join(uncertain) or "无"),
                    f"分析说明来源：{source}；步骤={len(reasoning.steps) if reasoning else 0}。",
                ],
                {
                    "positive": len(positive),
                    "uncertain": len(uncertain),
                    "reasoning_source": source,
                    "reasoning_steps": len(reasoning.steps) if reasoning else 0,
                },
            )

        return (
            f"节点按预定义工作流规则完成：{summary}",
            [f"工具输出：{summary}"],
            {},
        )

    @staticmethod
    def _normalize_rag_queries(values: object) -> list[str]:
        if not isinstance(values, list):
            return []
        queries: list[str] = []
        for value in values:
            if isinstance(value, dict):
                value = value.get("query", value.get("text", ""))
            query = str(value).strip()
            if query and query not in queries:
                queries.append(query)
        return queries

    @staticmethod
    def _node_tool_name(node_name: str) -> str:
        return {
            "parse_input": "parse_input",
            "plan_tools": "task_planner",
            "parse_report": "report_parser_tool",
            "run_text_classifier": "text_classifier_tool",
            "run_report_graph": "report_graph_tool",
            "run_ct_classifier": "ct_classifier_tool",
            "run_organ_segmentation": "organ_segmentation_tool",
            "run_lesion_grounding": "lesion_grounding_tool",
            "plan_rag_queries": "agentic_rag_query_planner",
            "retrieve_medical_knowledge": "medical_rag_tool",
            "grade_retrieval": "retrieval_grader",
            "rewrite_query_if_needed": "query_rewriter",
            "retrieve_similar_cases": "similar_case_retriever_tool",
            "extract_evidence": "evidence_extractor_tool",
            "check_consistency": "consistency_checker_tool",
            "generate_json": "structured_output_generator",
            "validate_output": "json_validator_tool",
            "human_approval": "human_approval_gate",
            "generate_chinese_explanation": "explanation_generator",
        }.get(node_name, node_name)

    async def _invoke_timed(self, name: str, node, data: dict) -> dict:
        return await self._timed_node(name, node)(data)

    async def _run_without_langgraph(self, initial: dict) -> AgentState:
        data = await self._invoke_timed("parse_input", self.parse_input, initial)
        data = await self._invoke_timed("plan_tools", self.plan_tools, data)
        state = AgentState.model_validate(data)
        if self._is_planned(state, "report_parser_tool"):
            data = await self._invoke_timed("parse_report", self.parse_report, data)
            data = await self._invoke_timed(
                "run_text_classifier", self.run_text_classifier, data
            )
            data = await self._invoke_timed(
                "run_report_graph", self.run_report_graph, data
            )
        state = AgentState.model_validate(data)
        if self._is_planned(state, "ct_classifier_tool"):
            data = await self._invoke_timed("run_ct_classifier", self.run_ct_classifier, data)
        state = AgentState.model_validate(data)
        if self._is_planned(state, "organ_segmentation_tool"):
            data = await self._invoke_timed(
                "run_organ_segmentation", self.run_organ_segmentation, data
            )
        state = AgentState.model_validate(data)
        if self._is_planned(state, "lesion_grounding_tool"):
            data = await self._invoke_timed(
                "run_lesion_grounding", self.run_lesion_grounding, data
            )
        state = AgentState.model_validate(data)
        if self._is_planned(state, "medical_rag_tool"):
            data = await self._invoke_timed("plan_rag_queries", self.plan_rag_queries, data)
            while True:
                data = await self._invoke_timed(
                    "retrieve_medical_knowledge", self.retrieve_medical_knowledge, data
                )
                data = await self._invoke_timed("grade_retrieval", self.grade_retrieval, data)
                state = AgentState.model_validate(data)
                if state.retrieval_sufficient or (
                    state.retrieval_attempts >= state.max_retrieval_attempts
                ):
                    break
                data = await self._invoke_timed(
                    "rewrite_query_if_needed", self.rewrite_query_if_needed, data
                )
        state = AgentState.model_validate(data)
        if self._is_planned(state, "similar_case_retriever_tool"):
            data = await self._invoke_timed(
                "retrieve_similar_cases", self.retrieve_similar_cases, data
            )
        for name, node in (
            ("extract_evidence", self.extract_evidence),
            ("check_consistency", self.check_consistency),
        ):
            data = await self._invoke_timed(name, node, data)
        for name, node in (
            ("generate_json", self.generate_json),
            ("validate_output", self.validate_output),
            ("human_approval", self.human_approval),
            ("generate_chinese_explanation", self.generate_chinese_explanation),
        ):
            data = await self._invoke_timed(name, node, data)
        return AgentState.model_validate(data)

    @staticmethod
    def route_after_input(data: dict) -> str:
        state = AgentState.model_validate(data)
        return "report" if state.request.report_text.strip() else "ct"

    @staticmethod
    def route_after_text(data: dict) -> str:
        state = AgentState.model_validate(data)
        if state.request.ct_volume_path and ChestCtAgent._is_planned(state, "ct_classifier_tool"):
            return "ct"
        return ChestCtAgent._route_after_analysis_state(state)

    @staticmethod
    def route_after_plan(data: dict) -> str:
        state = AgentState.model_validate(data)
        if state.request.report_text.strip() and ChestCtAgent._is_planned(
            state, "report_parser_tool"
        ):
            return "report"
        if state.request.ct_volume_path and ChestCtAgent._is_planned(state, "ct_classifier_tool"):
            return "ct"
        return ChestCtAgent._route_after_analysis_state(state)

    @staticmethod
    def route_after_ct(data: dict) -> str:
        state = AgentState.model_validate(data)
        if ChestCtAgent._is_planned(state, "organ_segmentation_tool"):
            return "organ"
        if ChestCtAgent._is_planned(state, "lesion_grounding_tool"):
            return "grounding"
        return ChestCtAgent._route_after_analysis_state(state)

    @staticmethod
    def route_after_organ_segmentation(data: dict) -> str:
        state = AgentState.model_validate(data)
        if ChestCtAgent._is_planned(state, "lesion_grounding_tool"):
            return "grounding"
        return ChestCtAgent._route_after_analysis_state(state)

    @staticmethod
    def route_after_analysis(data: dict) -> str:
        return ChestCtAgent._route_after_analysis_state(AgentState.model_validate(data))

    @staticmethod
    def _route_after_analysis_state(state: AgentState) -> str:
        if ChestCtAgent._is_planned(state, "medical_rag_tool"):
            return "rag"
        if ChestCtAgent._is_planned(state, "similar_case_retriever_tool"):
            return "similar"
        return "evidence"

    @staticmethod
    def _is_planned(state: AgentState, tool_name: str) -> bool:
        return bool(state.tool_plan and any(step.tool == tool_name for step in state.tool_plan.steps))

    @staticmethod
    def route_after_retrieval_grade(data: dict) -> str:
        state = AgentState.model_validate(data)
        if not state.retrieval_sufficient and state.retrieval_attempts < state.max_retrieval_attempts:
            return "rewrite"
        if ChestCtAgent._is_planned(state, "similar_case_retriever_tool"):
            return "similar"
        return "evidence"

    async def parse_input(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        if state.request.ct_volume_path:
            path = Path(state.request.ct_volume_path)
            source_name = state.request.ct_source_name or path.name
            state.ct_input_name = source_name.replace("\\", "/").rsplit("/", 1)[-1]
            if path.is_file():
                state.ct_input_size_bytes = path.stat().st_size
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                state.ct_input_sha256 = digest.hexdigest()
        state.tool_trace.append("parse_input")
        return state.model_dump(mode="python")

    async def plan_tools(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        state.tool_plan = await self.planner.plan(state.request)
        if self.settings.agent_dynamic_planning:
            state.llm_calls += 1
            state.llm_fallbacks += int(state.tool_plan.generated_by != "llm")
            if state.tool_plan.fallback_reason:
                state.llm_fallback_reasons.append(
                    f"task_planner:{state.tool_plan.fallback_reason}"
                )
        state.tool_trace.append("task_planner")
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

    async def run_report_graph(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        state.report_graph = await self.report_graph.extract(state.request.report_text)
        if state.report_graph.warning:
            state.consistency_warnings.append(state.report_graph.warning)
        state.tool_trace.append("report_graph_tool")
        return state.model_dump(mode="python")

    async def run_ct_classifier(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        rendered = self.ct_preprocess.render_preview_slices(
            state.request.case_id, state.request.ct_volume_path
        )
        preview_images = state.request.ct_preview_images or rendered
        state.ct_preview_images = preview_images
        state.ct_predictions, ct_warnings, state.ct_cache_hit = self.ct_classifier.predict(
            state.request.ct_volume_path, preview_images
        )
        quality_warning = next(
            (warning for warning in ct_warnings if warning.startswith("CT质量门控触发")),
            None,
        )
        state.ct_quality_degraded = quality_warning is not None
        state.ct_quality_reason = quality_warning
        state.consistency_warnings.extend(ct_warnings)
        state.image_evidence_by_label = build_visual_evidence(state.ct_predictions, preview_images)
        state.tool_trace.append("ct_preprocess_tool")
        state.tool_trace.append("ct_classifier_tool")
        state.tool_trace.append("visual_evidence_tool")
        return state.model_dump(mode="python")

    async def run_organ_segmentation(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        requested_regions = {
            region
            for prediction in state.report_predictions + state.ct_predictions
            if prediction.status in {"positive", "uncertain"}
            for region in LABEL_BY_ID[prediction.name].anatomy_regions
        }
        requested_regions.update(
            alias
            for prediction in state.report_predictions + state.ct_predictions
            if prediction.status in {"positive", "uncertain"}
            for alias in LESION_MASK_ALIASES.get(prediction.name, set())
        )
        state.anatomy_masks, warnings = self.organ_segmentation.segment(
            state.request.case_id,
            state.request.ct_volume_path,
            requested_regions=requested_regions or None,
        )
        state.consistency_warnings.extend(warnings)
        state.tool_trace.append("organ_segmentation_tool")
        return state.model_dump(mode="python")

    async def run_lesion_grounding(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        source_predictions = state.ct_predictions or state.report_predictions
        state.region_findings, grounded_evidence = ground_findings(
            source_predictions, state.anatomy_masks
        )
        state.image_evidence_by_label.update(grounded_evidence)
        if not state.region_findings:
            state.consistency_warnings.append(
                "没有可用的病灶级标注；系统未将病例级预览冒充病灶定位。"
            )
        state.tool_trace.append("lesion_grounding_tool")
        return state.model_dump(mode="python")

    async def plan_rag_queries(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        positive_labels = [
            item.name
            for item in state.report_predictions + state.ct_predictions
            if item.status in {"positive", "uncertain"} and item.confidence >= 0.3
        ]
        fallback = {"queries": sorted(set(positive_labels + [state.request.question]))}
        call = await self.qwen.chat_json(
            system="You plan concise retrieval queries for a radiology Agentic RAG system.",
            user=(
                "Create retrieval queries for medical definitions, imaging findings, and report terms. "
                f"Question: {state.request.question}\nReport: {state.request.report_text[:2000]}\n"
                f"Candidate labels: {positive_labels}"
            ),
            fallback=fallback,
        )
        state.llm_calls += 1
        state.llm_fallbacks += int(not call.used_remote)
        if call.fallback_reason:
            state.llm_fallback_reasons.append(
                f"agentic_rag_query_planner:{call.fallback_reason}"
            )
        result = call.value
        queries = result.get("queries", fallback["queries"])
        state.rag_queries = self._normalize_rag_queries(queries) or fallback["queries"]
        state.rag_query_history.append(state.rag_queries.copy())
        state.tool_trace.append("agentic_rag_query_planner")
        return state.model_dump(mode="python")

    async def retrieve_medical_knowledge(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        state.retrieved_docs, state.rag_backend = await self.medical_rag.retrieve(
            state.rag_queries,
            top_k=5,
        )
        state.retrieval_attempts += 1
        state.retrieval_history.append(
            RetrievalAttemptTrace(
                attempt=state.retrieval_attempts,
                queries=state.rag_queries.copy(),
                backend=state.rag_backend,
                documents=state.retrieved_docs.copy(),
            )
        )
        state.tool_trace.append("medical_rag_tool")
        return state.model_dump(mode="python")

    async def retrieve_similar_cases(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        label_scores: dict[str, float] = {}
        for item in state.report_predictions + state.ct_predictions:
            if item.status in {"positive", "uncertain"} and item.confidence >= 0.3:
                label_scores[item.name] = max(
                    label_scores.get(item.name, 0.0), item.confidence
                )
        anatomy_regions = sorted(
            {
                finding.region
                for finding in state.region_findings
                if finding.status in {"positive", "uncertain"}
            }
        )
        top_k = state.request.top_k_similar or self.settings.top_k_similar
        state.similar_cases = self.similar_cases.retrieve(
            state.request.report_text,
            sorted(label_scores),
            top_k=top_k,
            query_case_id=state.request.case_id,
            label_scores=label_scores,
            anatomy_regions=anatomy_regions,
        )
        state.tool_trace.append("similar_case_retriever_tool")
        return state.model_dump(mode="python")

    async def grade_retrieval(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        expected_labels = {
            item.name
            for item in state.report_predictions + state.ct_predictions
            if item.status in {"positive", "uncertain"} and item.confidence >= 0.3
        }
        state.retrieval_sufficient = grade_retrieval(
            state.retrieved_docs,
            expected_labels=expected_labels,
        )
        if state.retrieval_history:
            state.retrieval_history[-1] = state.retrieval_history[-1].model_copy(
                update={"sufficient": state.retrieval_sufficient}
            )
        state.tool_trace.append("retrieval_grader")
        return state.model_dump(mode="python")

    async def rewrite_query_if_needed(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        positive_labels = [
            item.name
            for item in state.report_predictions + state.ct_predictions
            if item.status in {"positive", "uncertain"}
        ]
        expanded_terms = []
        for label in positive_labels:
            entry = LABEL_KNOWLEDGE.get(label, {})
            expanded_terms.extend(entry.get("terms", [])[:3])
        fallback = {
            "queries": sorted(
                set(state.rag_queries + [label.replace("_", " ") for label in positive_labels] + expanded_terms)
            )
        }
        call = await self.qwen.chat_json(
            system=(
                "Rewrite insufficient radiology retrieval queries. Return JSON with a queries list. "
                "Use only the supplied labels and report concepts."
            ),
            user=(
                f"Previous queries: {state.rag_queries}\n"
                f"Candidate labels: {positive_labels}\n"
                f"Retrieved titles: {[doc.title for doc in state.retrieved_docs]}"
            ),
            fallback=fallback,
        )
        state.llm_calls += 1
        state.llm_fallbacks += int(not call.used_remote)
        if call.fallback_reason:
            state.llm_fallback_reasons.append(f"query_rewriter:{call.fallback_reason}")
        state.rag_queries = (
            self._normalize_rag_queries(call.value.get("queries")) or fallback["queries"]
        )
        state.rag_query_history.append(state.rag_queries.copy())
        state.tool_trace.append("query_rewriter")
        return state.model_dump(mode="python")

    async def extract_evidence(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        rule_evidence = extract_evidence(state.request.report_text, LABEL_IDS)
        graph_evidence = report_graph_to_evidence(state.report_graph)
        merged: dict[str, list] = {}
        for label in LABEL_IDS:
            items = graph_evidence.get(label, []) + rule_evidence.get(label, [])
            seen: set[tuple[str, str, str]] = set()
            merged[label] = []
            for item in items:
                key = (item.sentence, item.polarity, item.matched_term.lower())
                if key in seen:
                    continue
                seen.add(key)
                merged[label].append(item)
        state.evidence_by_label = merged
        state.tool_trace.append("evidence_extractor_tool")
        return state.model_dump(mode="python")

    async def check_consistency(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        state.fusion_predictions, warnings = fuse_predictions(
            state.report_predictions,
            state.ct_predictions,
            positive_threshold=self.settings.positive_label_threshold,
            uncertain_threshold=self.settings.min_label_confidence,
            strong_negative_threshold=self.settings.strong_negative_threshold,
            evidence_by_label=state.evidence_by_label,
            calibration=self.fusion_calibration,
        )
        state.fusion_predictions, credibility_warnings = apply_credibility_gate(
            state.fusion_predictions,
            state.ct_predictions,
            state.evidence_by_label,
            state.ct_quality_degraded,
        )
        state.consistency_warnings.extend(warnings)
        state.consistency_warnings.extend(credibility_warnings)
        state.tool_trace.append("consistency_checker_tool")
        return state.model_dump(mode="python")

    async def generate_json(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        docs_by_label: dict[str, list[str]] = {}
        for doc in state.retrieved_docs:
            label = str(doc.metadata.get("label", ""))
            if label:
                docs_by_label.setdefault(label, []).append(doc.doc_id)
        label_outputs: list[LabelOutput] = []
        for prediction in state.fusion_predictions:
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
                    rag_support=prediction.name in docs_by_label,
                    rag_sources=docs_by_label.get(prediction.name, []),
                    need_human_review=True,
                )
            )

        fusion_by_label = {item.name: item for item in state.fusion_predictions}
        region_findings = []
        for finding in state.region_findings:
            prediction = fusion_by_label.get(finding.label)
            if prediction is None or prediction.status == "negative":
                continue
            region_findings.append(
                finding.model_copy(
                    update={
                        "status": prediction.status,
                        "confidence": prediction.confidence,
                    }
                )
            )

        has_report = bool(state.request.report_text.strip())
        has_ct = bool(state.request.ct_volume_path)
        if has_report and has_ct:
            input_mode = "report_and_ct"
        elif has_ct:
            input_mode = "ct_only"
        else:
            input_mode = "report_only"

        response = AnalyzeResponse(
            case_id=state.request.case_id,
            labels=label_outputs,
            ct_preview_images=state.ct_preview_images,
            similar_cases=state.similar_cases,
            explanation_zh="",
            disclaimer=self.settings.disclaimer,
            tool_trace=state.tool_trace.copy(),
            warnings=state.consistency_warnings.copy(),
            execution=ExecutionMetadata(
                input_mode=input_mode,
                retrieval_attempts=state.retrieval_attempts,
                retrieval_sufficient=state.retrieval_sufficient,
                rag_backend=state.rag_backend,
                ct_cache_hit=state.ct_cache_hit,
                ct_model_variant=(
                    self.settings.ctclip_variant
                    if state.request.ct_volume_path
                    else "not_used"
                ),
                ct_input_name=state.ct_input_name,
                ct_input_size_bytes=state.ct_input_size_bytes,
                ct_input_sha256=state.ct_input_sha256,
                ct_quality_degraded=state.ct_quality_degraded,
                ct_quality_reason=state.ct_quality_reason,
                llm_calls=state.llm_calls,
                llm_fallbacks=state.llm_fallbacks,
                llm_fallback_reasons=state.llm_fallback_reasons.copy(),
            ),
            region_findings=region_findings,
            anatomy_masks=state.anatomy_masks,
            approval=state.approval,
            report_graph=state.report_graph,
        )
        state.draft_response = response
        state.tool_trace.append("structured_output_generator")
        return state.model_dump(mode="python")

    async def validate_output(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        if state.draft_response is None:
            raise RuntimeError("Missing draft response before validation.")
        validated, warnings = validate_response(state.draft_response)
        state.tool_trace.append("json_validator_tool")
        validated.warnings.extend(warnings)
        validated.tool_trace = state.tool_trace.copy()
        state.final_response = validated
        return state.model_dump(mode="python")

    async def human_approval(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        if state.final_response is None:
            raise RuntimeError("Missing final response before human approval.")
        reasons: list[str] = []
        if state.request.require_human_approval:
            reasons.append("请求方要求人工审批。")
        if state.final_response.warnings:
            reasons.append("存在工具降级或多模态一致性警告。")
        if any(item.status == "uncertain" for item in state.final_response.labels):
            reasons.append("至少一个标签处于不确定状态。")
        if state.request.ct_volume_path:
            unsupported = [
                item.name_zh
                for item in state.final_response.labels
                if item.status == "positive"
                and item.evidence_from_image.grounding_type != "lesion_mask"
            ]
            if unsupported:
                reasons.append(
                    "阳性结论缺少病灶级mask：" + "、".join(unsupported[:6])
                )
        if state.ct_quality_degraded:
            reasons.append("CT 分类输出触发质量门控，不能作为阳性结论。")
        required = bool(reasons)
        state.approval = HumanApproval(
            required=required,
            status="pending" if required else "not_required",
            reasons=reasons,
        )
        state.final_response.approval = state.approval
        for label in state.final_response.labels:
            label.need_human_review = required or label.status != "negative"
        state.tool_trace.append("human_approval_gate")
        state.final_response.tool_trace = state.tool_trace.copy()
        return state.model_dump(mode="python")

    @staticmethod
    def _build_reasoning_fallback(
        state: AgentState, explanation_zh: str
    ) -> ModelReasoningReport:
        if state.final_response is None:
            return ModelReasoningReport(generated_by="deterministic_fallback")

        stage_names = {
            "plan_tools": "工具规划",
            "run_text_classifier": "报告候选识别",
            "run_report_graph": "报告实体关系抽取",
            "run_ct_classifier": "CT候选识别",
            "run_organ_segmentation": "器官分割",
            "run_lesion_grounding": "病灶定位",
            "grade_retrieval": "检索证据评估",
            "check_consistency": "多模态证据融合",
            "human_approval": "人工审批判断",
        }
        focus_nodes = set(stage_names)
        steps: list[ModelReasoningStep] = []
        for event in state.execution_events:
            if event.node not in focus_nodes or not event.decision_summary:
                continue
            unresolved = event.key_metrics.get("unresolved", 0)
            uncertainty = (
                f"该步骤仍有{unresolved}项未解决。" if unresolved else ""
            )
            steps.append(
                ModelReasoningStep(
                    order=len(steps) + 1,
                    stage=stage_names[event.node],
                    decision=event.decision_summary,
                    evidence=event.decision_basis[:4],
                    uncertainty=uncertainty,
                )
            )

        if not steps:
            selected = [
                item
                for item in state.final_response.labels
                if item.status in {"positive", "uncertain"}
            ]
            steps.append(
                ModelReasoningStep(
                    order=1,
                    stage="结构化标签汇总",
                    decision="根据门控后的标签状态生成中文结果。",
                    evidence=[
                        "；".join(
                            f"{item.name_zh}={item.status_zh}({item.confidence:.2f})"
                            for item in selected
                        )
                        or "没有阳性或不确定标签。"
                    ],
                    uncertainty="未运行完整工作流时，只能提供标签级摘要。",
                )
            )

        positive = [
            item.name_zh
            for item in state.final_response.labels
            if item.status == "positive"
        ]
        uncertain = [
            item.name_zh
            for item in state.final_response.labels
            if item.status == "uncertain"
        ]
        limitations = list(
            dict.fromkeys(
                state.final_response.warnings + state.final_response.approval.reasons
            )
        )[:10]
        if not limitations:
            limitations.append("结果仍需结合原始CT和正式影像报告进行人工复核。")
        return ModelReasoningReport(
            generated_by="deterministic_fallback",
            structured_steps_by="audit_trace",
            summary_zh=(
                f"最终门控保留阳性{len(positive)}项、不确定{len(uncertain)}项；"
                f"结论摘要为：{explanation_zh[:300]}"
            ),
            steps=steps[:12],
            limitations=limitations,
        )

    async def generate_chinese_explanation(self, data: dict) -> dict:
        state = AgentState.model_validate(data)
        if state.final_response is None:
            raise RuntimeError("Missing final response before explanation.")
        positive_labels = [
            item for item in state.final_response.labels if item.status == "positive"
        ]
        uncertain_labels = sorted(
            (item for item in state.final_response.labels if item.status == "uncertain"),
            key=lambda item: item.confidence,
            reverse=True,
        )[:5]
        selected_labels = positive_labels + uncertain_labels
        labels = [
            {"name": item.name, "status": item.status, "confidence": item.confidence}
            for item in selected_labels
        ]
        explanation_parts: list[str] = []
        if positive_labels:
            if state.final_response.execution.input_mode == "ct_only":
                explanation_parts.append(
                    "主要影像发现："
                    + "；".join(
                        f"模型提示{LABEL_ZH.get(item.name, item.name)}"
                        f"（可信分数 {item.confidence:.2f}）"
                        for item in positive_labels
                    )
                    + "。"
                )
            else:
                explanation_parts.append(
                    "主要结论："
                    + "；".join(
                        f"{LABEL_ZH.get(item.name, item.name)}为阳性"
                        f"（融合分数 {item.confidence:.2f}）"
                        for item in positive_labels
                    )
                    + "。"
                )
        else:
            if state.final_response.execution.input_mode == "ct_only":
                explanation_parts.append("本次未检出达到主要发现阈值的异常。")
            else:
                explanation_parts.append("当前没有达到阳性阈值的主要结论。")
        if uncertain_labels:
            explanation_parts.append(
                "建议重点复核："
                + "；".join(
                f"{LABEL_ZH.get(item.name, item.name)}为{STATUS_ZH[item.status]}"
                    f"（模型分数 {item.confidence:.2f}）"
                    for item in uncertain_labels
                )
                + "。"
            )
        if state.final_response.execution.input_mode == "ct_only":
            negative_count = sum(
                item.status == "negative" for item in state.final_response.labels
            )
            if negative_count:
                explanation_parts.append(f"其余{negative_count}类异常未达到候选阈值。")
            explanation_parts.append(
                "本结果为AI影像分析结论；未提供影像报告时，"
                "最终结论以影像科医生复核为准。"
            )
        else:
            explanation_parts.append("以上结果必须结合原始 CT 和报告进行人工复核。")
        fallback = "".join(explanation_parts)
        if state.ct_quality_degraded and not state.request.report_text.strip():
            fallback = (
                "本次 CT-CLIP 输出触发质量门控：模型同时将过多异常类别判为阳性，"
                "该输出被视为退化结果。系统已取消全部 CT 阳性结论，当前不能据此判断"
                "患者同时存在这些疾病。请补充正式影像报告，或由影像科医生复核原始 CT。"
            )
        selected_names = {item["name"] for item in labels}
        selected_evidence = {
            label: [item.model_dump(mode="json") for item in evidence]
            for label, evidence in state.evidence_by_label.items()
            if label in selected_names
        }
        fallback_reasoning = self._build_reasoning_fallback(state, fallback)
        reasoning_context = {
            "task": state.request.question,
            "input_mode": state.final_response.execution.input_mode,
            "selected_tools": [
                step.tool for step in state.tool_plan.steps
            ]
            if state.tool_plan
            else [],
            "labels": [
                {
                    "name": item.name,
                    "name_zh": item.name_zh,
                    "status": item.status,
                    "confidence": item.confidence,
                    "source_scores": item.source_scores,
                    "report_evidence": [
                        evidence.model_dump(mode="json")
                        for evidence in item.evidence_from_report
                    ],
                    "image_evidence": item.evidence_from_image.model_dump(mode="json"),
                    "rag_support": item.rag_support,
                    "rag_sources": item.rag_sources,
                }
                for item in selected_labels
            ],
            "retrieval": [
                {
                    "title": document.title,
                    "score": document.score,
                    "label": document.metadata.get("label"),
                }
                for document in state.retrieved_docs[:8]
            ],
            "evidence_by_label": selected_evidence,
            "retrieval_sufficient": state.retrieval_sufficient,
            "warnings": state.final_response.warnings[:12],
            "approval_reasons": state.final_response.approval.reasons,
            "workflow_observations": [
                {
                    "stage": event.node,
                    "tool_output": event.summary,
                    "facts": event.decision_basis,
                    "metrics": event.key_metrics,
                }
                for event in state.execution_events
                if event.node
                in {
                    "plan_tools",
                    "run_text_classifier",
                    "run_report_graph",
                    "run_ct_classifier",
                    "run_organ_segmentation",
                    "run_lesion_grounding",
                    "grade_retrieval",
                    "check_consistency",
                    "human_approval",
                }
            ],
            "disclaimer": state.final_response.disclaimer,
        }
        fallback_analysis = (
            fallback_reasoning.summary_zh
            + "\n\n结果形成过程：\n"
            + "\n".join(
                f"{step.order}. {step.stage}：{step.decision} "
                f"依据：{'；'.join(step.evidence)}"
                for step in fallback_reasoning.steps
            )
            + "\n\n仍需注意：\n"
            + "\n".join(f"- {item}" for item in fallback_reasoning.limitations)
        )
        call = await self.qwen.chat_text(
            system=(
                "You are the explanation component of a chest CT coursework agent. "
                "Write a concise public evidence-based analysis in Chinese explaining how the "
                "supplied tool outputs led to the final result. This is an outward explanation, "
                "not private chain-of-thought. Use the headings '结论', '结果形成过程', and "
                "'不确定性与限制'. Under 结果形成过程, write 5-8 numbered steps covering "
                "tool selection, label scores, report/image evidence, retrieval when used, "
                "multimodal fusion and the review trigger. Use only supplied facts. "
                "Never invent observations, masks, scores, thresholds, or citations. Mention "
                "only supplied labels. Do not treat missing evidence as support. Do not output "
                "JSON or a code block."
            ),
            user=json.dumps(reasoning_context, ensure_ascii=False),
            fallback=fallback_analysis,
            max_tokens=max(self.settings.llm_text_max_tokens, 1200),
        )
        state.llm_calls += 1
        state.llm_fallbacks += int(not call.used_remote)
        if call.fallback_reason:
            state.llm_fallback_reasons.append(f"explanation_generator:{call.fallback_reason}")
        public_analysis = call.value.strip() or fallback_analysis
        reasoning = fallback_reasoning.model_copy(
            update={
                "generated_by": (
                    "qwen" if call.used_remote else "deterministic_fallback"
                ),
                "structured_steps_by": "audit_trace",
                "summary_zh": public_analysis,
                "raw_response_zh": public_analysis if call.used_remote else "",
            }
        )
        state.final_response.explanation_zh = fallback
        state.final_response.model_reasoning = reasoning
        state.final_response.execution.llm_calls = state.llm_calls
        state.final_response.execution.llm_fallbacks = state.llm_fallbacks
        state.final_response.execution.llm_fallback_reasons = (
            state.llm_fallback_reasons.copy()
        )
        state.tool_trace.append("explanation_generator")
        state.final_response.tool_trace = state.tool_trace.copy()
        return state.model_dump(mode="python")
