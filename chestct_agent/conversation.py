from collections.abc import Awaitable, Callable
import csv
import inspect
import json
import time
from typing import Any

from chestct_agent.llm import QwenClient
from chestct_agent.labels import LABEL_SPECS
from chestct_agent.memory import AgentMemory
from chestct_agent.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    ExecutionEvent,
    RetrievedDocument,
)
from chestct_agent.tools.rag import MedicalRagTool


ChatEventCallback = Callable[[ExecutionEvent], Awaitable[None]]


class CaseConversationAgent:
    """Case-bound follow-up agent with persistent dialogue and optional RAG."""

    def __init__(
        self,
        qwen: QwenClient,
        memory: AgentMemory,
        medical_rag: MedicalRagTool,
    ) -> None:
        self.qwen = qwen
        self.memory = memory
        self.medical_rag = medical_rag

    async def answer(
        self,
        request: ChatRequest,
        event_callback: ChatEventCallback | None = None,
    ) -> ChatResponse:
        events: list[ExecutionEvent] = []

        context = await self._step(
            events,
            event_callback,
            node="load_case_context",
            tool="conversation_memory_tool",
            start_summary="正在读取本病例分析结果和历史对话",
            operation=lambda: self.memory.get_case_context(
                request.session_id, request.case_id
            ),
            finish_summary=lambda value: (
                "已载入病例上下文"
                if value is not None
                else "未找到该会话对应的病例上下文"
            ),
        )
        if context is None:
            raise LookupError("请先在当前会话中完成一次病例分析。")
        analyze_request, analyze_response = context
        history = self.memory.get_messages(request.session_id, request.case_id, limit=12)

        fallback_plan = self._fallback_plan(request.message)
        plan_call = await self._step(
            events,
            event_callback,
            node="plan_followup",
            tool="followup_planner",
            start_summary="正在判断问题需要病例证据、医学检索还是相似病例",
            operation=lambda: self.qwen.chat_json(
                system=(
                    "You plan tools for a chest CT case follow-up agent. Return JSON only with "
                    "intent, use_rag, use_similar_cases. intent must be one of case_summary, "
                    "evidence, medical_knowledge, similar_cases, result_interpretation."
                ),
                user=json.dumps(
                    {
                        "question": request.message,
                        "case_id": request.case_id,
                        "available_positive_labels": [
                            item.name_zh
                            for item in analyze_response.labels
                            if item.status == "positive"
                        ],
                        "available_uncertain_labels": [
                            item.name_zh
                            for item in analyze_response.labels
                            if item.status == "uncertain"
                        ],
                    },
                    ensure_ascii=False,
                ),
                fallback=fallback_plan,
            ),
            finish_summary=lambda call: (
                "Qwen 已完成追问路由"
                if call.used_remote
                else "使用规则完成追问路由"
            ),
        )
        plan = self._normalize_plan(plan_call.value, fallback_plan)
        tools_used = ["conversation_memory_tool", "followup_planner"]

        documents: list[RetrievedDocument] = []
        rag_backend = "not_used"
        if plan["use_rag"]:
            documents, rag_backend = await self._step(
                events,
                event_callback,
                node="retrieve_followup_knowledge",
                tool="medical_rag_tool",
                start_summary="正在检索与本病例和当前问题相关的医学知识",
                operation=lambda: self.medical_rag.retrieve(
                    self._rag_queries(request.message, analyze_response), top_k=5
                ),
                finish_summary=lambda value: (
                    f"召回 {len(value[0])} 篇知识文档；后端={value[1]}"
                ),
            )
            tools_used.append("medical_rag_tool")
        if plan["use_similar_cases"]:
            tools_used.append("similar_case_context_tool")

        reference_evaluation: dict[str, Any] = {}
        if plan["intent"] == "result_interpretation":
            reference_evaluation = await self._step(
                events,
                event_callback,
                node="load_dataset_reference",
                tool="dataset_reference_tool",
                start_summary="正在检查该开发样例是否有可用的CT-RATE参考标签",
                operation=lambda: self._dataset_reference(analyze_response),
                finish_summary=lambda value: (
                    "已完成当前结果与CT-RATE弱标签对照"
                    if value
                    else "当前病例没有开发集参考标签，不能计算单例命中情况"
                ),
            )
            if reference_evaluation:
                tools_used.append("dataset_reference_tool")

        prompt = self._answer_prompt(
            request,
            analyze_request,
            analyze_response,
            history,
            plan,
            documents,
            rag_backend,
            reference_evaluation,
        )
        fallback_answer = self._fallback_answer(
            request.message,
            analyze_response,
            plan["intent"],
            documents,
            reference_evaluation,
        )
        answer_call = await self._step(
            events,
            event_callback,
            node="generate_followup_answer",
            tool="qwen_response_generator",
            start_summary="正在结合病例结果、历史对话和检索证据生成回答",
            operation=lambda: self.qwen.chat_text(
                system=(
                    "你是胸部CT病例多轮问答Agent。只根据提供的病例结果、报告证据、"
                    "历史对话和检索文档回答。直接给出中文答案，明确区分主要发现与待复核候选；"
                    "不得把不确定项改写成确诊，不得编造切片、大小、治疗或预后信息。"
                    "状态由每个标签各自的校准阈值决定，禁止跨标签比较原始分数，"
                    "也禁止把缺少定位说成状态降级原因。开发集参考是报告派生弱标签，不是金标准。"
                    "若存在development_reference_evaluation，先明确回答完全正确、部分正确或不正确。"
                    "planned_intent为evidence时只回答本病例证据和状态机制，不扩展通用病因或治疗。"
                    "不要每句话重复免责声明，必要的复核边界只在结尾简短说明；回答控制在500字以内。"
                ),
                user=prompt,
                fallback=fallback_answer,
            ),
            finish_summary=lambda call: (
                "Qwen 已生成基于当前病例的回答"
                if call.used_remote
                else "远端模型不可用，已生成规则化回答"
            ),
        )
        tools_used.append("qwen_response_generator")

        await self._step(
            events,
            event_callback,
            node="save_conversation",
            tool="conversation_memory_tool",
            start_summary="正在保存本轮问题和回答",
            operation=lambda: self._save_turn(request, answer_call.value),
            finish_summary=lambda _: "本轮对话已写入会话记忆",
        )
        history = self.memory.get_messages(request.session_id, request.case_id, limit=20)
        return ChatResponse(
            session_id=request.session_id,
            case_id=request.case_id,
            answer_zh=answer_call.value,
            intent=plan["intent"],
            tools_used=tools_used,
            execution_events=events,
            retrieved_documents=documents,
            reference_evaluation=reference_evaluation,
            used_remote_model=plan_call.used_remote or answer_call.used_remote,
            history=history,
        )

    async def _step(
        self,
        events: list[ExecutionEvent],
        callback: ChatEventCallback | None,
        *,
        node: str,
        tool: str,
        start_summary: str,
        operation,
        finish_summary,
    ):
        sequence = len(events) + 1
        await self._publish(
            callback,
            ExecutionEvent(
                sequence=sequence,
                node=node,
                tool=tool,
                status="running",
                summary=start_summary,
            ),
        )
        started = time.perf_counter()
        value = operation()
        if inspect.isawaitable(value):
            value = await value
        event = ExecutionEvent(
            sequence=sequence,
            node=node,
            tool=tool,
            status="success",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            summary=finish_summary(value),
        )
        events.append(event)
        await self._publish(callback, event)
        return value

    @staticmethod
    async def _publish(
        callback: ChatEventCallback | None, event: ExecutionEvent
    ) -> None:
        if callback is None:
            return
        try:
            await callback(event)
        except Exception:
            pass

    def _save_turn(self, request: ChatRequest, answer: str) -> None:
        self.memory.append_message(
            request.session_id, request.case_id, "user", request.message
        )
        self.memory.append_message(
            request.session_id, request.case_id, "assistant", answer
        )

    @staticmethod
    def _fallback_plan(message: str) -> dict[str, Any]:
        text = message.lower()
        similar = any(term in text for term in ("相似", "类似", "病例", "对比"))
        evidence = any(
            term in text for term in ("证据", "依据", "报告", "切片", "图像", "为什么提示")
        )
        knowledge = any(
            term in text
            for term in (
                "是什么",
                "什么意思",
                "严重",
                "风险",
                "原因",
                "怎么办",
                "处理",
                "鉴别",
                "进展",
            )
        )
        if similar:
            intent = "similar_cases"
        elif evidence:
            intent = "evidence"
        elif knowledge:
            intent = "medical_knowledge"
        elif any(term in text for term in ("对不对", "准确", "可信", "结果")):
            intent = "result_interpretation"
        else:
            intent = "case_summary"
        return {
            "intent": intent,
            "use_rag": knowledge,
            "use_similar_cases": similar,
        }

    @staticmethod
    def _normalize_plan(
        value: dict[str, Any], fallback: dict[str, Any]
    ) -> dict[str, Any]:
        allowed = {
            "case_summary",
            "evidence",
            "medical_knowledge",
            "similar_cases",
            "result_interpretation",
        }
        intent = str(value.get("intent", fallback["intent"]))
        if intent not in allowed:
            intent = str(fallback["intent"])
        if fallback["intent"] != "case_summary":
            intent = str(fallback["intent"])
        return {
            "intent": intent,
            "use_rag": bool(value.get("use_rag", False) or fallback["use_rag"]),
            "use_similar_cases": bool(
                value.get("use_similar_cases", False) or fallback["use_similar_cases"]
            ),
        }

    @staticmethod
    def _rag_queries(message: str, response: AnalyzeResponse) -> list[str]:
        labels = [
            item.name_zh
            for item in response.labels
            if item.status in {"positive", "uncertain"}
        ][:6]
        return [message, "胸部CT " + " ".join(labels)]

    @staticmethod
    def _answer_prompt(
        chat_request: ChatRequest,
        analyze_request: AnalyzeRequest,
        response: AnalyzeResponse,
        history,
        plan: dict[str, Any],
        documents: list[RetrievedDocument],
        rag_backend: str,
        reference_evaluation: dict[str, Any],
    ) -> str:
        case_context = {
            "case_id": response.case_id,
            "original_report": analyze_request.report_text[:8000],
            "original_summary": response.explanation_zh,
            "labels": [
                {
                    "name": item.name_zh,
                    "status": item.status_zh,
                    "confidence": item.confidence,
                    "report_evidence": [
                        evidence.sentence for evidence in item.evidence_from_report
                    ],
                    "slice_range": item.evidence_from_image.slice_range,
                    "localized": item.evidence_from_image.localized,
                }
                for item in response.labels
                if item.status != "negative"
            ],
            "region_findings": [item.model_dump() for item in response.region_findings],
            "similar_cases": [item.model_dump() for item in response.similar_cases[:5]],
            "warnings": response.warnings,
            "classification_policy": (
                "positive/uncertain/negative由每个异常独立校准的选择性阈值决定；"
                "不同异常的confidence不能横向比较；slice定位是独立证据，不负责状态升降级。"
            ),
        }
        payload = {
            "current_question": chat_request.message,
            "planned_intent": plan["intent"],
            "conversation_history": [item.model_dump() for item in history],
            "case_context": case_context,
            "rag_backend": rag_backend,
            "retrieved_knowledge": [
                {
                    "title": item.title,
                    "text": item.text,
                    "score": item.score,
                    "source": item.metadata.get("source", ""),
                }
                for item in documents
            ],
            "development_reference_evaluation": reference_evaluation,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _fallback_answer(
        message: str,
        response: AnalyzeResponse,
        intent: str,
        documents: list[RetrievedDocument],
        reference_evaluation: dict[str, Any],
    ) -> str:
        positive = [item.name_zh for item in response.labels if item.status == "positive"]
        uncertain = [item.name_zh for item in response.labels if item.status == "uncertain"]
        if intent == "evidence":
            evidence = [
                evidence.sentence
                for item in response.labels
                for evidence in item.evidence_from_report
                if item.status != "negative"
            ]
            if evidence:
                return "当前直接报告证据包括：" + "；".join(dict.fromkeys(evidence)) + "。"
            return (
                "当前没有可引用的报告原文证据；主要发现来自CT模型，"
                "可在结果中的切片预览和来源分数中继续核对。"
            )
        if intent == "similar_cases" and response.similar_cases:
            cases = "；".join(
                f"{item.case_id}（匹配：{'、'.join(item.matched_labels_zh)}）"
                for item in response.similar_cases[:5]
            )
            return f"当前召回的相似病例为：{cases}。"
        if intent == "medical_knowledge" and "严重" in message:
            return (
                "当前结果能指出异常类型，但没有病灶大小、范围变化和临床症状，"
                "因此不能仅凭分类分数判断严重程度。应重点核对原始CT中的范围、位置及既往对比。"
            )
        if intent == "medical_knowledge" and documents:
            return "检索到的相关知识提示：" + "；".join(
                item.text for item in documents[:3]
            )
        if intent == "result_interpretation" and reference_evaluation:
            strict_hits = "、".join(reference_evaluation["strict_hits"]) or "无"
            candidate_hits = "、".join(reference_evaluation["candidate_hits"]) or "无"
            missed = "、".join(reference_evaluation["missed"]) or "无"
            false_positive = "、".join(reference_evaluation["false_positive"]) or "无"
            return (
                f"按CT-RATE开发集弱标签对照，主要发现命中：{strict_hits}；"
                f"复核候选命中：{candidate_hits}；漏检：{missed}；"
                f"主要发现误报：{false_positive}。"
            )
        main = "、".join(positive) if positive else "未检出达到主要发现阈值的异常"
        review = "、".join(uncertain) if uncertain else "无"
        return f"本病例的主要影像发现为：{main}。建议重点复核：{review}。"

    def _dataset_reference(self, response: AnalyzeResponse) -> dict[str, Any]:
        path = (
            self.qwen.settings.data_dir
            / "dataset"
            / "multi_abnormality_labels"
            / "valid_predicted_labels.csv"
        )
        if not path.exists():
            return {}
        volume_name = response.case_id + ".nii.gz"
        reference: set[str] | None = None
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("VolumeName") != volume_name:
                    continue
                reference = {
                    spec.id
                    for spec in LABEL_SPECS
                    if row.get(spec.source_column) == "1"
                }
                break
        if reference is None:
            return {}
        positive = {item.name for item in response.labels if item.status == "positive"}
        uncertain = {item.name for item in response.labels if item.status == "uncertain"}
        def to_zh(labels: set[str]) -> list[str]:
            return sorted(spec.zh for spec in LABEL_SPECS if spec.id in labels)
        strict_hits = positive & reference
        candidate_hits = uncertain & reference
        return {
            "reference_labels": to_zh(reference),
            "strict_hits": to_zh(strict_hits),
            "candidate_hits": to_zh(candidate_hits),
            "missed": to_zh(reference - positive - uncertain),
            "false_positive": to_zh(positive - reference),
            "strict_precision": len(strict_hits) / len(positive) if positive else 0.0,
            "strict_recall": len(strict_hits) / len(reference) if reference else 0.0,
            "covered_recall": (
                len(strict_hits | candidate_hits) / len(reference) if reference else 0.0
            ),
            "reference_type": "CT-RATE report-derived weak labels",
        }
