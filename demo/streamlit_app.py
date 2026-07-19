import asyncio
from base64 import b64encode
import csv
from html import escape
import json
import os
from pathlib import Path
import uuid

import httpx
import streamlit as st

from chestct_agent.agent.graph import ChestCtAgent
from chestct_agent.config import get_settings
from chestct_agent.input_ingestion import decode_report_bytes, ingest_ct_upload
from chestct_agent.knowledge import INPUT_MODE_ZH, LABEL_ZH, TOOL_TRACE_ZH
from chestct_agent.labels import LABEL_SPECS
from chestct_agent.schemas import (
    AgentState,
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    CorrectionRequest,
    LabelCorrection,
)
from demo.ui_theme import apply_ui_theme, render_product_header


st.set_page_config(page_title="ChestCT Agent", layout="wide")
apply_ui_theme()
render_product_header()

NODE_NAME_ZH = {
    "parse_input": "解析输入",
    "plan_tools": "规划工具",
    "parse_report": "解析报告",
    "run_text_classifier": "报告分类",
    "run_report_graph": "RadGraph-XL报告图谱",
    "run_ct_classifier": "CT 预处理与分类",
    "run_organ_segmentation": "器官分割",
    "run_lesion_grounding": "病灶与区域定位",
    "plan_rag_queries": "规划检索问题",
    "retrieve_medical_knowledge": "检索医学知识",
    "grade_retrieval": "评估检索结果",
    "rewrite_query_if_needed": "改写检索问题",
    "retrieve_similar_cases": "检索相似病例",
    "extract_evidence": "提取证据",
    "check_consistency": "检查多模态一致性",
    "generate_json": "生成结构化结果",
    "validate_output": "校验输出",
    "human_approval": "人工审批门",
    "generate_chinese_explanation": "生成中文结论与Qwen分析说明",
    "apply_external_correction": "应用外部事实纠错",
    "load_case_context": "读取病例与会话记忆",
    "plan_followup": "规划本轮追问工具",
    "retrieve_followup_knowledge": "检索追问相关医学知识",
    "generate_followup_answer": "生成病例追问回答",
    "load_dataset_reference": "对照开发样例参考标签",
    "save_conversation": "保存本轮对话",
}

TOOL_PURPOSE_ZH = {
    "report_parser_tool": "拆分 Findings、Impression 和完整报告",
    "text_classifier_tool": "从报告预测统一的 18 类异常",
    "report_graph_tool": "抽取解剖/观察实体及修饰、位置和提示关系",
    "ct_classifier_tool": "使用 CT-LiPro/CT-CLIP 对 3D CT 进行 18 类分类",
    "organ_segmentation_tool": "读取并对齐 RadGenome 器官及区域 mask",
    "lesion_grounding_tool": "将异常定位到真实切片、bbox 或 mask",
    "medical_rag_tool": "执行 BM25、Dense、RRF 和 Qwen Reranker 检索",
    "similar_case_retriever_tool": "检索非当前病例的 CT-RATE 相似病例",
    "evidence_extractor_tool": "抽取阳性、阴性、不确定和历史报告证据",
    "consistency_checker_tool": "融合 CT 与报告并识别冲突",
    "structured_output_generator": "生成统一的 18 类结构化结果",
    "json_validator_tool": "校验并约束最终 JSON",
    "human_approval_gate": "对冲突、弱证据和高风险结果触发人工审批",
    "explanation_generator": "基于已有证据生成中文结论",
    "conversation_memory_tool": "读取或保存同一病例的多轮对话上下文",
    "followup_planner": "判断本轮问题需要病例证据、RAG还是相似病例",
    "similar_case_context_tool": "读取首次分析时召回的相似病例",
    "qwen_response_generator": "结合病例上下文和历史对话生成回答",
    "dataset_reference_tool": "仅对CT-RATE开发样例计算单病例命中、漏检和误报",
    "human_correction_tool": "应用医生逐标签纠错并保留修改前后的审计记录",
    "dataset_oracle_tool": "在开发沙箱中使用隐藏弱标签反馈，不能用于正式评估",
}

PLANNER_SOURCE_ZH = {
    "llm": "Qwen 动态规划",
    "policy": "规则规划",
    "policy_fallback": "规则降级规划",
}

EVENT_STATUS_ZH = {
    "running": "执行中",
    "success": "成功",
    "recovered": "重试后成功",
    "degraded": "失败后降级",
}

MODEL_REASONING_SOURCE_ZH = {
    "qwen": "Qwen生成",
    "deterministic_fallback": "确定性回退",
    "not_used": "未生成",
}

MODEL_REASONING_STEPS_SOURCE_ZH = {
    "qwen": "Qwen生成",
    "audit_trace": "系统审计轨迹",
    "not_used": "未生成",
}

def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def render_report_graph(response: AnalyzeResponse) -> None:
    graph = response.report_graph
    st.markdown(
        '<p class="section-note">RadGraph-XL 只解析影像报告文字：抽取解剖、观察、'
        '否定/不确定断言及其关系。它不读取 CT，也不产生 CT-CLIP 分类分数。</p>',
        unsafe_allow_html=True,
    )
    if graph.backend == "not_used":
        st.info("本次没有报告输入，未运行报告知识图谱。")
        return
    columns = st.columns(4)
    columns[0].metric("图谱后端", graph.backend)
    columns[1].metric("实体节点", len(graph.nodes))
    columns[2].metric("关系边", len(graph.edges))
    columns[3].metric(
        "规范异常映射", sum(node.canonical_label is not None for node in graph.nodes)
    )
    if graph.warning:
        st.warning(graph.warning)

    assertion_zh = {
        "definitely_present": "明确存在",
        "definitely_absent": "明确否定",
        "uncertain": "不确定",
    }
    entity_zh = {"anatomy": "解剖", "observation": "观察"}
    relation_zh = {
        "modify": "修饰",
        "located_at": "位于",
        "suggestive_of": "提示",
    }
    fill_by_assertion = {
        "definitely_present": "#123B32",
        "definitely_absent": "#25252B",
        "uncertain": "#49370B",
    }
    visible_nodes = sorted(graph.nodes, key=lambda item: item.start_ix)[:60]
    visible_ids = {item.node_id for item in visible_nodes}
    dot = [
        "digraph ReportGraph {",
        'graph [rankdir="LR", bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.65"];',
        'node [shape="box", style="rounded,filled", color="#6E6E78", fontcolor="#F5F5F7", fontname="Microsoft YaHei", fontsize="10", margin="0.12,0.08"];',
        'edge [color="#6E6E78", fontcolor="#B8B8C0", fontname="Microsoft YaHei", fontsize="9", arrowsize="0.7"];',
    ]
    for node in visible_nodes:
        label_parts = [
            node.text,
            f"{entity_zh[node.entity_type]} · {assertion_zh[node.assertion]}",
        ]
        if node.canonical_label:
            label_parts.append(LABEL_ZH.get(node.canonical_label, node.canonical_label))
        label = _dot_escape("\n".join(label_parts))
        fill = fill_by_assertion[node.assertion]
        dot.append(f'n{_dot_escape(node.node_id)} [label="{label}", fillcolor="{fill}"];')
    for edge in graph.edges:
        if edge.source_id not in visible_ids or edge.target_id not in visible_ids:
            continue
        dot.append(
            f'n{_dot_escape(edge.source_id)} -> n{_dot_escape(edge.target_id)} '
            f'[label="{relation_zh[edge.relation]}"];'
        )
    dot.append("}")
    st.graphviz_chart("\n".join(dot), width="stretch", height=560)

    with st.expander("查看实体节点明细"):
        st.dataframe(
            [
                {
                    "ID": node.node_id,
                    "文本": node.text,
                    "实体类型": entity_zh[node.entity_type],
                    "断言状态": assertion_zh[node.assertion],
                    "规范异常": LABEL_ZH.get(
                        node.canonical_label, node.canonical_label or "未映射"
                    ),
                    "报告句": node.sentence,
                }
                for node in graph.nodes
            ],
            hide_index=True,
            width="stretch",
        )
    with st.expander("查看关系边明细"):
        node_text = {node.node_id: node.text for node in graph.nodes}
        st.dataframe(
            [
                {
                    "起点": node_text.get(edge.source_id, edge.source_id),
                    "关系": relation_zh[edge.relation],
                    "终点": node_text.get(edge.target_id, edge.target_id),
                }
                for edge in graph.edges
            ],
            hide_index=True,
            width="stretch",
        )


WORKFLOW_PHASES = (
    ("input", "01", "输入与规划"),
    ("perception", "02", "多模态感知"),
    ("retrieval", "03", "动态检索"),
    ("fusion", "04", "证据融合"),
    ("delivery", "05", "结果交付"),
)

WORKFLOW_NODE_PHASE = {
    "parse_input": "input",
    "plan_tools": "input",
    "parse_report": "perception",
    "run_text_classifier": "perception",
    "run_report_graph": "perception",
    "run_ct_classifier": "perception",
    "run_organ_segmentation": "perception",
    "run_lesion_grounding": "perception",
    "plan_rag_queries": "retrieval",
    "retrieve_medical_knowledge": "retrieval",
    "grade_retrieval": "retrieval",
    "rewrite_query_if_needed": "retrieval",
    "retrieve_similar_cases": "retrieval",
    "extract_evidence": "fusion",
    "check_consistency": "fusion",
    "generate_json": "delivery",
    "validate_output": "delivery",
    "human_approval": "delivery",
    "generate_chinese_explanation": "delivery",
}


def _render_workflow_observatory(response: AnalyzeResponse) -> None:
    events_by_phase = {phase: [] for phase, _, _ in WORKFLOW_PHASES}
    for event in response.execution_events:
        phase = WORKFLOW_NODE_PHASE.get(event.node, "delivery")
        events_by_phase[phase].append(event)

    phase_html = []
    for phase, number, title in WORKFLOW_PHASES:
        events = events_by_phase[phase]
        degraded = any(event.status in {"degraded", "recovered"} for event in events)
        status_class = "attention" if degraded else "complete"
        status_text = "已恢复" if degraded else "完成"
        duration = sum(event.duration_ms for event in events)
        phase_html.append(
            f'<div class="workflow-phase {status_class}">'
            f'<span class="phase-number">{number}</span>'
            '<span class="phase-status-dot"></span>'
            f'<strong>{escape(title)}</strong>'
            f'<small>{len(events)} 个节点 · {duration / 1000:.1f}s · {status_text}</small>'
            "</div>"
        )

    event_html = []
    for event in response.execution_events:
        status_class = "attention" if event.status in {"degraded", "recovered"} else "complete"
        node_name = NODE_NAME_ZH.get(event.node, event.node)
        tool_name = TOOL_TRACE_ZH.get(event.tool, event.tool)
        event_html.append(
            f'<article class="workflow-event {status_class}">'
            f'<span class="event-index">{event.sequence:02d}</span>'
            '<div class="event-copy">'
            f'<div><strong>{escape(node_name)}</strong><span>{escape(EVENT_STATUS_ZH.get(event.status, event.status))}</span></div>'
            f'<p>{escape(event.summary)}</p>'
            f'<small>{escape(tool_name)} · {event.duration_ms} ms · {event.attempts} 次调用</small>'
            "</div></article>"
        )

    plan = response.agent_plan
    planner = (
        PLANNER_SOURCE_ZH.get(plan.generated_by, plan.generated_by)
        if plan
        else "无规划记录"
    )
    rag_state = "充分" if response.rag_trace.final_sufficient else "需复核"
    gate_state = "待医生审批" if response.approval.required else "自动门控通过"
    st.markdown(
        '<section class="agent-observatory">'
        '<header class="observatory-header">'
        '<div><span class="observatory-kicker">AGENT OBSERVATORY</span>'
        '<h3>每一步都留下可验证的执行回执。</h3></div>'
        f'<div class="observatory-state"><span></span>{"降级完成" if response.execution.degraded else "执行完成"}</div>'
        "</header>"
        f'<div class="workflow-phases">{"".join(phase_html)}</div>'
        '<div class="workflow-telemetry">'
        f'<div><span>PLANNER</span><strong>{escape(planner)}</strong></div>'
        f'<div><span>TOOLS</span><strong>{len(plan.steps) if plan else 0}</strong></div>'
        f'<div><span>NODES</span><strong>{len(response.execution_events)}</strong></div>'
        f'<div><span>RAG EVIDENCE</span><strong>{rag_state}</strong></div>'
        f'<div><span>GATE</span><strong>{gate_state}</strong></div>'
        "</div>"
        '<div class="execution-stream-heading"><span>EXECUTION STREAM</span>'
        f'<strong>{len(response.execution_events)} 个节点</strong></div>'
        f'<div class="workflow-events">{"".join(event_html)}</div>'
        "</section>",
        unsafe_allow_html=True,
    )


def render_agent_workbench(response: AnalyzeResponse) -> None:
    plan = response.agent_plan
    _render_workflow_observatory(response)
    st.markdown(
        '<div class="agent-detail-heading"><span>DEEP DIVE</span>'
        '<strong>执行与证据明细</strong></div>',
        unsafe_allow_html=True,
    )

    (
        plan_tab,
        timeline_tab,
        decision_tab,
        rag_tab,
        graph_tab,
        recovery_tab,
    ) = st.tabs(
        [
            "任务规划",
            "执行时间线",
            "决策依据",
            "Agentic RAG",
            "RadGraph-XL",
            "恢复与审批",
        ]
    )

    with plan_tab:
        if plan:
            st.markdown(f"**任务目标：** {plan.objective}")
            st.dataframe(
                [
                    {
                        "顺序": index,
                        "工具": TOOL_TRACE_ZH.get(step.tool, step.tool),
                        "调用原因": TOOL_PURPOSE_ZH.get(step.tool, step.reason),
                        "执行要求": "必须" if step.required else "可选，可降级",
                    }
                    for index, step in enumerate(plan.steps, start=1)
                ],
                hide_index=True,
                width="stretch",
            )
            if plan.fallback_reason:
                st.warning(f"动态规划发生降级：{plan.fallback_reason}")
        else:
            st.info("本次响应没有任务规划记录。")

    with timeline_tab:
        if response.execution_events:
            st.dataframe(
                [
                    {
                        "顺序": event.sequence,
                        "节点": NODE_NAME_ZH.get(event.node, event.node),
                        "工具": TOOL_TRACE_ZH.get(event.tool, event.tool),
                        "状态": EVENT_STATUS_ZH.get(event.status, event.status),
                        "调用次数": event.attempts,
                        "耗时(ms)": event.duration_ms,
                        "本步产出": event.summary,
                    }
                    for event in response.execution_events
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("旧版结果没有逐节点执行记录，请重新分析该病例。")

    with decision_tab:
        if response.execution_events:
            for event in response.execution_events:
                node_name = NODE_NAME_ZH.get(event.node, event.node)
                status = EVENT_STATUS_ZH.get(event.status, event.status)
                with st.expander(
                    f"{event.sequence}. {node_name} | {status}",
                    expanded=event.node in {
                        "plan_tools",
                        "run_ct_classifier",
                        "grade_retrieval",
                        "human_approval",
                    },
                ):
                    st.markdown(f"**决策：** {event.decision_summary or event.summary}")
                    for basis in event.decision_basis:
                        st.markdown(f"- {basis}")
                    if event.key_metrics:
                        st.json(event.key_metrics, expanded=False)
        else:
            st.info("旧版结果没有决策依据记录，请重新分析该病例。")

    with rag_tab:
        rag_trace = response.rag_trace
        if not rag_trace.query_history and not rag_trace.attempts:
            st.info("Agent 判断本次任务不需要医学知识检索，因此没有调用 RAG。")
        else:
            rag_columns = st.columns(3)
            rag_columns[0].metric("检索轮数", len(rag_trace.attempts))
            rag_columns[1].metric(
                "证据判定", "充分" if rag_trace.final_sufficient else "不足"
            )
            rag_columns[2].metric("查询版本", len(rag_trace.query_history))

            for version, queries in enumerate(rag_trace.query_history, start=1):
                label = "初始查询" if version == 1 else f"第 {version - 1} 次改写"
                st.markdown(f"**{label}：** " + "；".join(queries))

            for attempt in rag_trace.attempts:
                verdict = (
                    "充分" if attempt.sufficient is True else "不足" if attempt.sufficient is False else "未评分"
                )
                with st.expander(
                    f"第 {attempt.attempt} 轮 | {attempt.backend} | 证据{verdict}",
                    expanded=True,
                ):
                    st.caption("检索问题：" + "；".join(attempt.queries))
                    rows = []
                    for rank, document in enumerate(attempt.documents, start=1):
                        scores = document.metadata.get("retrieval_scores", {})
                        label = str(document.metadata.get("label", ""))
                        rows.append(
                            {
                                "排名": rank,
                                "文档": document.title,
                                "对应异常": LABEL_ZH.get(label, label),
                                "来源": document.metadata.get("source", ""),
                                "年份": document.metadata.get("publication_year", ""),
                                "PMID/PMCID": document.metadata.get("pmid")
                                or document.metadata.get("pmcid", ""),
                                "原文链接": document.metadata.get("url", ""),
                                "最终分数": document.score,
                                "BM25": scores.get("bm25", 0.0),
                                "Dense": scores.get("dense", 0.0),
                                "RRF": scores.get("rrf", 0.0),
                                "Reranker": scores.get("reranker", 0.0),
                                "证据摘要": document.text[:160],
                            }
                        )
                    st.dataframe(
                        rows,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "原文链接": st.column_config.LinkColumn(
                                "原文链接", display_text="查看来源"
                            )
                        },
                    )

    with graph_tab:
        render_report_graph(response)

    with recovery_tab:
        recovery_columns = st.columns(4)
        recovery_columns[0].metric("Qwen 调用", response.execution.llm_calls)
        recovery_columns[1].metric("LLM 降级", response.execution.llm_fallbacks)
        recovery_columns[2].metric("恢复成功", response.execution.recovered_failures)
        recovery_columns[3].metric(
            "人工审批", "待审批" if response.approval.required else "无需强制审批"
        )
        if response.execution.ct_quality_degraded:
            st.error(response.execution.ct_quality_reason or "CT 分类输出触发质量门控。")
        if response.execution.failed_tools:
            st.warning("发生过失败的工具：" + "、".join(response.execution.failed_tools))
        if response.execution.llm_fallback_reasons:
            st.warning(
                "LLM 降级原因：" + "；".join(response.execution.llm_fallback_reasons)
            )
        if response.approval.reasons:
            st.error("人工审批原因：" + " ".join(response.approval.reasons))
        if not (
            response.execution.failed_tools
            or response.execution.llm_fallback_reasons
            or response.approval.reasons
        ):
            st.success("本次执行没有工具失败、LLM 降级或强制人工审批。")


@st.cache_data
def load_validation_reports(csv_path: str) -> dict[str, str]:
    reports: dict[str, str] = {}
    path = Path(csv_path)
    if not path.exists():
        return reports
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            volume_name = row.get("VolumeName", "")
            findings = row.get("Findings_EN", "")
            impression = row.get("Impressions_EN", "")
            reports[volume_name] = f"Findings: {findings}\nImpression: {impression}"
    return reports


@st.cache_data
def load_validation_labels(csv_path: str) -> dict[str, set[str]]:
    labels_by_volume: dict[str, set[str]] = {}
    path = Path(csv_path)
    if not path.exists():
        return labels_by_volume
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            volume_name = row.get("VolumeName", "")
            labels_by_volume[volume_name] = {
                spec.id for spec in LABEL_SPECS if row.get(spec.source_column) == "1"
            }
    return labels_by_volume


def render_dataset_case_evaluation(response: AnalyzeResponse) -> None:
    if any(
        event.source == "dataset_weak_label" for event in response.correction_history
    ):
        st.error(
            "该结果已经读取隐藏弱标签并发生标签泄漏，开发评估已禁用。"
            "请重新运行原始模型结果后再查看命中、漏检和误报。"
        )
        return
    labels_by_volume = load_validation_labels(
        "data/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
    )
    reference = labels_by_volume.get(response.case_id + ".nii.gz")
    if reference is None:
        return
    positive = {item.name for item in response.labels if item.status == "positive"}
    uncertain = {item.name for item in response.labels if item.status == "uncertain"}
    strict_hits = positive & reference
    candidate_hits = uncertain & reference
    missed = reference - positive - uncertain
    false_positive = positive - reference
    strict_precision = len(strict_hits) / len(positive) if positive else 0.0
    strict_recall = len(strict_hits) / len(reference) if reference else 0.0
    covered_recall = (
        len(strict_hits | candidate_hits) / len(reference) if reference else 0.0
    )

    st.subheader("开发样例标签对照")
    columns = st.columns(4)
    columns[0].metric("主要发现精确率", f"{strict_precision:.0%}")
    columns[1].metric("主要发现召回率", f"{strict_recall:.0%}")
    columns[2].metric("含候选覆盖率", f"{covered_recall:.0%}")
    columns[3].metric("主要发现误报", len(false_positive))
    rows = []
    for result_type, labels in (
        ("主要发现命中", strict_hits),
        ("复核候选命中", candidate_hits),
        ("漏检", missed),
        ("主要发现误报", false_positive),
    ):
        for label in sorted(labels):
            rows.append({"对照结果": result_type, "异常类型": LABEL_ZH.get(label, label)})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "参考来自CT-RATE predicted_labels，是报告派生弱标签；该面板用于开发误差分析，"
        "不把弱标签当作放射科金标准。"
    )


@st.cache_resource
def get_agent() -> ChestCtAgent:
    return ChestCtAgent()


def configured_api_url() -> str:
    return os.environ.get("CHESTCT_API_URL", "http://127.0.0.1:8080").strip()


def analyze_request(request: AnalyzeRequest, progress=None) -> AnalyzeResponse:
    final_response = None
    with httpx.stream(
        "POST",
        configured_api_url().rstrip("/") + "/api/analyze/stream",
        json=request.model_dump(mode="json"),
        timeout=900,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            message = json.loads(line)
            if message.get("type") == "node":
                event = message["event"]
                if progress is not None:
                    render_live_event(progress, event)
            elif message.get("type") == "result":
                final_response = message["response"]
            elif message.get("type") == "error":
                raise RuntimeError(str(message.get("message", "Agent execution failed.")))
    if final_response is None:
        raise RuntimeError("Agent stream ended without a final response.")
    return AnalyzeResponse.model_validate(final_response)


def render_live_event(progress, event: dict[str, object]) -> None:
    node = str(event["node"])
    node_name = NODE_NAME_ZH.get(node, node)
    status_value = str(event["status"])
    if status_value == "running":
        progress.update(
            label=f"当前步骤 {event['sequence']}：{node_name}",
            state="running",
            expanded=True,
        )
        progress.write(f"开始 | {event['summary']}")
        return
    status = EVENT_STATUS_ZH.get(status_value, status_value)
    progress.write(
        f"完成 | {node_name} | {status} | "
        f"{float(event['duration_ms']):.0f} ms | {event['summary']}"
    )
    decision = str(event.get("decision_summary", "")).strip()
    if decision:
        progress.write(f"依据 | {decision}")
    basis = event.get("decision_basis")
    if isinstance(basis, list) and basis:
        progress.caption("依据明细 | " + "；".join(str(item) for item in basis[:3]))


def analyze_upload_request(
    case_id: str,
    session_id: str,
    question: str,
    report_text: str,
    ct_file,
    report_file,
    require_human_approval: bool,
    progress=None,
) -> AnalyzeResponse:
    api_url = configured_api_url()
    if api_url:
        files = {}
        if ct_file is not None:
            files["ct_file"] = (
                ct_file.name,
                ct_file.getvalue(),
                ct_file.type or "application/octet-stream",
            )
        if report_file is not None:
            files["report_file"] = (
                report_file.name,
                report_file.getvalue(),
                report_file.type or "text/plain",
            )
        final_response = None
        with httpx.stream(
            "POST",
            api_url.rstrip("/") + "/api/analyze/upload/stream",
            data={
                "case_id": case_id,
                "session_id": session_id,
                "question": question,
                "report_text": report_text,
                "require_human_approval": str(require_human_approval).lower(),
            },
            files=files or None,
            timeout=900,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                message = json.loads(line)
                if message.get("type") == "node":
                    event = message["event"]
                    if progress is not None:
                        render_live_event(progress, event)
                elif message.get("type") == "result":
                    final_response = message["response"]
                elif message.get("type") == "error":
                    raise RuntimeError(str(message.get("message", "Agent execution failed.")))
        if final_response is None:
            raise RuntimeError("Agent stream ended without a final response.")
        return AnalyzeResponse.model_validate(final_response)

    report_parts = [report_text.strip()] if report_text.strip() else []
    if report_file is not None:
        report_parts.append(decode_report_bytes(report_file.getvalue()))
    ct_path = None
    if ct_file is not None:
        ct_path = ingest_ct_upload(
            ct_file.name,
            ct_file.getvalue(),
            case_id,
            get_settings().upload_dir,
        )
    request = AnalyzeRequest(
        case_id=case_id,
        session_id=session_id,
        report_text="\n".join(report_parts),
        question=question,
        ct_volume_path=str(ct_path) if ct_path else None,
        ct_source_name=ct_file.name if ct_file is not None else None,
        require_human_approval=require_human_approval,
    )
    return asyncio.run(get_agent().run(AgentState(request=request)))


def submit_human_corrections(
    case_id: str,
    session_id: str,
    reviewer: str,
    corrections: list[LabelCorrection],
) -> AnalyzeResponse:
    request = CorrectionRequest(
        session_id=session_id,
        reviewer=reviewer,
        source="human",
        corrections=corrections,
    )
    api_url = configured_api_url()
    if api_url:
        response = httpx.post(
            api_url.rstrip("/") + f"/api/cases/{case_id}/corrections",
            json=request.model_dump(mode="json"),
            timeout=180,
        )
        response.raise_for_status()
        return AnalyzeResponse.model_validate(response.json())
    return asyncio.run(get_agent().correct_case(case_id, request))


def submit_dataset_sandbox_correction(
    case_id: str, session_id: str
) -> AnalyzeResponse:
    api_url = configured_api_url()
    if api_url:
        response = httpx.post(
            api_url.rstrip("/") + f"/api/cases/{case_id}/sandbox-correct",
            json={"session_id": session_id},
            timeout=180,
        )
        response.raise_for_status()
        return AnalyzeResponse.model_validate(response.json())
    return asyncio.run(get_agent().correct_case_with_dataset(case_id, session_id))


def chat_request(
    session_id: str,
    case_id: str,
    message: str,
    progress=None,
) -> ChatResponse:
    request = ChatRequest(session_id=session_id, case_id=case_id, message=message)
    api_url = configured_api_url()

    final_response = None
    with httpx.stream(
        "POST",
        api_url.rstrip("/") + "/api/chat/stream",
        json=request.model_dump(mode="json"),
        timeout=180,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("type") == "node":
                event = payload["event"]
                if progress is not None:
                    render_live_event(progress, event)
            elif payload.get("type") == "result":
                final_response = payload["response"]
            elif payload.get("type") == "error":
                raise RuntimeError(str(payload.get("message", "追问 Agent 执行失败。")))
    if final_response is None:
        raise RuntimeError("追问 Agent 未返回最终回答。")
    return ChatResponse.model_validate(final_response)


@st.cache_data
def load_ablation_metrics(evaluation_dir: str) -> list[dict[str, object]]:
    patient_path = Path(evaluation_dir) / "ablation_patient_metrics.json"
    if patient_path.exists():
        payload = json.loads(patient_path.read_text(encoding="utf-8"))
        rows = []
        names = {"report": "报告文本", "ct": "3D CT", "fusion": "CT + 报告融合"}
        for source in ("report", "ct", "fusion"):
            item = payload.get("sources", {}).get(source, {})
            metrics = item.get("calibrated", {})
            if not metrics:
                continue
            rows.append(
                {
                    "模式": names[source],
                    "患者数": metrics.get("patients", 0),
                    "Micro-F1": metrics.get("micro_f1"),
                    "Macro-F1": metrics.get("macro_f1"),
                    "Macro-AUROC": metrics.get("macro_auroc"),
                    "Macro-AUPRC": metrics.get("macro_auprc"),
                    "ECE": metrics.get("macro_ece"),
                }
            )
        return rows

    rows: list[dict[str, object]] = []
    for mode in ("report_only", "ct_only", "multimodal"):
        path = Path(evaluation_dir) / f"{mode}_metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        classification = metrics.get("classification", {})
        latency = metrics.get("latency_ms", {})
        rows.append(
            {
                "模式": INPUT_MODE_ZH.get(
                    "report_and_ct" if mode == "multimodal" else mode,
                    mode,
                ),
                "病例数": classification.get("matched_cases", 0),
                "Micro-F1": classification.get("micro_f1"),
                "Macro-F1": classification.get("macro_f1"),
                "平均耗时(ms)": latency.get("mean"),
                "路由完整率": metrics.get("tool_route_complete_rate"),
            }
        )
    return rows


@st.cache_data
def load_ctclip_metrics(metrics_path: str) -> dict[str, object]:
    path = Path(metrics_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_json_artifact(path: str) -> dict[str, object]:
    artifact = Path(path)
    if not artifact.exists():
        return {}
    return json.loads(artifact.read_text(encoding="utf-8"))


@st.cache_data
def load_ct_model_card(variant: str) -> dict[str, object]:
    metrics_path = Path("artifacts/evaluation") / f"ablation_patient_metrics_{variant}.json"
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    sources = payload.get("sources", {})
    metrics = sources.get("ct", {}).get("calibrated", {})
    return {
        "variant": variant,
        "split": payload.get("split", {}),
        "metrics": metrics,
        "source_metrics": {
            source: sources.get(source, {}).get("calibrated", {})
            for source in ("report", "ct", "fusion")
        },
        "label_source": payload.get("label_source", ""),
    }


def render_result_model_card(response: AnalyzeResponse) -> None:
    if response.execution.input_mode == "report_only":
        return
    variant = response.execution.ct_model_variant
    names = {"lipro": "CT-LiPro v2（18类微调）", "zeroshot": "CT-CLIP v2（零样本）"}
    card = load_ct_model_card(variant)
    st.subheader("结果依据与模型表现")
    if not card:
        st.warning(f"{names.get(variant, variant)} 尚未完成本机患者级独立测试，当前结果不能作为项目关键结果。")
        return

    metrics = card["metrics"]
    st.markdown(f"**影像模型：{names.get(variant, variant)}**")
    columns = st.columns(6)
    columns[0].metric("独立测试患者", int(metrics.get("patients", 0)))
    columns[1].metric("Micro-P", f"{metrics.get('micro_precision', 0):.3f}")
    columns[2].metric("Micro-R", f"{metrics.get('micro_recall', 0):.3f}")
    columns[3].metric("Micro-F1", f"{metrics.get('micro_f1', 0):.3f}")
    columns[4].metric("Macro-AUROC", f"{metrics.get('macro_auroc', 0):.3f}")
    columns[5].metric("Macro-AUPRC", f"{metrics.get('macro_auprc', 0):.3f}")

    if response.execution.input_mode == "report_and_ct":
        source_names = {"report": "报告文本", "ct": "3D CT", "fusion": "CT + 报告融合"}
        comparison = []
        for source in ("report", "ct", "fusion"):
            source_metrics = card["source_metrics"].get(source, {})
            if not source_metrics:
                continue
            comparison.append(
                {
                    "输入模式": source_names[source],
                    "Micro-P": source_metrics.get("micro_precision"),
                    "Micro-R": source_metrics.get("micro_recall"),
                    "Micro-F1": source_metrics.get("micro_f1"),
                    "Macro-F1": source_metrics.get("macro_f1"),
                    "Macro-AUROC": source_metrics.get("macro_auroc"),
                    "Macro-AUPRC": source_metrics.get("macro_auprc"),
                }
            )
        st.dataframe(comparison, hide_index=True, width="stretch")

    selected = sorted(
        (item for item in response.labels if item.status in {"positive", "uncertain"}),
        key=lambda item: item.confidence,
        reverse=True,
    )[:8]
    per_label = metrics.get("per_label", {})
    rows = []
    for item in selected:
        validation = per_label.get(item.name, {})
        rows.append(
            {
                "本病例候选": item.name_zh,
                "状态": item.status_zh,
                "本病例分数": item.source_scores.get("ct_model", item.confidence),
                "测试集F1": validation.get("f1"),
                "测试集AUROC": validation.get("auroc"),
                "测试集AUPRC": validation.get("auprc"),
                "测试集阳性数": validation.get("positive_count"),
            }
        )
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "指标来自本地患者级无重叠测试集，参考标签为CT-RATE报告派生弱标签；"
        "它能支撑课程项目的可复现实验结论，但不能等同于临床外部验证。"
    )


def _finding_group_html(title: str, status: str, labels: list, empty_text: str) -> str:
    rows = "".join(
        '<div class="finding-row">'
        f'<span class="finding-name">{escape(label.name_zh)}</span>'
        f'<span class="finding-score">{label.confidence:.2f}</span>'
        "</div>"
        for label in labels
    )
    if not rows:
        rows = f'<div class="empty-finding">{escape(empty_text)}</div>'
    return (
        f'<section class="finding-group {status}">'
        '<div class="finding-heading">'
        f"<strong>{escape(title)}</strong><span>{len(labels)} 项</span>"
        "</div>"
        f"{rows}</section>"
    )


def render_label_overview(response: AnalyzeResponse) -> None:
    positives = sorted(
        (label for label in response.labels if label.status == "positive"),
        key=lambda item: item.confidence,
        reverse=True,
    )
    uncertain = sorted(
        (label for label in response.labels if label.status == "uncertain"),
        key=lambda item: item.confidence,
        reverse=True,
    )
    negatives = sorted(
        (label for label in response.labels if label.status == "negative"),
        key=lambda item: item.name_zh,
    )
    positive_title = (
        "主要影像发现"
        if response.execution.input_mode == "ct_only"
        else "主要阳性结论"
    )
    st.markdown(
        '<div class="finding-grid">'
        + _finding_group_html(positive_title, "positive", positives, "未检出主要阳性结论")
        + _finding_group_html("待人工复核", "uncertain", uncertain, "当前没有不确定候选")
        + "</div>",
        unsafe_allow_html=True,
    )
    if negatives:
        with st.expander(f"查看其余 {len(negatives)} 项阴性结果"):
            st.dataframe(
                [
                    {
                        "异常": label.name_zh,
                        "状态": label.status_zh,
                        "可信分数": label.confidence,
                    }
                    for label in negatives
                ],
                hide_index=True,
                width="stretch",
            )


def render_candidate_evidence(response: AnalyzeResponse) -> None:
    candidates = sorted(
        (
            label
            for label in response.labels
            if label.status in {"positive", "uncertain"}
        ),
        key=lambda item: (item.status != "positive", -item.confidence),
    )
    if not candidates:
        st.info("本次没有需要展开的阳性或不确定证据。")
        return
    for label in candidates:
        with st.expander(
            f"{label.name_zh} · {label.status_zh} · {label.confidence:.2f}",
            expanded=label.status == "positive",
        ):
            score_columns = st.columns(3)
            score_columns[0].metric("融合分数", f"{label.confidence:.2f}")
            score_columns[1].metric(
                "CT 模型", f"{label.source_scores.get('ct_model', 0.0):.2f}"
            )
            score_columns[2].metric(
                "报告模型", f"{label.source_scores.get('report_model', 0.0):.2f}"
            )
            if label.evidence_from_report:
                st.markdown("**报告原文证据**")
                for item in label.evidence_from_report:
                    st.write(item)
            else:
                st.caption("没有可定位的报告原文证据。")
            if label.evidence_from_image.preview_images:
                st.caption(label.evidence_from_image.note)
                image_columns = st.columns(
                    min(3, len(label.evidence_from_image.preview_images))
                )
                for column, image_path in zip(
                    image_columns,
                    label.evidence_from_image.preview_images,
                    strict=False,
                ):
                    path = Path(image_path)
                    if path.exists():
                        column.image(str(path), caption=path.name, width="stretch")


def render_model_reasoning(response: AnalyzeResponse) -> None:
    reasoning = response.model_reasoning
    st.subheader("Qwen结果形成说明")
    reasoning_columns = st.columns(3)
    reasoning_columns[0].metric(
        "生成来源",
        MODEL_REASONING_SOURCE_ZH.get(reasoning.generated_by, reasoning.generated_by),
    )
    reasoning_columns[1].metric(
        "步骤封装",
        MODEL_REASONING_STEPS_SOURCE_ZH.get(
            reasoning.structured_steps_by, reasoning.structured_steps_by
        ),
    )
    reasoning_columns[2].metric("限制项", len(reasoning.limitations))
    if reasoning.raw_response_zh:
        with st.expander("查看 Qwen 公开分析说明"):
            st.markdown(reasoning.raw_response_zh)
    elif reasoning.summary_zh:
        st.markdown(reasoning.summary_zh)
    if reasoning.steps:
        st.markdown("**结构化审计对照：**")
    for step in reasoning.steps:
        with st.expander(
            f"{step.order}. {step.stage}",
            expanded=False,
        ):
            st.markdown(f"**判断：** {step.decision}")
            if step.evidence:
                st.markdown("**采用依据：**")
                for evidence in step.evidence:
                    st.markdown(f"- {evidence}")
            if step.uncertainty:
                st.warning(step.uncertainty)
    if reasoning.limitations:
        st.markdown("**仍需注意：**")
        for limitation in reasoning.limitations:
            st.markdown(f"- {limitation}")


def _local_image_data_url(path_value: str | Path) -> str:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return ""
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{b64encode(path.read_bytes()).decode('ascii')}"


def _render_result_stage(response: AnalyzeResponse) -> None:
    positives = sorted(
        (label for label in response.labels if label.status == "positive"),
        key=lambda item: item.confidence,
        reverse=True,
    )
    uncertain = sorted(
        (label for label in response.labels if label.status == "uncertain"),
        key=lambda item: item.confidence,
        reverse=True,
    )
    focus = (positives or uncertain or [None])[0]
    if focus is None:
        title = "未检出主要异常"
        eyebrow = "NO PRIMARY FINDING"
        confidence = "—"
        confidence_label = "融合可信分数"
        score_detail = "18 类结果均未达到候选阈值"
    else:
        title = focus.name_zh
        eyebrow = "PRIMARY FINDING" if focus.status == "positive" else "REVIEW CANDIDATE"
        confidence = f"{focus.confidence:.2f}"
        confidence_label = (
            "融合可信分数" if focus.decision_source == "model" else "纠错前模型分数"
        )
        score_detail = (
            f"CT {focus.source_scores.get('ct_model', 0.0):.2f} · "
            f"报告 {focus.source_scores.get('report_model', 0.0):.2f}"
        )

    image_paths: list[str] = []
    if focus is not None:
        image_paths.extend(focus.evidence_from_image.preview_images)
    image_paths.extend(response.ct_preview_images)
    image_url = ""
    image_name = "CT 证据影像"
    for image_path in image_paths:
        image_url = _local_image_data_url(image_path)
        if image_url:
            image_name = Path(image_path).name
            break

    explanation = response.explanation_zh.replace("**", "").strip()
    correction_source = (
        response.correction_history[-1].source if response.correction_history else None
    )
    if correction_source == "human":
        review_state = "医生逐标签复核已完成"
    elif correction_source == "dataset_weak_label":
        review_state = "隐藏弱标签训练沙箱"
    else:
        review_state = "建议医生复核" if response.approval.required else "证据融合已完成"
    status_class = "review" if response.approval.required else "ready"
    image_html = (
        f'<img src="{image_url}" alt="{escape(image_name)}">'
        if image_url
        else '<div class="result-media-empty">3D CT EVIDENCE</div>'
    )
    st.markdown(
        '<section class="result-stage">'
        '<div class="result-stage-copy">'
        f'<span class="result-eyebrow">{eyebrow}</span>'
        f'<h2>{escape(title)}</h2>'
        f'<p>{escape(explanation)}</p>'
        f'<div class="result-confidence"><strong>{confidence}</strong>'
        f'<span>{escape(confidence_label)}<br><small>{escape(score_detail)}</small></span></div>'
        f'<div class="result-gate {status_class}"><span></span>{review_state}</div>'
        "</div>"
        '<div class="result-stage-media">'
        f'{image_html}<div class="result-reticle"></div><div class="result-scan-line"></div>'
        f'<div class="result-image-meta"><span>AXIAL · LUNG WINDOW</span><strong>{escape(image_name)}</strong></div>'
        "</div></section>"
        '<div class="result-facts">'
        f'<div><span>INPUT</span><strong>{escape(INPUT_MODE_ZH.get(response.execution.input_mode, response.execution.input_mode))}</strong></div>'
        f'<div><span>POSITIVE</span><strong>{len(positives)}</strong></div>'
        f'<div><span>REVIEW</span><strong>{len(uncertain)}</strong></div>'
        f'<div><span>LATENCY</span><strong>{response.execution.total_latency_ms / 1000:.1f}s</strong></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    review_notes = list(dict.fromkeys([*response.warnings, *response.approval.reasons]))
    if review_notes:
        st.markdown(
            '<div class="result-review-note"><span>REVIEW GATE</span>'
            f'<p>{escape(" ".join(review_notes))}</p></div>',
            unsafe_allow_html=True,
        )


def render_analysis_result(response: AnalyzeResponse) -> None:
    _render_result_stage(response)

    provenance_items = [f"<strong>病例</strong> {escape(response.case_id)}"]
    if response.execution.ct_input_name:
        size_text = "未知大小"
        if response.execution.ct_input_size_bytes is not None:
            size_text = f"{response.execution.ct_input_size_bytes / (1024 ** 2):.1f} MB"
        fingerprint = (
            response.execution.ct_input_sha256[:16]
            if response.execution.ct_input_sha256
            else "未计算"
        )
        provenance_items.extend(
            [
                f"<strong>CT</strong> {escape(response.execution.ct_input_name)}",
                f"<strong>SHA-256</strong> {escape(fingerprint)}",
                f"<strong>大小</strong> {escape(size_text)}",
            ]
        )
    st.markdown(
        '<div class="input-provenance">'
        + "".join(f"<span>{item}</span>" for item in provenance_items)
        + "</div>",
        unsafe_allow_html=True,
    )

    conclusion_tab, evidence_tab, agent_tab, model_tab, data_tab = st.tabs(
        ["临床结论", "证据与影像", "Agent 轨迹", "模型评估", "结构化数据"]
    )

    with conclusion_tab:
        st.subheader("中文结论")
        st.markdown(response.explanation_zh)
        st.divider()
        st.subheader("异常概览")
        render_label_overview(response)

    with evidence_tab:
        if response.ct_preview_images:
            st.subheader("CT 预览")
            preview_columns = st.columns(min(5, len(response.ct_preview_images)))
            for column, image_path in zip(
                preview_columns, response.ct_preview_images, strict=False
            ):
                path = Path(image_path)
                if path.exists():
                    column.image(str(path), caption=path.name, width="stretch")
        st.subheader("逐项证据")
        render_candidate_evidence(response)
        if response.region_findings:
            st.subheader("区域级报告")
            st.dataframe(
                [finding.model_dump() for finding in response.region_findings],
                hide_index=True,
                width="stretch",
            )
        st.subheader("相似病例")
        if response.similar_cases:
            strategy_labels = {
                "report_text": "报告文本",
                "predicted_conditions": "CT预测条件",
                "hybrid": "报告+预测条件",
                "region_aware": "报告/条件+解剖区域",
            }
            st.dataframe(
                [
                    {
                        "病例 ID": case.case_id,
                        "患者 ID": case.patient_id,
                        "相似度": case.score,
                        "检索策略": strategy_labels.get(
                            case.retrieval_strategy, case.retrieval_strategy
                        ),
                        "报告相似": case.score_breakdown.get("report_similarity", 0.0),
                        "条件重合": case.score_breakdown.get("condition_overlap", 0.0),
                        "区域重合": case.score_breakdown.get("region_overlap", 0.0),
                        "匹配异常": "、".join(case.matched_labels_zh),
                        "匹配区域": "、".join(case.matched_regions),
                        "来源": case.source,
                        "报告原文摘要": case.summary,
                    }
                    for case in response.similar_cases
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("本次没有召回非自身相似病例。")

    with agent_tab:
        render_agent_workbench(response)
        st.divider()
        render_model_reasoning(response)

    with model_tab:
        render_result_model_card(response)

    with data_tab:
        st.json(response.model_dump())


def render_correction_loop(
    response: AnalyzeResponse, session_id: str, developer_demo: bool
) -> None:
    st.divider()
    st.subheader("纠错闭环")
    st.caption(
        "模型不能自证正确。这里接收独立反馈，保留原始预测和证据，再由Qwen根据纠正后的结构化状态重写结论。"
    )

    if response.correction_history:
        history_rows = []
        for event in response.correction_history:
            changed = [
                item for item in event.items if item.before_status != item.after_status
            ]
            history_rows.append(
                {
                    "时间": event.created_at,
                    "来源": "医生复核" if event.source == "human" else "隐藏弱标签沙箱",
                    "复核者": event.reviewer,
                    "提交标签": len(event.items),
                    "实际修改": len(changed),
                    "修改明细": "；".join(
                        f"{LABEL_ZH.get(item.label, item.label)}："
                        f"{item.before_status}→{item.after_status}"
                        for item in changed
                    ) or "状态确认，无修改",
                }
            )
        st.dataframe(history_rows, hide_index=True, width="stretch")

    if developer_demo:
        already_applied = any(
            event.source == "dataset_weak_label"
            for event in response.correction_history
        )
        st.warning(
            "训练沙箱会向Agent揭示CT-RATE报告派生弱标签并据此改答案，存在标签泄漏。"
            "它只演示‘获得事实反馈后如何纠错’，纠错后的结果不能计入模型评估。"
        )
        if st.button(
            "运行隐藏标签纠错沙箱",
            icon=":material/model_training:",
            disabled=already_applied,
            width="stretch",
        ):
            try:
                corrected = submit_dataset_sandbox_correction(response.case_id, session_id)
            except (RuntimeError, httpx.HTTPError) as exc:
                st.error(f"沙箱纠错失败：{exc}")
            else:
                st.session_state.analysis_response = corrected.model_dump(mode="json")
                chat_key = f"chat_messages::{session_id}::{response.case_id}"
                st.session_state.setdefault(chat_key, []).append(
                    {"role": "assistant", "content": corrected.explanation_zh}
                )
                st.rerun()
        return

    source_name = {
        "model": "模型",
        "human_correction": "医生纠正",
        "dataset_oracle": "弱标签沙箱",
    }
    rows = [
        {
            "label": item.name,
            "异常": item.name_zh,
            "当前结论": item.status_zh,
            "当前来源": source_name[item.decision_source],
            "医生结论": "保留当前",
            "纠错理由": "",
        }
        for item in response.labels
    ]
    with st.form(f"correction_form::{response.case_id}::{len(response.correction_history)}"):
        reviewer = st.text_input("复核医生/标注员", placeholder="填写脱敏工号或姓名缩写")
        edited = st.data_editor(
            rows,
            hide_index=True,
            width="stretch",
            disabled=["label", "异常", "当前结论", "当前来源"],
            column_config={
                "label": None,
                "医生结论": st.column_config.SelectboxColumn(
                    options=["保留当前", "阳性", "阴性", "不确定"],
                    required=True,
                ),
                "纠错理由": st.column_config.TextColumn(
                    help="记录支持纠错的报告、切片、病理或随访依据"
                ),
            },
            key=f"correction_editor::{response.case_id}::{len(response.correction_history)}",
        )
        submitted = st.form_submit_button(
            "提交逐标签纠错",
            type="primary",
            icon=":material/fact_check:",
            width="stretch",
        )
    if not submitted:
        return
    if not reviewer.strip():
        st.error("必须填写复核者，纠错记录不能匿名写入审计库。")
        return
    status_by_zh = {"阳性": "positive", "阴性": "negative", "不确定": "uncertain"}
    edited_rows = edited.to_dict(orient="records") if hasattr(edited, "to_dict") else edited
    current_status = {item.name: item.status for item in response.labels}
    corrections = []
    for row in edited_rows:
        label = str(row["label"])
        selected = str(row["医生结论"])
        corrections.append(
            LabelCorrection(
                label=label,
                corrected_status=(
                    current_status[label]
                    if selected == "保留当前"
                    else status_by_zh[selected]
                ),
                reason=str(row["纠错理由"]).strip(),
            )
        )
    try:
        corrected = submit_human_corrections(
            response.case_id, session_id, reviewer.strip(), corrections
        )
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        st.error(f"提交纠错失败：{exc}")
        return
    st.session_state.analysis_response = corrected.model_dump(mode="json")
    chat_key = f"chat_messages::{session_id}::{response.case_id}"
    st.session_state.setdefault(chat_key, []).append(
        {"role": "assistant", "content": corrected.explanation_zh}
    )
    st.rerun()


def render_case_chat(response: AnalyzeResponse, session_id: str) -> None:
    st.divider()
    st.subheader("病例多轮问答")
    state_key = f"chat_messages::{session_id}::{response.case_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = [
            {"role": "assistant", "content": response.explanation_zh}
        ]

    for message in st.session_state[state_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "继续询问本病例的发现、证据、严重程度或相似病例",
        key=f"case_chat::{response.case_id}",
    )
    if not prompt:
        return

    st.session_state[state_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        live_status = None
        try:
            with st.status("追问 Agent 正在执行", expanded=True) as live_status:
                chat_response = chat_request(
                    session_id,
                    response.case_id,
                    prompt,
                    progress=live_status,
                )
                live_status.update(
                    label="本轮追问完成", state="complete", expanded=False
                )
        except (ValueError, RuntimeError, httpx.HTTPError) as exc:
            if live_status is not None:
                live_status.update(
                    label="本轮追问失败", state="error", expanded=True
                )
            st.error(str(exc))
            return

        st.markdown(chat_response.answer_zh)
        with st.expander("本轮 Agent 执行步骤"):
            st.dataframe(
                [
                    {
                        "顺序": event.sequence,
                        "步骤": NODE_NAME_ZH.get(event.node, event.node),
                        "工具": TOOL_TRACE_ZH.get(event.tool, event.tool),
                        "耗时(ms)": event.duration_ms,
                        "产出": event.summary,
                    }
                    for event in chat_response.execution_events
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "本轮路由："
                + chat_response.intent
                + " | 模型："
                + ("Qwen API" if chat_response.used_remote_model else "本地规则降级")
            )
            if chat_response.retrieved_documents:
                st.dataframe(
                    [
                        {
                            "文档": item.title,
                            "分数": item.score,
                            "来源": item.metadata.get("source", ""),
                            "摘要": item.text[:180],
                        }
                        for item in chat_response.retrieved_documents
                    ],
                    hide_index=True,
                    width="stretch",
                )
            if chat_response.reference_evaluation:
                reference = chat_response.reference_evaluation
                st.write(
                    {
                        "主要发现命中": reference.get("strict_hits", []),
                        "复核候选命中": reference.get("candidate_hits", []),
                        "漏检": reference.get("missed", []),
                        "主要发现误报": reference.get("false_positive", []),
                    }
                )
    st.session_state[state_key].append(
        {"role": "assistant", "content": chat_response.answer_zh}
    )


if "chestct_session_id" not in st.session_state:
    st.session_state.chestct_session_id = uuid.uuid4().hex
session_id = st.session_state.chestct_session_id

with st.sidebar:
    st.markdown("### 工作区")
    workspace_mode = st.segmented_control(
        "数据来源",
        ["上传检查", "开发样例"],
        default="上传检查",
        selection_mode="single",
        width="stretch",
    )
    st.divider()
    st.caption(f"会话 {session_id[:8]}")

developer_demo = workspace_mode == "开发样例"
default_question = (
    "该检查可能存在哪些胸部异常，位于哪些区域，有什么报告和图像证据？"
    "请检索相关医学知识和相似病例。"
)

if not developer_demo:
    st.markdown('<div class="workspace-kicker">NEW STUDY</div>', unsafe_allow_html=True)
    st.subheader("新建分析")
    with st.form("patient_upload_form"):
        identity_columns = st.columns([1, 2])
        case_id = identity_columns[0].text_input(
            "检查编号",
            value="uploaded_case",
            help="建议使用脱敏编号，请勿填写患者姓名或身份证号。",
        )
        question = identity_columns[1].text_input("希望分析的问题", value=default_question)
        upload_columns = st.columns(2)
        with upload_columns[0]:
            ct_file = st.file_uploader(
                "胸部 CT 检查",
                type=["nii", "gz", "zip"],
                help="支持 NIfTI（.nii/.nii.gz）或包含同一检查 DICOM 序列的 ZIP。",
            )
        with upload_columns[1]:
            report_file = st.file_uploader(
                "影像报告文件（可选）",
                type=["txt"],
            )
        report_text = st.text_area("影像报告文字（可选）", value="", height=104)
        action_columns = st.columns([2, 1])
        with action_columns[0]:
            require_human_approval = st.checkbox("提交给医生人工复核", value=True)
        with action_columns[1]:
            analysis_clicked = st.form_submit_button(
                "开始分析",
                type="primary",
                icon=":material/play_arrow:",
                width="stretch",
            )
    selected_path = None
else:
    volume_paths = sorted(Path("data/dataset/valid_fixed").rglob("*.nii.gz"))
    volume_by_name = {path.name: path.resolve() for path in volume_paths}
    reports = load_validation_reports(
        "data/dataset/radiology_text_reports/validation_reports.csv"
    )
    st.markdown('<div class="workspace-kicker">DEVELOPMENT DATA</div>', unsafe_allow_html=True)
    st.subheader("开发样例")
    with st.form("developer_demo_form"):
        selected_case = st.selectbox("CT-RATE 样例", list(volume_by_name), index=0)
        selected_path = volume_by_name[selected_case]
        case_id = selected_case.removesuffix(".nii.gz")
        question = st.text_input("测试问题", value=default_question)
        report_text = st.text_area(
            "数据集报告",
            value=reports.get(selected_case, ""),
            height=120,
        )
        require_human_approval = st.checkbox("强制人工复核", value=False)
        analysis_clicked = st.form_submit_button(
            "运行开发样例",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        )
    ct_file = None
    report_file = None

if analysis_clicked:
    st.session_state.pop("analysis_response", None)
    st.session_state.pop("analysis_session_id", None)
    live_status = None
    try:
        if developer_demo:
            with st.status("Agent 正在执行", expanded=True) as live_status:
                request = AnalyzeRequest(
                    case_id=case_id,
                    session_id=session_id,
                    report_text=report_text,
                    question=question,
                    ct_volume_path=str(selected_path),
                    ct_source_name=selected_path.name,
                    require_human_approval=require_human_approval,
                )
                response = analyze_request(request, progress=live_status)
                live_status.update(
                    label="Agent 工作流完成", state="complete", expanded=False
                )
        else:
            with st.status("Agent 正在执行", expanded=True) as live_status:
                response = analyze_upload_request(
                    case_id,
                    session_id,
                    question,
                    report_text,
                    ct_file,
                    report_file,
                    require_human_approval,
                    progress=live_status,
                )
                live_status.update(label="Agent 工作流完成", state="complete", expanded=False)
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        if live_status is not None:
            live_status.update(label="Agent 工作流失败", state="error", expanded=True)
        st.error(f"无法开始分析：{exc}")
        st.stop()
    st.session_state.analysis_response = response.model_dump(mode="json")
    st.session_state.analysis_session_id = session_id


stored_response = st.session_state.get("analysis_response")
if stored_response:
    response = AnalyzeResponse.model_validate(stored_response)
    render_analysis_result(response)
    render_correction_loop(
        response,
        st.session_state.get("analysis_session_id", session_id),
        developer_demo,
    )
    if developer_demo:
        render_dataset_case_evaluation(response)
    render_case_chat(
        response,
        st.session_state.get("analysis_session_id", session_id),
    )


if not developer_demo:
    st.stop()


ablation_metrics = load_ablation_metrics("artifacts/evaluation")
if ablation_metrics:
    st.divider()
    st.subheader("评估结果概览")
    st.dataframe(ablation_metrics, hide_index=True, width="stretch")
    case_count = max(int(row.get("患者数", row.get("病例数", 0))) for row in ablation_metrics)
    if case_count < 30:
        st.info(
            f"当前仅有 {case_count} 个本地验证病例，且使用数据集提供的弱标签；"
            "这些指标只用于检查流程，不能代表模型真实性能。"
        )


evidence_metrics = load_json_artifact("artifacts/evaluation/report_evidence_metrics.json")
rag_metrics = load_json_artifact("artifacts/evaluation/rag_metrics.json")
similar_case_metrics = load_json_artifact(
    "artifacts/evaluation/similar_case_metrics.json"
)
prompt_metrics = load_json_artifact("artifacts/evaluation/qwen_prompt_metrics.json")
if evidence_metrics or rag_metrics or similar_case_metrics or prompt_metrics:
    st.divider()
    st.subheader("文本、检索与 Agent 评估")
    summary_rows = []
    if prompt_metrics:
        metrics = prompt_metrics.get("metrics", {})
        summary_rows.append(
            {
                "实验": "Qwen Prompt 报告分类",
                "样本": prompt_metrics.get("cases"),
                "指标1": f"Macro-F1 {metrics.get('macro_f1', 0):.3f}",
                "指标2": f"Macro-AUROC {metrics.get('macro_auroc', 0):.3f}",
                "备注": "患者级子集；远端 fallback 数为 " + str(prompt_metrics.get("fallback_count", 0)),
            }
        )
    if evidence_metrics:
        metrics = evidence_metrics.get("positive_evidence_proxy", {})
        summary_rows.append(
            {
                "实验": "报告证据句代理评估",
                "样本": evidence_metrics.get("cases"),
                "指标1": f"Micro-P {metrics.get('micro_precision', 0):.3f}",
                "指标2": f"Micro-R/F1 {metrics.get('micro_recall', 0):.3f}/{metrics.get('micro_f1', 0):.3f}",
                "备注": "弱标签代理，不是人工 span 金标准",
            }
        )
    if rag_metrics:
        baseline = rag_metrics.get("baseline_bm25", {})
        hybrid = rag_metrics.get("hybrid", {})
        summary_rows.extend(
            [
                {
                    "实验": "BM25 检索",
                    "样本": baseline.get("queries"),
                    "指标1": f"Recall@5 {baseline.get('recall@5', 0):.3f}",
                    "指标2": f"MRR/nDCG {baseline.get('mrr', 0):.3f}/{baseline.get('ndcg@5', 0):.3f}",
                    "备注": f"均值 {baseline.get('latency_ms', {}).get('mean', 0):.1f} ms",
                },
                {
                    "实验": "BM25 + Dense + Reranker",
                    "样本": hybrid.get("queries"),
                    "指标1": f"Recall@5 {hybrid.get('recall@5', 0):.3f}",
                    "指标2": f"MRR/nDCG {hybrid.get('mrr', 0):.3f}/{hybrid.get('ndcg@5', 0):.3f}",
                    "备注": f"CPU 均值 {hybrid.get('latency_ms', {}).get('mean', 0):.1f} ms",
                },
            ]
        )
    if similar_case_metrics:
        retrieval = similar_case_metrics.get("retrieval", {})
        baseline = similar_case_metrics.get("random_patient_baseline", {})
        summary_rows.append(
            {
                "实验": "CT-RATE相似病例检索",
                "样本": similar_case_metrics.get("queries"),
                "指标1": (
                    f"Top1 Jaccard {retrieval.get('top1_jaccard', 0):.3f} "
                    f"(随机 {baseline.get('top1_jaccard', 0):.3f})"
                ),
                "指标2": f"Top5标签召回 {retrieval.get('label_recall', 0):.3f}",
                "备注": "查询仅用验证报告；验证弱标签未输入检索器；按患者去重",
            }
        )
    st.dataframe(summary_rows, hide_index=True, width="stretch")
    st.info("分类与证据参考均为 CT-RATE 数据集提供的弱标签，不能解释为临床性能。")


ctclip_metrics = load_ctclip_metrics("artifacts/evaluation/ctclip_metrics.json")
if ctclip_metrics:
    st.divider()
    st.subheader("CT-CLIP v2 验证结果")

    overview_columns = st.columns(4)
    overview_columns[0].metric("完成病例", int(ctclip_metrics["completed_cases"]))
    overview_columns[1].metric("覆盖率", f"{ctclip_metrics['coverage']:.1%}")
    overview_columns[2].metric("Micro-F1", f"{ctclip_metrics['micro_f1']:.3f}")
    overview_columns[3].metric("Macro-F1", f"{ctclip_metrics['macro_f1']:.3f}")

    quality_columns = st.columns(4)
    quality_columns[0].metric("Micro-AUROC", f"{ctclip_metrics['micro_auroc']:.3f}")
    quality_columns[1].metric("Macro-AUROC", f"{ctclip_metrics['macro_auroc']:.3f}")
    quality_columns[2].metric("Macro-AUPRC", f"{ctclip_metrics['macro_auprc']:.3f}")
    quality_columns[3].metric(
        "中位推理耗时",
        f"{ctclip_metrics['latency_ms']['median'] / 1000:.2f} 秒",
    )

    st.info(
        "评估标签来自 CT-RATE 提供的 predicted_labels（弱标签），分类阈值为 "
        f"{ctclip_metrics['threshold']:.2f}。AUROC/AUPRC 更适合观察当前模型排序能力；"
        "这些结果不能作为临床性能结论。"
    )

    per_label_rows = []
    for label, metrics in ctclip_metrics.get("per_label", {}).items():
        per_label_rows.append(
            {
                "异常类型": LABEL_ZH.get(label, label),
                "英文标签": label,
                "阳性数": metrics["positive_count"],
                "F1": round(metrics["f1"], 3),
                "AUROC": round(metrics["auroc"], 3),
                "AUPRC": round(metrics["auprc"], 3),
            }
        )
    per_label_rows.sort(key=lambda row: row["AUROC"], reverse=True)
    st.dataframe(per_label_rows, hide_index=True, width="stretch")
