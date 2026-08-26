"""Shared eight-label feedback panel for Stage-2 Streamlit demos."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from chestct_agent.config import Settings
from chestct_agent.feedback import FeedbackItem, FeedbackSubmission
from chestct_agent.memory import AgentMemory
from chestct_agent.modality_memory import stage2_result_to_request, stage2_result_to_response
from chestct_agent.modality_paths import model_version_tag
from chestct_agent.stage2_pipeline import LABELS


def record_result_to_memory(
    result: dict[str, Any],
    modality: str,
    session_id: str,
    memory_db: Path,
) -> None:
    memory = AgentMemory(Settings(memory_db_path=memory_db))
    request = stage2_result_to_request(result, modality, session_id)
    response = stage2_result_to_response(result, modality)
    memory.record(request, response, plan=None)


def render_feedback_panel(
    result: dict[str, Any],
    modality: str,
    session_id: str,
    memory_db: Path,
) -> None:
    st.subheader("纠错与反馈闭环")
    st.caption(
        "提交后进入 pending 队列；仅 approved 反馈可进入候选 SFT。CT 与 CXR 按 model_version 中的模态隔离。"
    )
    adapter_dir = Path((result.get("provenance") or {}).get("adapter_dir", "adapter"))
    model_version = model_version_tag(modality, adapter_dir)
    case_id = result.get("input", {}).get("case_id", "unknown")
    stage2_json = result.get("stage2_json") or {}
    labels = {
        item.get("name"): item
        for item in (stage2_json.get("labels") or [])
        if isinstance(item, dict)
    }
    reviewer = st.text_input("复核者", value="demo-reviewer", key=f"fb_reviewer_{modality}")
    rows = []
    for name in LABELS:
        current = labels.get(name, {})
        rows.append(
            {
                "label": name,
                "当前状态": current.get("status", "uncertain"),
                "纠正为": current.get("status", "uncertain"),
                "理由": "",
            }
        )
    edited = st.data_editor(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "label": st.column_config.TextColumn(disabled=True),
            "当前状态": st.column_config.TextColumn(disabled=True),
            "纠正为": st.column_config.SelectboxColumn(options=["positive", "negative", "uncertain"]),
            "理由": st.column_config.TextColumn(),
        },
        key=f"fb_editor_{modality}_{case_id}",
    )
    if st.button("提交反馈到 pending 队列", key=f"fb_submit_{modality}_{case_id}"):
        if not reviewer.strip():
            st.error("必须填写复核者。")
            return
        record_result_to_memory(result, modality, session_id, memory_db)
        memory = AgentMemory(Settings(memory_db_path=memory_db))
        items = [
            FeedbackItem(
                label=row["label"],
                corrected_status=row["纠正为"],
                reason=str(row.get("理由", "")).strip(),
            )
            for row in edited
            if row["纠正为"] != row["当前状态"]
        ]
        if not items:
            st.warning("没有标签状态变更，未提交反馈。")
            return
        try:
            events = memory.submit_feedback(
                case_id,
                FeedbackSubmission(
                    session_id=session_id,
                    reviewer=reviewer.strip(),
                    reviewer_role="clinician",
                    model_version=model_version,
                    items=items,
                ),
            )
        except (LookupError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.success(f"已提交 {len(events)} 条 pending 反馈。model_version={model_version}")
