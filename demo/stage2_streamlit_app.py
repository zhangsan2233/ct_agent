"""Defence UI for the frozen CT-CLIP + Stage-2 agent.

Launch from the project root on the GPU server:
CUDA_VISIBLE_DEVICES=1 streamlit run demo/stage2_streamlit_app.py --server.address 0.0.0.0
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.stage2_pipeline import LABELS, Stage2Agent, Stage2Paths
from demo.feedback_panel import render_feedback_panel


st.set_page_config(page_title="ChestCT-Agent Stage-2", page_icon="🫁", layout="wide")
st.title("ChestCT-Agent：CT-CLIP + Stage-2 演示")
st.warning("仅用于课程与科研答辩演示，不构成临床诊断；所有结果均需人工复核。")


def default_paths() -> Stage2Paths:
    return Stage2Paths.defaults(ROOT)


def render_llm_2d_review(result: dict, run_dir: Path) -> None:
    review_block = result.get("llm_2d_review") or {}
    if not review_block.get("enabled"):
        return
    st.subheader("LLM 2D 对照评审（实验支路）")
    st.caption("主结果仍以 CT-CLIP + Stage-2 JSON 为准；本节为基座模型 2D 切片对照，不构成临床诊断。")
    if review_block.get("degraded"):
        st.warning(f"2D 评审降级：{review_block.get('degraded_reason', 'unknown')}")
        return
    review = review_block.get("review") or {}
    section_titles = {
        "match_summary_zh": "一、符合度",
        "prediction_standard_zh": "二、自己的预测标准",
        "reason_zh": "三、原因",
        "agreement_zh": "四、是否同意主结果",
        "reader_advice_zh": "五、对阅读者的建议",
    }
    for key, title in section_titles.items():
        if review.get(key):
            st.markdown(f"**{title}**")
            st.write(review[key])
    if review_block.get("review_incomplete"):
        st.info("评审五段中有字段由模板补全，请结合原始投票人工判断。")
    votes = review_block.get("votes") or []
    if votes:
        st.markdown("**2D 投票明细**")
        st.dataframe(
            [
                {
                    "label": item.get("name"),
                    "vote": item.get("vote"),
                    "confidence": item.get("confidence"),
                    "evidence_zh": item.get("evidence_zh"),
                }
                for item in votes
            ],
            use_container_width=True,
            hide_index=True,
        )
    slice_paths = review_block.get("slice_paths") or []
    if slice_paths:
        st.markdown("**LLM 看过的 2D 切片**")
        cols = st.columns(min(3, len(slice_paths)))
        for index, rel_path in enumerate(slice_paths):
            image_path = run_dir / rel_path
            with cols[index % len(cols)]:
                if image_path.is_file():
                    st.image(str(image_path), caption=image_path.name, use_container_width=True)
                else:
                    st.caption(rel_path)
    report_rel = review_block.get("report_markdown_path")
    if report_rel:
        report_path = run_dir / report_rel
        if report_path.is_file():
            st.download_button(
                "下载 llm_2d_review.md",
                data=report_path.read_text(encoding="utf-8"),
                file_name=f"{result['input']['case_id']}_llm_2d_review.md",
                mime="text/markdown",
            )


with st.sidebar:
    st.header("运行设置")
    device = st.text_input("CUDA device", value="cuda:0")
    runs_dir = Path(st.text_input("结果目录", value=str(ROOT / "artifacts" / "agent_runs")))
    memory_db = Path(st.text_input("审计库", value=str(ROOT / "artifacts" / "memory" / "stage2_demo.sqlite3")))
    session_id = st.text_input("session_id", value="stage2-demo")
    enable_llm_2d_review = st.checkbox(
        "实验：2D LLM 对照评审（不改主结果）",
        value=False,
        help="使用 Qwen 基座（无 Stage-2 adapter）查看 2D 切片并写五段评审。",
    )
    st.caption("推荐在启动前设置 CUDA_VISIBLE_DEVICES=1，以使用服务器空闲 GPU。")
    st.caption("多模态接口（CT 正式 / X 光示意）：`streamlit run demo/multimodal_app.py`")

paths = default_paths()
asset_errors = Stage2Agent(paths, device).readiness_errors()
if asset_errors:
    st.error("模型资产未就绪：\n\n" + "\n\n".join(asset_errors))
    st.stop()

left, right = st.columns(2)
with left:
    case_id = st.text_input("病例 ID", value="demo_case")
    uploaded_ct = st.file_uploader("上传 CT（.nii 或 .nii.gz）", type=["nii", "gz"])
    local_ct = st.text_input("或填写服务器 CT 路径", value="")
with right:
    report_text = st.text_area("放射学报告 / Impression", height=220, placeholder="粘贴英文或中文报告文本")
    uploaded_report = st.file_uploader("或上传报告文本（.txt）", type=["txt"])

if uploaded_report is not None:
    report_text = uploaded_report.getvalue().decode("utf-8", errors="replace")
    st.text_area("已读取的报告", value=report_text, height=120, disabled=True)


def materialize_ct() -> Path | None:
    if uploaded_ct is None:
        return Path(local_ct) if local_ct.strip() else None
    suffix = ".nii.gz" if uploaded_ct.name.endswith(".nii.gz") else ".nii"
    target = runs_dir / "uploads" / f"{int(time.time())}_{case_id}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(uploaded_ct.getvalue())
    return target


if st.button("运行 CT-CLIP 与 Stage-2", type="primary", use_container_width=True):
    ct_path = materialize_ct()
    if not case_id.strip() or ct_path is None or not report_text.strip():
        st.error("请同时提供病例 ID、CT 文件/路径和报告文本。")
    elif not ct_path.is_file():
        st.error(f"找不到 CT 文件：{ct_path}")
    else:
        agent = Stage2Agent(paths, device=device)
        run_dir = runs_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{case_id}"
        spinner = (
            "先运行冻结 CT-CLIP，再运行 2D 基座评审与 Stage-2；单例通常需约 3–5 分钟..."
            if enable_llm_2d_review
            else "先运行冻结 CT-CLIP，再运行 Stage-2；单例通常需约 2–3 分钟..."
        )
        try:
            with st.spinner(spinner):
                result = agent.analyze(
                    case_id=case_id.strip(),
                    ct_path=ct_path,
                    report_text=report_text,
                    run_dir=run_dir,
                    enable_llm_2d_review=enable_llm_2d_review,
                )
        except Exception as exc:
            st.error(f"运行失败：{type(exc).__name__}: {exc}")
            st.info("请确认 GPU 空闲、CT-CLIP/Qwen/adapter 路径完整，并查看服务器终端日志。")
        else:
            st.session_state["last_result"] = result
            st.success(f"完成，用时 {result['elapsed_seconds']:.1f} 秒；结果已保存到 {run_dir}")
            st.subheader("CT-CLIP 影像证据分数")
            st.bar_chart(result["ctclip_scores"])
            st.dataframe(
                [{"label": label, "ctclip_score": result["ctclip_scores"][label]} for label in LABELS],
                use_container_width=True,
                hide_index=True,
            )
            st.subheader("Stage-2 结构化 JSON")
            if result["validation"]["schema_valid"]:
                st.success("JSON 解析和八标签 Schema 校验通过。")
            else:
                st.error("JSON 输出不完整；已保存原始输出，建议重试。")
                st.json(result["validation"])
            st.json(result["stage2_json"] or {"raw_stage2_output": result["raw_stage2_output"]})
            if result.get("report_zh"):
                st.subheader("中文完整报告")
                st.markdown(result["report_zh"])
            render_llm_2d_review(result, run_dir)
            st.subheader("可追溯记录")
            st.write(result["summary_zh"])
            st.code(str(run_dir / "result.json"))
            st.download_button(
                "下载本次 result.json",
                data=json.dumps(result, ensure_ascii=False, indent=2),
                file_name=f"{case_id}_result.json",
                mime="application/json",
            )

if "last_result" in st.session_state:
    render_feedback_panel(st.session_state["last_result"], "ct_chest", session_id, memory_db)
