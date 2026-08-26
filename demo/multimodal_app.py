"""Multi-modality defence UI: chest CT production + chest X-ray schematic full pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.modalities import (
    ModalityNotReady,
    analyze_study,
    build_stage2_agent,
    list_modalities,
    write_placeholder_cxr,
)
from demo.feedback_panel import render_feedback_panel

st.set_page_config(page_title="Chest Imaging Agent 多模态接口", page_icon="🫁", layout="wide")
st.title("胸部影像 Agent：多模态接口")
st.warning("仅用于课程答辩示意，不构成临床诊断。CT 与 CXR 共用 8 标签 JSON、中文报告与纠错闭环。")

SPECS = {item["id"]: item for item in list_modalities()}
OPTIONS = [
    (spec["id"], f"{spec['title_zh']} · {spec['status']} · {spec['dimensionality'].upper()}")
    for spec in list_modalities()
]

with st.sidebar:
    st.header("运行设置")
    runs_dir = Path(st.text_input("结果目录", value=str(ROOT / "artifacts" / "modality_runs")))
    memory_db = Path(st.text_input("审计库", value=str(ROOT / "artifacts" / "memory" / "modality_demo.sqlite3")))
    session_id = st.text_input("session_id", value="multimodal-demo")
    device = st.text_input("CUDA device", value="cuda:0")
    st.caption("同一套 8 标签合同；按 modality 路由编码器与平行 adapter。")
    st.json(list_modalities())

choice_label = st.selectbox("选择检查模态", OPTIONS, format_func=lambda item: item[1], index=1)
modality = choice_label[0]
spec = SPECS[modality]
st.info(spec["note_zh"])

case_id = st.text_input("病例 ID", value="modality_demo")
left, right = st.columns(2)
with left:
    if spec["dimensionality"] == "2d":
        uploaded = st.file_uploader("上传胸部 X 光（PNG/JPEG）", type=["png", "jpg", "jpeg", "webp", "bmp"])
        local_path = st.text_input("或填写服务器图片路径", value="")
        use_placeholder = st.checkbox("接口烟测：合成占位图", value=False)
    else:
        uploaded = st.file_uploader("上传三维体积（.nii / .nii.gz）", type=["nii", "gz"])
        local_path = st.text_input("或填写服务器 NIfTI 路径", value="")
        use_placeholder = False
with right:
    report_text = st.text_area(
        "影像报告（必填，与 CT Stage-2 一致）",
        height=220,
        placeholder="示例：There is a pulmonary nodule in the right lung. No emphysema.",
    )

button_label = {
    "production": "运行胸部 CT 正式链路（Stage-2）",
    "schematic": "运行胸部 X 光全链路（编码器 + CXR adapter）",
    "interface_only": "查询该模态（应返回未就绪）",
}[spec["status"]]


def materialize_image() -> Path | None:
    if use_placeholder:
        target = runs_dir / "uploads" / f"{int(time.time())}_{case_id}_placeholder.png"
        return write_placeholder_cxr(target)
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or (".nii.gz" if uploaded.name.endswith(".nii.gz") else "")
        target = runs_dir / "uploads" / f"{int(time.time())}_{case_id}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(uploaded.getvalue())
        return target
    if local_path.strip():
        return Path(local_path.strip())
    return None


if st.button(button_label, type="primary", use_container_width=True):
    image_path = materialize_image()
    run_dir = runs_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{case_id}"
    stage2_agent = None
    if spec["status"] != "interface_only":
        agent = build_stage2_agent(ROOT, modality, device=device)
        errors = agent.readiness_errors()
        if errors:
            st.error("资产未就绪：\n\n" + "\n\n".join(errors))
            if modality == "cxr_chest":
                st.info("可先运行：python scripts/init_cxr_adapter.py && pip install torchxrayvision")
            st.stop()
        stage2_agent = agent
    try:
        with st.spinner("编码器 → Stage-2 JSON → 中文报告..."):
            result = analyze_study(
                modality=modality,
                case_id=case_id.strip() or "modality_demo",
                image_path=image_path,
                report_text=report_text,
                run_dir=run_dir,
                stage2_agent=stage2_agent,
                root=ROOT,
                device=device,
            )
    except ModalityNotReady as exc:
        st.error(f"接口已注册，后端未就绪：{exc}")
    except Exception as exc:
        st.error(f"运行失败：{type(exc).__name__}: {exc}")
    else:
        st.session_state["last_result"] = result
        st.session_state["last_modality"] = modality
        st.success(f"完成，用时 {result.get('elapsed_seconds', 0):.2f} 秒")
        st.caption(result.get("warning", ""))
        if result.get("image_qc") and image_path is not None:
            st.image(str(image_path), caption="预览", use_container_width=True)
        if result.get("ctclip_scores"):
            st.subheader("影像证据分数（映射到 Stage-2 八标签）")
            st.bar_chart(result["ctclip_scores"])
        if result.get("validation", {}).get("schema_valid"):
            st.success("JSON schema 与证据回显校验通过。")
        else:
            st.error("校验未通过：" + str(result.get("validation", {}).get("errors")))
        if result.get("stage2_json"):
            st.subheader("Stage-2 JSON")
            st.json(result["stage2_json"])
        if result.get("report_zh"):
            st.subheader("中文完整报告")
            st.markdown(result["report_zh"])
        st.download_button(
            "下载 result.json",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name=f"{case_id}_{modality}_result.json",
            mime="application/json",
        )

if "last_result" in st.session_state:
    render_feedback_panel(
        st.session_state["last_result"],
        st.session_state.get("last_modality", modality),
        session_id,
        memory_db,
    )
