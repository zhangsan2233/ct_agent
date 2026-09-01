from __future__ import annotations

from base64 import b64encode
from html import escape
import json
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="ChestCT Agent 项目介绍", page_icon="🫁", layout="wide")


ROOT = Path(__file__).resolve().parents[2]
MEMORY_PATH = (
    ROOT
    / "artifacts"
    / "patchchestct_qwen_tools_100_50"
    / "tool_memory"
    / "audited_candidates.json"
)
HERO_IMAGE_PATH = ROOT / "static" / "cases" / "valid_24_a_1" / "slice_127_lung.png"

LABEL_ZH = {
    "arterial_wall_calcification": "动脉壁钙化",
    "atelectasis": "肺不张",
    "bronchiectasis": "支气管扩张",
    "consolidation": "肺实变",
    "coronary_wall_calcification": "冠状动脉壁钙化",
    "hiatal_hernia": "食管裂孔疝",
    "lung_opacity": "肺部密度增高影",
    "lymphadenopathy": "淋巴结肿大",
    "pericardial_effusion": "心包积液",
}

REVIEW_TARGETS = {
    "arterial_wall_calcification:FN:other_or_unknown": "纵隔窗主动脉壁及前后相邻层面，确认细小或断续钙化。",
    "arterial_wall_calcification:FP:other_or_unknown": "主动脉旁纵隔窗连续层面，重点区分椎体、肋骨等骨性高密度结构。",
    "atelectasis:FN:other_or_unknown": "线性影所在肺叶的连续肺窗层面，复核容积缩小、血管聚拢等征象。",
    "atelectasis:FP:other_or_unknown": "单侧高密度区域、纵隔位置及残余含气肺组织，区分肺不张与胸腔积液。",
    "bronchiectasis:FN:other_or_unknown": "双肺上叶和下叶的连续肺窗层面，检查支气管管径及管壁。",
    "bronchiectasis:FP:other_or_unknown": "可疑气道前后相邻肺窗层面，区分真实支气管扩张与实变内空气支气管征。",
    "consolidation:FN:other_or_unknown": "高密度影所在区域的连续肺窗层面，重点寻找空气支气管征。",
    "consolidation:FN:visual_misinterpretation": "右下肺可疑密度增高区域及其前后相邻肺窗层面。",
    "consolidation:FP:other_or_unknown": "单侧高密度区域的连续肺窗层面，同时观察肺容积和纵隔移位。",
    "coronary_wall_calcification:FN:other_or_unknown": "包含心脏的中胸部纵隔窗连续层面，沿冠状动脉走行复核。",
    "coronary_wall_calcification:FP:other_or_unknown": "心脏周围纵隔窗连续层面，区分冠状动脉钙化与主动脉钙化。",
    "hiatal_hernia:FN:other_or_unknown": "膈肌、食管裂孔和胃泡附近的连续纵隔窗层面。",
    "hiatal_hernia:FP:other_or_unknown": "膈肌上下相邻层面，确认胃泡位置和膈肌连续性。",
    "lung_opacity:FN:other_or_unknown": "全肺连续肺窗，重点复核双下肺及磨玻璃样密度改变。",
    "lung_opacity:FP:other_or_unknown": "可疑密度增高区域的连续肺窗，确认是否仍存在正常含气肺组织。",
    "lymphadenopathy:FN:other_or_unknown": "纵隔、双侧肺门和隆突下区域的连续纵隔窗层面。",
    "pericardial_effusion:FN:other_or_unknown": "心脏中下段纵隔窗连续层面，沿心包腔复核液性密度。",
}

REJECTION_ZH = {
    "source_reference_leakage": "来源经验含有事后参考标签痕迹，存在答案泄漏风险。",
    "fewer_than_3_independent_supporting_cases": "独立支持病例少于3例，尚不足以作为稳定经验。",
    "reference_leakage": "经验包含测试时不可获得的参考答案信息。",
    "single_case_candidate": "目前仅由单个病例支持。",
    "not_an_error": "审计后未确认这是一个真实错误。",
}

PROCESS_STEPS = [
    ("01", "接收检查", "读取3D NIfTI胸部CT和可选影像报告，建立本次病例上下文。"),
    ("02", "构造视觉输入", "完成HU窗宽窗位、肺窗/纵隔窗渲染和全胸部粗筛切片采样。"),
    ("03", "发现候选", "CT-CLIP给出疾病分数，Qwen3-VL直接查看切片并形成独立视觉判断。"),
    ("04", "针对性复核", "Agent按候选疾病追加解剖区域连续切片，并调用分割、定位、报告和检索工具。"),
    ("05", "证据融合", "Qwen3.6比较CT、视觉、报告、RAG和工具证据，处理冲突并形成结构化标签。"),
    ("06", "Memory复查", "按疾病和FP/FN类型检索经验，要求当前影像证据满足门控后才允许修改判断。"),
    ("07", "输出与反馈", "生成中文报告、图像证据和可审计轨迹，并接收医生逐标签反馈与图像框选。"),
]

TOOL_GROUPS = [
    (
        "影像理解",
        [
            {
                "name": "ct_preprocess_tool",
                "engine": "NiBabel + NumPy",
                "io": "NIfTI → HU体积、肺窗/纵隔窗、轴位切片",
                "purpose": "把完整3D CT转换成后续模型可读取且能回到原始层号的视觉输入。",
            },
            {
                "name": "ct_classifier_tool",
                "engine": "CT-CLIP v2 zero-shot",
                "io": "3D CT → 18类疾病分数",
                "purpose": "完成全胸部粗筛并提出候选疾病，不单独作为最终诊断。",
            },
            {
                "name": "qwen_slice_vqa_tool",
                "engine": "Qwen3-VL-30B-A3B-Instruct",
                "io": "关键切片 + 疾病问题 → 视觉判断与可见证据",
                "purpose": "让多模态模型真正查看肺窗和纵隔窗，而不是只读取CT-CLIP分数。",
            },
            {
                "name": "anatomy_slice_router_tool",
                "engine": "解剖区域规则 + 连续切片提取",
                "io": "候选疾病 → 对应解剖区域连续层面",
                "purpose": "针对淋巴结、食管裂孔、心包等区域追加切片，降低均匀采样漏诊。",
            },
            {
                "name": "ct_attribution_tool",
                "engine": "CT-CLIP Gradient × Token",
                "io": "疾病logit + 24³图像token → 3D归因体积与叠加图",
                "purpose": "解释哪些CT区域推动了CT-CLIP分数；归因图不等同于病灶分割。",
            },
            {
                "name": "organ_segmentation_tool",
                "engine": "TotalSegmentator / RadGenome mask",
                "io": "3D CT + 器官名 → 器官mask、体积和层面范围",
                "purpose": "提供可靠的解剖范围，帮助切片导航、区域测量和病灶约束。",
            },
            {
                "name": "lesion_grounding_tool",
                "engine": "mask、候选框与切片映射",
                "io": "异常候选 + 空间证据 → slice、bbox或mask",
                "purpose": "将检查级结论落到真实层面和区域，供医生复核。",
            },
        ],
    ),
    (
        "报告与知识",
        [
            {
                "name": "report_parser_tool",
                "engine": "结构化报告解析",
                "io": "报告文本 → Findings、Impression和证据句",
                "purpose": "把报告拆成可与影像逐标签对齐的文本证据。",
            },
            {
                "name": "report_graph_tool",
                "engine": "Modern RadGraph-XL",
                "io": "报告文本 → 实体、否定、位置和关系图",
                "purpose": "识别报告中的异常、解剖位置和否定关系，减少关键词误判。",
            },
            {
                "name": "medical_rag_tool",
                "engine": "BM25 + Qwen3-Embedding + Qwen3-Reranker + Qdrant",
                "io": "医学问题 → 排序后的知识证据",
                "purpose": "检索疾病定义、影像表现和鉴别信息，为解释提供外部知识。",
            },
            {
                "name": "similar_case_retriever_tool",
                "engine": "CT-RATE病例索引",
                "io": "病例表征与标签 → 非当前病例Top-K",
                "purpose": "提供相似病例参考，同时排除当前病例自身，避免检索泄漏。",
            },
        ],
    ),
    (
        "Agent决策与反馈",
        [
            {
                "name": "agent_planner",
                "engine": "Qwen3.6-35B-A3B + LangGraph",
                "io": "病例状态与任务 → 工具计划和后续节点",
                "purpose": "决定本轮需要分类、视觉复核、检索、分割还是报告工具。",
            },
            {
                "name": "stage2_fusion_tool",
                "engine": "微调Qwen3.5 Stage-2八类复核模型",
                "io": "CT分数、报告和证据 → 8类复核建议",
                "purpose": "作为独立的专项证据融合工具，不替代通用Agent规划器。",
            },
            {
                "name": "consistency_checker_tool",
                "engine": "逐标签冲突规则",
                "io": "CT、报告、视觉与检索证据 → 冲突和复核项",
                "purpose": "阻止单一工具的高分直接覆盖相互矛盾的证据。",
            },
            {
                "name": "experience_memory_tool",
                "engine": "PatchChestCT错误轨迹 + DeepSeek经验库",
                "io": "疾病、初始状态和当前证据 → 匹配经验与复查指令",
                "purpose": "让Agent针对已知FP/FN模式重新查看当前病例，而不是背诵历史答案。",
            },
            {
                "name": "json_validator_tool",
                "engine": "Pydantic结构校验",
                "io": "Agent草稿 → 合法18类JSON",
                "purpose": "保证标签、置信度、证据和报告字段满足固定接口。",
            },
            {
                "name": "human_feedback_tool",
                "engine": "SQLite反馈队列 + CT框选",
                "io": "医生纠错与bbox → 待审核反馈和Memory候选",
                "purpose": "保存事实性反馈，审核后用于Memory更新和后续训练数据构造。",
            },
        ],
    ),
]


st.markdown(
    """
    <style>
    :root {
        --mem-bg: #050506;
        --mem-surface: #111113;
        --mem-surface-2: #18181b;
        --mem-line: #343438;
        --mem-text: #f5f5f7;
        --mem-muted: #a1a1a6;
        --mem-blue: #2997ff;
        --mem-fp: #ff453a;
        --mem-fn: #64d2ff;
        --mem-green: #30d158;
        --mem-amber: #ff9f0a;
    }

    html, body, [class*="css"] {
        font-family: "SF Pro Text", "SF Pro Display", -apple-system,
            BlinkMacSystemFont, "PingFang SC", "Helvetica Neue", "Segoe UI",
            "Microsoft YaHei", sans-serif;
        letter-spacing: 0;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--mem-bg);
        color: var(--mem-text);
    }

    [data-testid="stHeader"] {
        background: rgba(5, 5, 6, 0.76);
        border-bottom: 1px solid rgba(52, 52, 56, 0.72);
        backdrop-filter: blur(22px) saturate(150%);
    }

    [data-testid="stSidebar"] {
        background: #09090a;
        border-right: 1px solid var(--mem-line);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1460px;
        padding-top: 1.6rem;
        padding-bottom: 5rem;
    }

    .project-hero {
        position: relative;
        min-height: 500px;
        display: flex;
        align-items: center;
        margin: 0 -1rem 3.8rem;
        padding: 4rem 3rem;
        overflow: hidden;
        border-bottom: 1px solid var(--mem-line);
        isolation: isolate;
    }

    .project-hero::before {
        content: "";
        position: absolute;
        inset: 0 0 0 42%;
        background-image: var(--hero-image);
        background-repeat: no-repeat;
        background-position: center;
        background-size: cover;
        opacity: 0.9;
        z-index: -2;
    }

    .project-hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, #050506 0%, #050506 34%, rgba(5, 5, 6, 0.78) 53%, rgba(5, 5, 6, 0.14) 100%);
        z-index: -1;
    }

    .hero-copy { max-width: 700px; }

    .hero-eyebrow {
        color: var(--mem-blue);
        font-size: 0.9rem;
        font-weight: 720;
        margin-bottom: 1rem;
    }

    .hero-title {
        color: var(--mem-text);
        font-size: clamp(3.4rem, 7vw, 7rem);
        line-height: 0.95;
        font-weight: 740;
        margin: 0;
    }

    .hero-description {
        color: #c7c7cc;
        font-size: 1.24rem;
        line-height: 1.7;
        max-width: 650px;
        margin: 1.5rem 0 0;
    }

    .section-heading {
        display: grid;
        grid-template-columns: minmax(190px, 0.34fr) minmax(0, 0.66fr);
        gap: 2.5rem;
        align-items: end;
        padding: 2.5rem 0 1.6rem;
        border-top: 1px solid var(--mem-line);
        margin-top: 2.5rem;
    }

    .section-heading h2 {
        color: var(--mem-text);
        font-size: clamp(2rem, 4vw, 3.7rem);
        line-height: 1.05;
        margin: 0;
    }

    .section-heading p {
        color: var(--mem-muted);
        font-size: 1.04rem;
        line-height: 1.68;
        margin: 0;
        max-width: 760px;
    }

    .process-list {
        border-top: 1px solid var(--mem-line);
        margin-bottom: 3.8rem;
    }

    .process-row {
        display: grid;
        grid-template-columns: 72px minmax(170px, 0.28fr) minmax(0, 0.72fr);
        gap: 1.3rem;
        align-items: baseline;
        padding: 1.25rem 0;
        border-bottom: 1px solid var(--mem-line);
    }

    .process-number {
        color: var(--mem-blue);
        font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
        font-size: 0.82rem;
    }

    .process-name {
        color: var(--mem-text);
        font-size: 1.1rem;
        font-weight: 680;
    }

    .process-description {
        color: #b8b8bd;
        font-size: 0.96rem;
        line-height: 1.62;
    }

    .tool-group-title {
        color: var(--mem-text);
        font-size: 1.36rem;
        font-weight: 680;
        margin: 2rem 0 0.9rem;
    }

    .tool-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1px;
        background: var(--mem-line);
        border: 1px solid var(--mem-line);
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 1.8rem;
    }

    .tool-item {
        background: #0d0d0f;
        padding: 1.3rem;
        min-height: 194px;
    }

    .tool-name {
        color: #ffffff;
        font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
        font-size: 0.98rem;
        font-weight: 680;
        overflow-wrap: anywhere;
    }

    .tool-engine {
        color: var(--mem-blue);
        font-size: 0.84rem;
        margin: 0.3rem 0 0.9rem;
    }

    .tool-io {
        color: #d5d5da;
        font-size: 0.88rem;
        line-height: 1.52;
        margin-bottom: 0.55rem;
    }

    .tool-purpose {
        color: var(--mem-muted);
        font-size: 0.88rem;
        line-height: 1.58;
    }

    .memory-build {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        border-top: 1px solid var(--mem-line);
        border-bottom: 1px solid var(--mem-line);
        margin: 0 0 2.4rem;
    }

    .memory-build-step {
        position: relative;
        padding: 1.25rem 1.1rem 1.45rem;
        border-right: 1px solid var(--mem-line);
    }

    .memory-build-step:last-child { border-right: 0; }

    .memory-build-index {
        color: var(--mem-blue);
        font-size: 0.75rem;
        margin-bottom: 0.55rem;
    }

    .memory-build-title {
        color: var(--mem-text);
        font-size: 1rem;
        font-weight: 680;
        margin-bottom: 0.45rem;
    }

    .memory-build-text {
        color: var(--mem-muted);
        font-size: 0.86rem;
        line-height: 1.56;
    }

    .memory-source-note {
        color: #d8d8dd;
        font-size: 0.98rem;
        line-height: 1.68;
        padding: 1.15rem 1.25rem;
        background: #101820;
        border-left: 3px solid var(--mem-blue);
        margin: 0 0 3.6rem;
    }

    .memory-kicker {
        color: var(--mem-blue);
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .memory-title {
        color: var(--mem-text);
        font-size: clamp(2.6rem, 5vw, 5.2rem);
        line-height: 1.02;
        font-weight: 720;
        margin: 0;
    }

    .memory-subtitle {
        color: var(--mem-muted);
        font-size: 1.15rem;
        line-height: 1.7;
        max-width: 820px;
        margin: 1.25rem 0 2.4rem;
    }

    .memory-statline {
        display: flex;
        flex-wrap: wrap;
        gap: 1.8rem;
        padding: 1.25rem 0 2rem;
        border-top: 1px solid var(--mem-line);
        border-bottom: 1px solid var(--mem-line);
        margin-bottom: 2rem;
    }

    .memory-stat strong {
        display: block;
        color: var(--mem-text);
        font-size: 1.8rem;
        line-height: 1.1;
    }

    .memory-stat span {
        color: var(--mem-muted);
        font-size: 0.86rem;
    }

    .memory-card {
        height: 100%;
        background: linear-gradient(180deg, var(--mem-surface-2), var(--mem-surface));
        border: 1px solid var(--mem-line);
        border-radius: 8px;
        padding: 1.35rem 1.35rem 1.15rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.22);
    }

    .memory-card-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        border-bottom: 1px solid var(--mem-line);
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }

    .memory-index {
        color: var(--mem-muted);
        font-size: 0.78rem;
        font-variant-numeric: tabular-nums;
        margin-bottom: 0.28rem;
    }

    .memory-disease {
        color: var(--mem-text);
        font-size: 1.45rem;
        line-height: 1.2;
        font-weight: 680;
    }

    .memory-type {
        min-width: 2.9rem;
        text-align: center;
        border-radius: 999px;
        padding: 0.28rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 760;
    }

    .memory-type.fp {
        color: #ffd5d2;
        background: rgba(255, 69, 58, 0.18);
        border: 1px solid rgba(255, 69, 58, 0.45);
    }

    .memory-type.fn {
        color: #d5f5ff;
        background: rgba(100, 210, 255, 0.15);
        border: 1px solid rgba(100, 210, 255, 0.42);
    }

    .memory-field {
        margin: 0 0 1rem;
    }

    .memory-label {
        color: var(--mem-muted);
        font-size: 0.76rem;
        font-weight: 650;
        margin-bottom: 0.3rem;
    }

    .memory-value {
        color: #e8e8ed;
        font-size: 0.95rem;
        line-height: 1.62;
    }

    .memory-lesson {
        color: #ffffff;
        font-size: 1.03rem;
        line-height: 1.68;
        padding-left: 0.9rem;
        border-left: 3px solid var(--mem-blue);
    }

    .case-chip {
        display: inline-block;
        color: #d7d7dc;
        background: #222225;
        border: 1px solid #3d3d42;
        border-radius: 999px;
        padding: 0.2rem 0.52rem;
        margin: 0.12rem 0.2rem 0.12rem 0;
        font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
        font-size: 0.76rem;
    }

    .memory-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding-top: 0.95rem;
        border-top: 1px solid var(--mem-line);
    }

    .retrieval-yes, .retrieval-no {
        font-size: 0.82rem;
        font-weight: 700;
    }

    .retrieval-yes { color: var(--mem-green); }
    .retrieval-no { color: var(--mem-amber); }

    .support-count {
        color: var(--mem-muted);
        font-size: 0.82rem;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label {
        color: #c7c7cc !important;
        font-weight: 600;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input {
        background: #151517 !important;
        border-color: #3a3a3e !important;
        color: #f5f5f7 !important;
    }

    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding-top: 2rem; }
        .project-hero { min-height: 560px; padding: 3rem 1.2rem; margin-left: -0.5rem; margin-right: -0.5rem; }
        .project-hero::before { inset: 38% 0 0 0; opacity: 0.58; }
        .project-hero::after { background: linear-gradient(180deg, #050506 0%, #050506 44%, rgba(5, 5, 6, 0.55) 70%, #050506 100%); }
        .hero-title { font-size: 3.55rem; }
        .hero-description { font-size: 1rem; }
        .section-heading { grid-template-columns: 1fr; gap: 0.8rem; }
        .process-row { grid-template-columns: 44px 1fr; }
        .process-description { grid-column: 2; }
        .tool-grid { grid-template-columns: 1fr; }
        .memory-build { grid-template-columns: 1fr; }
        .memory-build-step { border-right: 0; border-bottom: 1px solid var(--mem-line); }
        .memory-build-step:last-child { border-bottom: 0; }
        .memory-title { font-size: 2.65rem; }
        .memory-subtitle { font-size: 1rem; }
        .memory-card { padding: 1.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_memories(path: str, modified_ns: int) -> list[dict]:
    del modified_ns
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("memory_group_audits", []))


def error_type(item: dict) -> str:
    parts = str(item.get("group_id", "")).split(":")
    return parts[1] if len(parts) > 1 and parts[1] in {"FP", "FN"} else "未知"


def trigger_text(item: dict) -> str:
    disease = LABEL_ZH.get(str(item.get("label", "")), str(item.get("label", "未知疾病")))
    kind = error_type(item)
    if kind == "FP":
        return f"初始判断将“{disease}”判为阳性时，触发假阳性复核，检查是否由相似结构、其他病变或采样偏差造成。"
    if kind == "FN":
        return f"初始判断将“{disease}”判为阴性时，触发漏诊复核，主动检查粗筛可能遗漏的区域和连续层面。"
    return "该疾病的初始判断需要再次复核。"


def rejection_text(item: dict) -> str:
    if item.get("candidate_usable_for_retrieval"):
        return "已满足当前检索条件，无拦截原因。"
    reasons = item.get("rejection_reasons") or ["未通过当前Memory审计门槛"]
    translated = [REJECTION_ZH.get(str(reason), str(reason)) for reason in reasons]
    return " ".join(translated)


def render_memory_card(item: dict, index: int) -> str:
    group_id = str(item.get("group_id", ""))
    kind = error_type(item)
    disease = LABEL_ZH.get(str(item.get("label", "")), str(item.get("label", "未知疾病")))
    lesson = str(item.get("memory_summary_text", "")).strip() or "暂无经验正文。"
    source_ids = item.get("source_case_ids") or []
    source_html = "".join(
        f'<span class="case-chip">{escape(str(case_id))}</span>' for case_id in source_ids
    ) or '<span class="memory-value">未记录</span>'
    retrievable = bool(item.get("candidate_usable_for_retrieval"))
    retrieval_class = "retrieval-yes" if retrievable else "retrieval-no"
    retrieval_text = "是，当前Agent会检索" if retrievable else "否，当前Agent不会检索"
    target = REVIEW_TARGETS.get(group_id, "重新查看可疑区域及其前后连续层面。")
    support_count = int(item.get("support_count", 0) or 0)
    return f"""
    <article class="memory-card">
        <div class="memory-card-head">
            <div>
                <div class="memory-index">MEMORY {index:02d} · {escape(group_id)}</div>
                <div class="memory-disease">{escape(disease)}</div>
            </div>
            <span class="memory-type {kind.lower()}">{escape(kind)}</span>
        </div>
        <div class="memory-field">
            <div class="memory-label">DEEPSEEK总结出的经验正文</div>
            <div class="memory-lesson">{escape(lesson)}</div>
        </div>
        <div class="memory-field">
            <div class="memory-label">触发条件</div>
            <div class="memory-value">{escape(trigger_text(item))}</div>
        </div>
        <div class="memory-field">
            <div class="memory-label">建议重新查看的区域或切片</div>
            <div class="memory-value">{escape(target)}</div>
        </div>
        <div class="memory-field">
            <div class="memory-label">来源PatchChestCT病例编号</div>
            <div>{source_html}</div>
        </div>
        <div class="memory-field">
            <div class="memory-label">未进入检索时的原因</div>
            <div class="memory-value">{escape(rejection_text(item))}</div>
        </div>
        <div class="memory-footer">
            <span class="{retrieval_class}">{retrieval_text}</span>
            <span class="support-count">支持病例数 {support_count}</span>
        </div>
    </article>
    """


def image_data_uri(path: Path) -> str:
    if not path.is_file():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64encode(path.read_bytes()).decode('ascii')}"


def render_process() -> str:
    rows = []
    for number, name, description in PROCESS_STEPS:
        rows.append(
            '<div class="process-row">'
            f'<div class="process-number">{escape(number)}</div>'
            f'<div class="process-name">{escape(name)}</div>'
            f'<div class="process-description">{escape(description)}</div>'
            "</div>"
        )
    return '<div class="process-list">' + "".join(rows) + "</div>"


def render_tool_group(title: str, tools: list[dict[str, str]]) -> str:
    items = []
    for tool in tools:
        items.append(
            '<article class="tool-item">'
            f'<div class="tool-name">{escape(tool["name"])}</div>'
            f'<div class="tool-engine">{escape(tool["engine"])}</div>'
            f'<div class="tool-io">输入输出：{escape(tool["io"])}</div>'
            f'<div class="tool-purpose">作用：{escape(tool["purpose"])}</div>'
            "</article>"
        )
    return (
        f'<div class="tool-group-title">{escape(title)}</div>'
        + '<div class="tool-grid">'
        + "".join(items)
        + "</div>"
    )


if not MEMORY_PATH.is_file():
    st.error(f"未找到Memory文件：{MEMORY_PATH}")
    st.stop()

memories = load_memories(str(MEMORY_PATH), MEMORY_PATH.stat().st_mtime_ns)

hero_uri = image_data_uri(HERO_IMAGE_PATH)
st.markdown(
    f"""
    <section class="project-hero" style="--hero-image: url('{hero_uri}')">
        <div class="hero-copy">
            <div class="hero-eyebrow">3D CHEST CT · TOOL-USING AGENT</div>
            <h1 class="hero-title">ChestCT<br>Agent</h1>
            <p class="hero-description">
                把完整胸部CT转换为可复核的疾病候选、图像证据和中文报告。
                Agent不直接替代影像模型，而是组织视觉、分类、分割、检索与经验工具，
                对每个结论进行针对性复查。
            </p>
        </div>
    </section>
    <div class="section-heading">
        <h2>项目流程</h2>
        <p>从原始3D CT开始，先粗筛再按疾病追加连续切片。工具提供独立证据，Agent负责选择、比较、复查和组织最终输出。</p>
    </div>
    {render_process()}
    <div class="section-heading">
        <h2>工具体系</h2>
        <p>每个工具只负责一个清晰职责。影像工具回答“看到了什么”，知识工具补充“如何解释”，Agent工具决定“下一步查什么以及证据如何合并”。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

for group_title, group_tools in TOOL_GROUPS:
    st.markdown(render_tool_group(group_title, group_tools), unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-heading">
        <h2>Memory如何生成</h2>
        <p>Memory不是DeepSeek凭空编写的医学答案。正确监督来自PatchChestCT人工标注；DeepSeek负责阅读Qwen轨迹、对照错误，并把多例重复错误压缩成可执行的复查经验。</p>
    </div>
    <div class="memory-build">
        <div class="memory-build-step">
            <div class="memory-build-index">STEP 01</div>
            <div class="memory-build-title">Qwen盲测</div>
            <div class="memory-build-text">只给Qwen无标注CT切片，保存逐标签判断、置信度、证据和工具轨迹。</div>
        </div>
        <div class="memory-build-step">
            <div class="memory-build-index">STEP 02</div>
            <div class="memory-build-title">人工答案对照</div>
            <div class="memory-build-text">使用PatchChestCT人工标签和Patch区域确认FP与FN，筛出真实错误。</div>
        </div>
        <div class="memory-build-step">
            <div class="memory-build-index">STEP 03</div>
            <div class="memory-build-title">DeepSeek复盘</div>
            <div class="memory-build-text">DeepSeek-V4-Flash-Vision-Exp查看盲测图、人工反馈图和Qwen公开轨迹，分析错误原因。</div>
        </div>
        <div class="memory-build-step">
            <div class="memory-build-index">STEP 04</div>
            <div class="memory-build-title">归并与审计</div>
            <div class="memory-build-text">按疾病、FP/FN和错误模式合并同类经验，并检查支持病例数与答案泄漏。</div>
        </div>
        <div class="memory-build-step">
            <div class="memory-build-index">STEP 05</div>
            <div class="memory-build-title">测试时复查</div>
            <div class="memory-build-text">新病例命中经验后重新查看指定区域；只有当前影像证据通过门控才修改答案。</div>
        </div>
    </div>
    <div class="memory-source-note">
        当前产物由48条Qwen错误轨迹归并为17组Memory。DeepSeek生成经验文本，PatchChestCT人工标注提供对错依据；Memory保存的是复查方法，不保存可直接套用的病例答案。
    </div>
    <div class="memory-kicker">Audited experience memory</div>
    <h2 class="memory-title">17组 Memory</h2>
    <p class="memory-subtitle">
        DeepSeek依据Qwen错误轨迹和PatchChestCT人工反馈总结的17组复查经验。
        全部候选均在此展示，是否进入当前Agent检索由独立审计门槛决定。
    </p>
    """,
    unsafe_allow_html=True,
)

retrievable_count = sum(bool(item.get("candidate_usable_for_retrieval")) for item in memories)
disease_count = len({str(item.get("label", "")) for item in memories})
source_count = len(
    {
        str(case_id)
        for item in memories
        for case_id in (item.get("source_case_ids") or [])
    }
)
st.markdown(
    f"""
    <div class="memory-statline">
        <div class="memory-stat"><strong>{len(memories)}</strong><span>全部Memory组</span></div>
        <div class="memory-stat"><strong>{disease_count}</strong><span>覆盖疾病</span></div>
        <div class="memory-stat"><strong>{source_count}</strong><span>来源病例</span></div>
        <div class="memory-stat"><strong>{retrievable_count}</strong><span>当前进入Agent检索</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

filter_columns = st.columns([1.25, 1, 2.4])
disease_options = ["全部疾病"] + sorted(
    {LABEL_ZH.get(str(item.get("label", "")), str(item.get("label", ""))) for item in memories}
)
with filter_columns[0]:
    selected_disease = st.selectbox("中文疾病名称", disease_options)
with filter_columns[1]:
    selected_error = st.selectbox("错误类型", ["全部", "FP", "FN"])
with filter_columns[2]:
    query = st.text_input("搜索Memory正文或病例编号", placeholder="例如：纵隔、连续切片、train_10187_a_1")

query_normalized = query.strip().lower()
filtered = []
for item in memories:
    disease = LABEL_ZH.get(str(item.get("label", "")), str(item.get("label", "")))
    if selected_disease != "全部疾病" and disease != selected_disease:
        continue
    if selected_error != "全部" and error_type(item) != selected_error:
        continue
    searchable = " ".join(
        [
            disease,
            str(item.get("group_id", "")),
            str(item.get("memory_summary_text", "")),
            " ".join(str(value) for value in item.get("source_case_ids") or []),
            REVIEW_TARGETS.get(str(item.get("group_id", "")), ""),
        ]
    ).lower()
    if query_normalized and query_normalized not in searchable:
        continue
    filtered.append(item)

st.caption(f"当前显示 {len(filtered)} / {len(memories)} 组Memory")

if not filtered:
    st.info("没有符合当前筛选条件的Memory。")
else:
    columns = st.columns(2, gap="large")
    for position, item in enumerate(filtered, start=1):
        with columns[(position - 1) % 2]:
            st.markdown(render_memory_card(item, memories.index(item) + 1), unsafe_allow_html=True)
