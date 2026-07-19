import asyncio
import csv
from pathlib import Path

import streamlit as st

from chestct_agent.agent.graph import ChestCtAgent
from chestct_agent.schemas import AgentState, AnalyzeRequest


st.set_page_config(page_title="ChestCT-Agent", layout="wide")
st.title("ChestCT-Agent")

st.caption("Coursework/research demo only. Not for clinical diagnosis.")


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


volume_paths = sorted(Path("data/dataset/valid_fixed").rglob("*.nii.gz"))
volume_by_name = {path.name: path.resolve() for path in volume_paths}
case_options = list(volume_by_name) + ["Manual input"]
selected_case = st.selectbox("Local CT-RATE case", case_options, index=0)
selected_path = volume_by_name.get(selected_case)
reports = load_validation_reports(
    "data/dataset/radiology_text_reports/validation_reports.csv"
)

default_case_id = selected_case.removesuffix(".nii.gz") if selected_path else "demo_case"
default_report = reports.get(
    selected_case,
    "Findings: Small bilateral pleural effusions are present. Mild bibasal atelectasis. "
    "No pneumothorax. Impression: Small pleural effusions.",
)

case_id = st.text_input("Case ID", value=default_case_id)
question = st.text_input("Question", value="What abnormalities are present?")
report_text = st.text_area(
    "Radiology report",
    value=default_report,
    height=180,
)
ct_volume_path = st.text_input(
    "Optional CT volume path (.nii/.nii.gz)",
    value=str(selected_path) if selected_path else "",
)

if st.button("Analyze", type="primary"):
    request = AnalyzeRequest(
        case_id=case_id,
        report_text=report_text,
        question=question,
        ct_volume_path=ct_volume_path or None,
    )
    agent = ChestCtAgent()
    with st.spinner("Running controlled Agentic RAG workflow..."):
        response = asyncio.run(agent.run(AgentState(request=request)))

    for warning in response.warnings:
        st.warning(warning)

    if response.ct_preview_images:
        st.subheader("CT Preview")
        preview_cols = st.columns(min(5, len(response.ct_preview_images)))
        for col, image_path in zip(preview_cols, response.ct_preview_images, strict=False):
            path = Path(image_path)
            if path.exists():
                col.image(str(path), caption=path.name, use_container_width=True)

    st.subheader("Structured Output")
    st.json(response.model_dump())

    st.subheader("Labels")
    for label in response.labels:
        with st.expander(f"{label.name} | {label.status} | {label.confidence:.2f}", expanded=True):
            st.write("Source scores:", label.source_scores)
            st.write("Report evidence:", label.evidence_from_report)
            if label.evidence_from_image.preview_images:
                cols = st.columns(min(3, len(label.evidence_from_image.preview_images)))
                for col, image_path in zip(cols, label.evidence_from_image.preview_images, strict=False):
                    path = Path(image_path)
                    if path.exists():
                        col.image(str(path), caption=path.name, use_container_width=True)

    st.subheader("Similar Cases")
    st.dataframe([case.model_dump() for case in response.similar_cases])

    st.subheader("Explanation")
    st.write(response.explanation_zh)
