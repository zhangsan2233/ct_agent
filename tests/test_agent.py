from pathlib import Path
import sys

import pandas as pd
import pytest
from PIL import Image

from chestct_agent.agent.graph import ChestCtAgent
from chestct_agent.agent.planner import ToolPolicy
from chestct_agent.calibration import CalibratedScore
from chestct_agent.config import Settings
from chestct_agent.conversation import CaseConversationAgent
from chestct_agent.evaluation import patient_id_from_case_id
from chestct_agent.labels import LABEL_IDS, LABEL_SPECS
from chestct_agent.llm import LlmCallResult
from chestct_agent.schemas import (
    AnatomyMaskResult,
    AgentState,
    AnalyzeRequest,
    AnalyzeResponse,
    CorrectionRequest,
    ExecutionMetadata,
    LabelCorrection,
    LabelOutput,
    LabelPrediction,
    QwenVisualLabelReview,
    QwenVisualRegion,
)
from chestct_agent.tools.consistency_checker import (
    apply_credibility_gate,
    apply_qwen_visual_review,
    fuse_predictions,
)
from chestct_agent.tools.ct_classifier import CtClassifierTool
from chestct_agent.tools.ct_preprocess import CtPreprocessTool
from chestct_agent.tools.evidence_extractor import extract_evidence
from chestct_agent.tools.lesion_grounding import ground_findings
from chestct_agent.tools.similar_cases import SimilarCaseRetrieverTool
from chestct_agent.tools.text_classifier import TextClassifierTool
from chestct_agent.tools.visual_evidence import build_visual_evidence
from chestct_agent.tools.report_parser import parse_report


def local_settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_compatible_api_key="replace-me",
        artifact_dir=tmp_path / "artifacts",
        static_dir=tmp_path / "static",
        knowledge_dir=tmp_path / "knowledge",
        embedding_backend="bm25",
        qdrant_path=tmp_path / "qdrant",
        calibration_path=tmp_path / "missing_calibration.joblib",
        radgraph_enabled=False,
        memory_db_path=tmp_path / "memory.sqlite3",
    )


def test_qwen_visual_review_only_changes_high_confidence_ct_candidates():
    predictions = [
        LabelPrediction(name="atelectasis", status="uncertain", confidence=0.64, source="ct"),
        LabelPrediction(name="consolidation", status="positive", confidence=0.77, source="ct"),
        LabelPrediction(name="pulmonary_nodule", status="negative", confidence=0.30, source="ct"),
    ]
    reviews = [
        QwenVisualLabelReview(name="atelectasis", status="positive", confidence=0.90),
        QwenVisualLabelReview(name="consolidation", status="negative", confidence=0.90),
        QwenVisualLabelReview(name="pulmonary_nodule", status="positive", confidence=0.70),
    ]

    updated, warnings = apply_qwen_visual_review(predictions, reviews, minimum_confidence=0.85)

    assert {item.name: item.status for item in updated} == {
        "atelectasis": "positive",
        "consolidation": "uncertain",
        "pulmonary_nodule": "negative",
    }
    assert len(warnings) == 2


def test_qwen_grounding_regions_render_as_separate_heatmaps(tmp_path: Path):
    settings = local_settings(tmp_path)
    source = tmp_path / "slice_0105_paired.jpg"
    Image.new("RGB", (768, 412), (80, 80, 80)).save(source)
    review = QwenVisualLabelReview(
        name="atelectasis",
        status="positive",
        confidence=0.91,
        regions=[
            QwenVisualRegion(
                slice_index=105,
                window="lung",
                bbox_2d=[250, 300, 700, 800],
                confidence=0.88,
                description_zh="右肺下叶可疑条片影",
            )
        ],
    )

    rendered = CtPreprocessTool(settings).render_qwen_grounding_heatmaps(
        "grounding-case", [str(source)], [review]
    )

    output = Path(rendered["atelectasis"][0])
    assert output.exists()
    with Image.open(output) as heatmap:
        assert heatmap.size == (768, 412)
        assert heatmap.getpixel((180, 250)) != (80, 80, 80)


@pytest.mark.asyncio
async def test_human_feedback_corrects_result_and_persists_audit(tmp_path: Path):
    agent = ChestCtAgent(local_settings(tmp_path))
    request = AnalyzeRequest(
        case_id="correct-case",
        session_id="correct-session",
        report_text="Findings: No pleural effusion.",
    )
    response = AnalyzeResponse(
        case_id="correct-case",
        labels=[
            LabelOutput(name="pleural_effusion", status="positive", confidence=0.8),
            LabelOutput(name="pulmonary_nodule", status="negative", confidence=0.2),
        ],
        disclaimer="research only",
    )
    agent.memory.record(request, response, None)

    corrected = await agent.correct_case(
        "correct-case",
        CorrectionRequest(
            session_id="correct-session",
            reviewer="doctor-a",
            corrections=[
                LabelCorrection(
                    label="pleural_effusion",
                    corrected_status="negative",
                    reason="原始CT复核未见积液",
                ),
                LabelCorrection(
                    label="pulmonary_nodule",
                    corrected_status="negative",
                    reason="确认阴性",
                ),
            ],
        ),
    )

    effusion = next(item for item in corrected.labels if item.name == "pleural_effusion")
    assert effusion.status == "negative"
    assert effusion.original_status == "positive"
    assert effusion.decision_source == "human_correction"
    assert corrected.approval.status == "approved"
    assert corrected.correction_history[-1].reviewer == "doctor-a"
    assert corrected.execution_events[-1].node == "apply_external_correction"
    assert len(agent.memory.get_corrections("correct-session", "correct-case")) == 1
    stored = agent.memory.get_case_context("correct-session", "correct-case")
    assert stored is not None
    assert stored[1].labels[0].status == "negative"


@pytest.mark.asyncio
async def test_dataset_oracle_is_leakage_marked_and_not_clinically_approved(tmp_path: Path):
    settings = local_settings(tmp_path).model_copy(update={"data_dir": tmp_path / "data"})
    labels_path = (
        settings.data_dir
        / "dataset"
        / "multi_abnormality_labels"
        / "valid_predicted_labels.csv"
    )
    labels_path.parent.mkdir(parents=True)
    row = {"VolumeName": "valid_1_a_1.nii.gz"}
    row.update({spec.source_column: "0" for spec in LABEL_SPECS})
    nodule = next(spec for spec in LABEL_SPECS if spec.id == "pulmonary_nodule")
    row[nodule.source_column] = "1"
    pd.DataFrame([row]).to_csv(labels_path, index=False)
    ct_path = (
        settings.data_dir
        / "dataset"
        / "valid_fixed"
        / "valid_1"
        / "valid_1_a"
        / "valid_1_a_1.nii.gz"
    )
    ct_path.parent.mkdir(parents=True)
    ct_path.touch()
    agent = ChestCtAgent(settings)
    request = AnalyzeRequest(
        case_id="valid_1_a_1",
        session_id="sandbox-session",
        report_text="Findings: No focal pulmonary nodule.",
        ct_volume_path=str(ct_path),
    )
    response = AnalyzeResponse(
        case_id="valid_1_a_1",
        labels=[
            LabelOutput(name="pulmonary_nodule", status="negative", confidence=0.2),
            LabelOutput(name="pleural_effusion", status="positive", confidence=0.7),
        ],
        disclaimer="research only",
    )
    agent.memory.record(request, response, None)

    corrected = await agent.correct_case_with_dataset(
        "valid_1_a_1", "sandbox-session"
    )

    statuses = {item.name: item.status for item in corrected.labels}
    assert statuses == {"pulmonary_nodule": "positive", "pleural_effusion": "negative"}
    assert corrected.approval.status == "pending"
    assert corrected.approval.required is True
    assert corrected.correction_history[-1].source == "dataset_weak_label"
    assert any("标签泄漏" in warning for warning in corrected.warnings)


def test_followup_policy_preserves_explicit_user_intent():
    fallback = CaseConversationAgent._fallback_plan(
        "这个结果对不对？有哪些命中、漏检和误报？"
    )
    plan = CaseConversationAgent._normalize_plan(
        {
            "intent": "case_summary",
            "use_rag": False,
            "use_similar_cases": False,
        },
        fallback,
    )

    assert plan["intent"] == "result_interpretation"


@pytest.mark.asyncio
async def test_parse_input_records_ct_file_identity(tmp_path: Path):
    volume_path = tmp_path / "stored-volume.nii.gz"
    volume_path.write_bytes(b"different CT content")
    request = AnalyzeRequest(
        case_id="identity-case",
        ct_volume_path=str(volume_path),
        ct_source_name=r"C:\fakepath\patient-scan.nii.gz",
    )

    data = await ChestCtAgent(local_settings(tmp_path)).parse_input(
        AgentState(request=request).model_dump(mode="python")
    )
    state = AgentState.model_validate(data)

    assert state.ct_input_name == "patient-scan.nii.gz"
    assert state.ct_input_size_bytes == len(b"different CT content")
    assert state.ct_input_sha256 == (
        "c7b744e691b9dfae03f497207dd37c492e59e252a27ee1db2fc3706dc874cdf8"
    )


@pytest.mark.asyncio
async def test_report_only_agent_uses_conditional_route(tmp_path: Path):
    request = AnalyzeRequest(
        case_id="test_case",
        report_text="Findings: Linear atelectasis is present. No pneumothorax.",
    )
    streamed_events = []

    async def capture_event(event):
        streamed_events.append(event)

    response = await ChestCtAgent(local_settings(tmp_path)).run(
        AgentState(request=request), event_callback=capture_event
    )

    assert response.case_id == "test_case"
    assert any(label.name == "atelectasis" for label in response.labels)
    assert response.execution.input_mode == "report_only"
    assert response.execution.retrieval_attempts == 1
    assert response.execution.retrieval_sufficient is True
    assert "medical_rag_tool" in response.tool_trace
    assert "report_graph_tool" in response.tool_trace
    assert "ct_classifier_tool" not in response.tool_trace
    atelectasis = next(label for label in response.labels if label.name == "atelectasis")
    assert atelectasis.name_zh == "肺不张"
    assert atelectasis.status_zh == "阳性"
    assert "肺不张" in response.explanation_zh
    assert response.agent_plan is not None
    assert response.execution_events
    assert [event.sequence for event in response.execution_events] == list(
        range(1, len(response.execution_events) + 1)
    )
    assert response.rag_trace.query_history
    assert response.rag_trace.attempts[0].documents
    assert response.model_reasoning.generated_by == "deterministic_fallback"
    assert response.model_reasoning.steps
    running_events = [event for event in streamed_events if event.status == "running"]
    completed_events = [event for event in streamed_events if event.status != "running"]
    assert len(running_events) == len(response.execution_events)
    assert [event.model_dump() for event in completed_events] == [
        event.model_dump() for event in response.execution_events
    ]
    assert all(event.decision_summary for event in response.execution_events)
    assert all(event.decision_basis for event in response.execution_events)
    retrieval_grade = next(
        event for event in response.execution_events if event.node == "grade_retrieval"
    )
    assert any("0.08" in item for item in retrieval_grade.decision_basis)
    approval_event = next(
        event for event in response.execution_events if event.node == "human_approval"
    )
    assert "required" in approval_event.key_metrics


@pytest.mark.asyncio
async def test_ct_only_explanation_leads_with_key_findings(tmp_path: Path):
    agent = ChestCtAgent(local_settings(tmp_path))
    response = AnalyzeResponse(
        case_id="ct_case",
        labels=[
            LabelOutput(
                name="lung_opacity",
                status="positive",
                confidence=0.74,
            ),
            LabelOutput(
                name="pleural_effusion",
                status="uncertain",
                confidence=0.62,
            ),
        ],
        disclaimer="research only",
        execution=ExecutionMetadata(input_mode="ct_only"),
    )
    state = AgentState(
        request=AnalyzeRequest(case_id="ct_case", ct_volume_path="dummy.nii.gz"),
        final_response=response,
    )

    result = AgentState.model_validate(
        await agent.generate_chinese_explanation(state.model_dump(mode="python"))
    )

    assert result.final_response is not None
    assert "主要影像发现" in result.final_response.explanation_zh
    assert "模型提示肺部密度增高影" in result.final_response.explanation_zh
    assert "建议重点复核" in result.final_response.explanation_zh
    assert "最终结论以影像科医生复核为准" in result.final_response.explanation_zh
    assert "确认为阳性" not in result.final_response.explanation_zh
    assert result.final_response.model_reasoning.generated_by == "deterministic_fallback"
    assert result.final_response.model_reasoning.steps


@pytest.mark.asyncio
async def test_qwen_generates_public_result_reasoning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    agent = ChestCtAgent(local_settings(tmp_path))
    response = AnalyzeResponse(
        case_id="reasoning_case",
        labels=[
            LabelOutput(
                name="pleural_effusion",
                status="positive",
                confidence=0.81,
                source_scores={"ct_model": 0.72, "report_model": 0.9},
            )
        ],
        disclaimer="research only",
        execution=ExecutionMetadata(input_mode="report_and_ct"),
    )
    state = AgentState(
        request=AnalyzeRequest(
            case_id="reasoning_case",
            ct_volume_path="dummy.nii.gz",
            report_text="Small pleural effusion is present.",
        ),
        final_response=response,
    )

    async def fake_chat_text(
        system: str, user: str, fallback: str, max_tokens: int | None = None
    ):
        assert "public evidence-based analysis" in system
        assert "pleural_effusion" in user
        assert max_tokens == 1200
        return LlmCallResult(
            value=(
                "## 结论\n胸腔积液为阳性。\n"
                "## 结果形成过程\n1. CT与报告共同提供支持。\n"
                "## 不确定性与限制\n仍需医生复核。"
            ),
            used_remote=True,
        )

    monkeypatch.setattr(agent.qwen, "chat_text", fake_chat_text)
    result = AgentState.model_validate(
        await agent.generate_chinese_explanation(state.model_dump(mode="python"))
    )

    assert result.final_response is not None
    assert result.final_response.model_reasoning.generated_by == "qwen"
    assert result.final_response.model_reasoning.structured_steps_by == "audit_trace"
    assert "CT与报告共同提供支持" in (
        result.final_response.model_reasoning.raw_response_zh
    )


@pytest.mark.asyncio
async def test_retrieval_rewrite_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    agent = ChestCtAgent(local_settings(tmp_path))

    async def empty_retrieval(queries: list[str], top_k: int = 5):
        return [], "tfidf"

    monkeypatch.setattr(agent.medical_rag, "retrieve", empty_retrieval)
    request = AnalyzeRequest(case_id="retry_case", report_text="The lungs are clear.")
    response = await agent.run(AgentState(request=request))

    assert response.execution.retrieval_attempts == 2
    assert response.execution.retrieval_sufficient is False
    assert response.tool_trace.count("query_rewriter") == 1
    assert response.tool_trace.count("medical_rag_tool") == 2


@pytest.mark.asyncio
async def test_optional_tool_failure_is_retried_and_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    agent = ChestCtAgent(local_settings(tmp_path))
    original = agent.medical_rag.retrieve
    calls = 0

    async def flaky_retrieval(queries: list[str], top_k: int = 5):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient retrieval failure")
        return await original(queries, top_k)

    monkeypatch.setattr(agent.medical_rag, "retrieve", flaky_retrieval)
    request = AnalyzeRequest(case_id="recovery_case", report_text="A pulmonary nodule is present.")
    response = await agent.run(AgentState(request=request))

    assert response.execution.recovered_failures == 1
    assert response.execution.degraded is False
    assert "medical_rag_tool" in response.execution.failed_tools


def test_fusion_marks_cross_modal_conflict_uncertain():
    report = [
        LabelPrediction(name="atelectasis", status="positive", confidence=0.8, source="report")
    ]
    ct = [LabelPrediction(name="atelectasis", status="negative", confidence=0.1, source="ct")]

    fused, warnings = fuse_predictions(report, ct)

    assert fused[0].status == "uncertain"
    assert fused[0].confidence == pytest.approx(0.485, abs=1e-4)
    assert warnings == ["报告与 CT 结果不一致，需人工复核：肺不张（atelectasis）"]


def test_soft_cross_modal_disagreement_keeps_high_fused_positive():
    report = [
        LabelPrediction(
            name="pulmonary_nodule",
            status="positive",
            confidence=0.9995,
            source="report",
        )
    ]
    ct = [
        LabelPrediction(
            name="pulmonary_nodule",
            status="negative",
            confidence=0.342,
            source="ct",
        )
    ]

    fused, warnings = fuse_predictions(report, ct)

    assert fused[0].status == "positive"
    assert fused[0].confidence == pytest.approx(0.7036, abs=1e-4)
    assert warnings


def test_fusion_uses_online_per_label_calibration_threshold():
    class FusionCalibration:
        @staticmethod
        def calibrate(source: str, label: str, score: float) -> CalibratedScore:
            assert source == "fusion"
            assert label == "atelectasis"
            return CalibratedScore(
                probability=score,
                positive_threshold=0.8,
                uncertain_threshold=0.4,
                calibrated=True,
                version="test-calibration",
            )

    report = [
        LabelPrediction(name="atelectasis", status="positive", confidence=0.7, source="report")
    ]
    ct = [
        LabelPrediction(name="atelectasis", status="positive", confidence=0.6, source="ct")
    ]

    fused, _ = fuse_predictions(report, ct, calibration=FusionCalibration())

    assert fused[0].confidence == pytest.approx(0.655, abs=1e-4)
    assert fused[0].status == "uncertain"
    assert fused[0].calibrated is True
    assert fused[0].calibration_version == "test-calibration"


def test_credibility_gate_downgrades_unsupported_positive():
    fused = [
        LabelPrediction(
            name="lymphadenopathy", status="positive", confidence=0.78, source="fusion"
        )
    ]

    gated, warnings = apply_credibility_gate(fused, [], {}, ct_quality_degraded=False)

    assert gated[0].status == "uncertain"
    assert warnings == [
        "可信度门控：缺少阳性报告证据或可靠 CT 支持，已将淋巴结肿大降级为待复核候选。"
    ]


def test_evidence_extractor_excludes_negated_findings():
    report = (
        "A pulmonary nodule is present. "
        "Pleural effusion was not detected. "
        "Pericardial effusion was not observed."
    )

    evidence = extract_evidence(report, ["pulmonary_nodule", "pleural_effusion"])

    assert evidence["pulmonary_nodule"][0].sentence == "A pulmonary nodule is present."
    assert evidence["pulmonary_nodule"][0].polarity == "positive"
    assert evidence["pleural_effusion"][0].polarity == "negative"


def test_negation_matching_does_not_treat_nodule_as_no(tmp_path: Path):
    classifier = TextClassifierTool(local_settings(tmp_path))

    positive = classifier.predict(parse_report("A pulmonary nodule is present."))
    negative = classifier.predict(parse_report("No pulmonary nodule is present."))
    positive_by_label = {item.name: item for item in positive}
    negative_by_label = {item.name: item for item in negative}

    assert positive_by_label["pulmonary_nodule"].status == "positive"
    assert negative_by_label["pulmonary_nodule"].status == "negative"


def test_report_classifier_always_returns_canonical_18_labels(tmp_path: Path):
    classifier = TextClassifierTool(local_settings(tmp_path))
    predictions = classifier.predict(parse_report("No pleural effusion. A lung nodule is present."))

    assert tuple(item.name for item in predictions) == LABEL_IDS
    assert len(predictions) == 18


def test_patient_id_groups_reconstructions():
    assert patient_id_from_case_id("valid_12_a_1") == "valid_12"
    assert patient_id_from_case_id("valid_12_b_2.nii.gz") == "valid_12"


def test_tool_policy_selects_grounding_only_for_location_request():
    ordinary = AnalyzeRequest(case_id="ct", ct_volume_path="case.nii.gz", question="有哪些异常？")
    localized = ordinary.model_copy(update={"question": "异常位于哪些区域，有什么图像证据？"})

    assert "organ_segmentation_tool" not in ToolPolicy.fallback_tools(ordinary)
    assert "lesion_grounding_tool" not in ToolPolicy.fallback_tools(ordinary)
    assert "ct_attribution_tool" in ToolPolicy.fallback_tools(ordinary)
    assert "organ_segmentation_tool" in ToolPolicy.fallback_tools(localized)
    assert "lesion_grounding_tool" in ToolPolicy.fallback_tools(localized)


def test_rag_query_normalization_extracts_query_from_planner_objects():
    normalized = ChestCtAgent._normalize_rag_queries(
        [
            {"intent": "definition", "query": "pulmonary nodule definition"},
            {"intent": "imaging", "query": "pulmonary nodule imaging"},
            "pulmonary nodule terminology",
        ]
    )

    assert normalized == [
        "pulmonary nodule definition",
        "pulmonary nodule imaging",
        "pulmonary nodule terminology",
    ]


def test_grounding_prefers_disease_specific_radgenome_mask():
    prediction = LabelPrediction(
        name="pulmonary_nodule", status="positive", confidence=0.8, source="ct"
    )
    masks = [
        AnatomyMaskResult(
            case_id="case",
            anatomy_name="lung",
            mask_type="region",
            mask_path="lung.nii.gz",
        ),
        AnatomyMaskResult(
            case_id="case",
            anatomy_name="lung nodule",
            mask_type="anatomy",
            mask_path="lung_nodule.nii.gz",
            slice_range=[10, 20],
            bbox_3d=[1, 2, 10, 4, 5, 20],
        ),
    ]

    findings, evidence = ground_findings([prediction], masks)

    assert findings[0].grounding_type == "lesion_mask"
    assert findings[0].region == "lung nodule"
    assert evidence["pulmonary_nodule"].grounding_type == "lesion_mask"


def test_fusion_keeps_ct_only_positive_label():
    report = [
        LabelPrediction(name="atelectasis", status="negative", confidence=0.1, source="report")
    ]
    ct = [
        LabelPrediction(name="pulmonary_nodule", status="positive", confidence=0.8, source="ct")
    ]
    fused, warnings = fuse_predictions(report, ct)
    by_label = {item.name: item for item in fused}
    assert by_label["pulmonary_nodule"].status == "positive"
    assert warnings == []


def test_visual_evidence_uses_real_slice_indices():
    predictions = [
        LabelPrediction(name="atelectasis", status="positive", confidence=0.8, source="ct")
    ]
    images = [
        "static/case/slice_050_lung.png",
        "static/case/slice_125_lung.png",
        "static/case/slice_200_lung.png",
    ]

    evidence = build_visual_evidence(predictions, images)["atelectasis"]

    assert evidence.slice_range == [50, 200]
    assert evidence.localized is False
    assert "不能定位" in evidence.note


def test_similar_case_retrieval_excludes_self_and_label_score_leakage(tmp_path: Path):
    prepared_dir = tmp_path / "artifacts" / "prepared"
    prepared_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"case_id": "query", "report_text": "right lung nodule", "labels": "pulmonary_nodule"},
            {"case_id": "other", "report_text": "right lung nodule", "labels": "atelectasis"},
        ]
    ).to_csv(prepared_dir / "case_index.csv", index=False)
    retriever = SimilarCaseRetrieverTool(local_settings(tmp_path))

    results = retriever.retrieve(
        "right lung nodule",
        labels=["pulmonary_nodule"],
        top_k=5,
        query_case_id="query",
    )

    assert [item.case_id for item in results] == ["other"]
    assert results[0].matched_labels == []


def test_ct_condition_retrieval_returns_unique_patients_without_report(tmp_path: Path):
    prepared_dir = tmp_path / "artifacts" / "prepared"
    prepared_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "case_id": "train_1_a_1.nii.gz",
                "report_text": "Right upper lobe pulmonary nodule.",
                "labels": "pulmonary_nodule",
            },
            {
                "case_id": "train_1_a_2.nii.gz",
                "report_text": "The same right upper lobe pulmonary nodule.",
                "labels": "pulmonary_nodule",
            },
            {
                "case_id": "train_2_a_1.nii.gz",
                "report_text": "Left lower lobe lung nodule.",
                "labels": "pulmonary_nodule;emphysema",
            },
            {
                "case_id": "train_3_a_1.nii.gz",
                "report_text": "No pulmonary nodule or pleural effusion.",
                "labels": "",
            },
        ]
    ).to_csv(prepared_dir / "case_index.csv", index=False)
    retriever = SimilarCaseRetrieverTool(local_settings(tmp_path))

    results = retriever.retrieve(
        "",
        labels=["pulmonary_nodule"],
        label_scores={"pulmonary_nodule": 0.82},
        top_k=5,
        query_case_id="valid_4_a_1.nii.gz",
    )

    assert [item.patient_id for item in results[:2]] == ["train_1", "train_2"]
    assert len({item.patient_id for item in results}) == len(results)
    assert all(item.retrieval_strategy == "predicted_conditions" for item in results)
    assert results[0].score_breakdown["condition_overlap"] > 0


def test_similar_case_retrieval_excludes_all_reconstructions_of_query_patient(
    tmp_path: Path,
):
    prepared_dir = tmp_path / "artifacts" / "prepared"
    prepared_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "case_id": "train_8_a_1.nii.gz",
                "report_text": "Pulmonary nodule.",
                "labels": "pulmonary_nodule",
            },
            {
                "case_id": "train_8_b_2.nii.gz",
                "report_text": "Pulmonary nodule in another scan.",
                "labels": "pulmonary_nodule",
            },
            {
                "case_id": "train_9_a_1.nii.gz",
                "report_text": "Pulmonary nodule.",
                "labels": "pulmonary_nodule",
            },
        ]
    ).to_csv(prepared_dir / "case_index.csv", index=False)
    retriever = SimilarCaseRetrieverTool(local_settings(tmp_path))

    results = retriever.retrieve(
        "Pulmonary nodule.",
        labels=["pulmonary_nodule"],
        top_k=5,
        query_case_id="train_8_c_1.nii.gz",
    )

    assert [item.patient_id for item in results] == ["train_9"]


def test_ct_classifier_caches_probabilities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    source = tmp_path / "source"
    (source / "CT_CLIP" / "ct_clip").mkdir(parents=True)
    (source / "transformer_maskgit" / "transformer_maskgit").mkdir(parents=True)
    (source / "CT_CLIP" / "ct_clip" / "ct_clip.py").write_text("", encoding="utf-8")
    (source / "transformer_maskgit" / "transformer_maskgit" / "ctvit.py").write_text(
        "", encoding="utf-8"
    )
    volume = tmp_path / "case.nii.gz"
    volume.write_bytes(b"volume")
    settings = local_settings(tmp_path).model_copy(
        update={
            "ctclip_checkpoint": checkpoint,
            "ctclip_source_dir": source,
            "ctclip_python": Path(sys.executable),
            "ct_attribution_enabled": False,
        }
    )
    tool = CtClassifierTool(settings)
    calls = 0

    def fake_predict(path: str) -> dict[str, float]:
        nonlocal calls
        calls += 1
        return {"atelectasis": 0.8}

    monkeypatch.setattr(tool.runtime, "predict", fake_predict)

    first = tool.predict(str(volume), [])
    second = tool.predict(str(volume), [])

    assert first[2] is False
    assert second[2] is True
    assert calls == 1


def test_ct_classifier_rejects_degenerate_all_positive_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    source = tmp_path / "source"
    (source / "CT_CLIP" / "ct_clip").mkdir(parents=True)
    (source / "transformer_maskgit" / "transformer_maskgit").mkdir(parents=True)
    (source / "CT_CLIP" / "ct_clip" / "ct_clip.py").write_text("", encoding="utf-8")
    (source / "transformer_maskgit" / "transformer_maskgit" / "ctvit.py").write_text(
        "", encoding="utf-8"
    )
    volume = tmp_path / "case.nii.gz"
    volume.write_bytes(b"volume")
    settings = local_settings(tmp_path).model_copy(
        update={
            "ctclip_checkpoint": checkpoint,
            "ctclip_source_dir": source,
            "ctclip_python": Path(sys.executable),
            "ct_cache_enabled": False,
            "ct_attribution_enabled": False,
        }
    )
    tool = CtClassifierTool(settings)
    monkeypatch.setattr(
        tool.runtime,
        "predict",
        lambda path: {label: 0.8 for label in LABEL_IDS},
    )

    predictions, warnings, _, _ = tool.predict(str(volume), [])

    assert not any(item.status == "positive" for item in predictions)
    assert all(item.status == "uncertain" for item in predictions)
    assert warnings and warnings[0].startswith("CT质量门控触发：CT-CLIP 将 18/18")
