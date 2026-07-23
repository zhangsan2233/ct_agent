from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from chestct_agent.labels import LABEL_ZH
from chestct_agent.knowledge import STATUS_ZH


class AnalyzeRequest(BaseModel):
    case_id: str = "demo_case"
    report_text: str = ""
    question: str = "What abnormalities are present?"
    ct_volume_path: str | None = None
    ct_source_name: str | None = None
    ct_preview_images: list[str] = Field(default_factory=list)
    top_k_similar: int | None = None
    session_id: str | None = None
    require_human_approval: bool = False

    @model_validator(mode="after")
    def has_analyzable_input(self) -> "AnalyzeRequest":
        if not self.report_text.strip() and not self.ct_volume_path:
            raise ValueError("At least one of report_text or ct_volume_path is required.")
        return self


class ParsedReport(BaseModel):
    findings: str = ""
    impression: str = ""
    full_report: str = ""


class LabelPrediction(BaseModel):
    name: str
    status: Literal["positive", "negative", "uncertain"] = "negative"
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["report", "ct", "fusion", "rule"] = "rule"
    calibrated: bool = False
    calibration_version: str | None = None


class QwenVisualRegion(BaseModel):
    slice_index: int = Field(ge=0)
    window: Literal["lung", "mediastinal"]
    bbox_2d: list[int] = Field(min_length=4, max_length=4)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    description_zh: str = ""

    @field_validator("bbox_2d")
    @classmethod
    def validate_normalized_bbox(cls, value: list[int]) -> list[int]:
        if len(value) != 4:
            raise ValueError("bbox_2d must contain four coordinates")
        x1, y1, x2, y2 = [max(0, min(1000, int(item))) for item in value]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_2d must have positive area")
        return [x1, y1, x2, y2]


class QwenVisualLabelReview(BaseModel):
    name: str
    status: Literal["positive", "negative", "uncertain"] = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    slice_indices: list[int] = Field(default_factory=list)
    evidence_zh: str = ""
    regions: list[QwenVisualRegion] = Field(default_factory=list)
    grounding_heatmap_images: list[str] = Field(default_factory=list)
    backend: str = "qwen"
    model: str = ""
    grounding_method: Literal["vlm_bbox_grounding", "qwen_bbox_grounding"] = (
        "vlm_bbox_grounding"
    )
    grounding_note: str = (
        "切片VLM视觉定位图由模型生成的区域框渲染，不是内部attention、病灶分割或诊断依据。"
    )


class DiagnosticToolEvidence(BaseModel):
    """Structured, independently auditable evidence from an imaging tool."""

    label: str
    tool: str
    backend: str
    model_version: str = ""
    verdict: Literal["positive", "negative", "uncertain", "unavailable"]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage: Literal["complete", "partial", "unavailable"] = "unavailable"
    metrics: dict[str, float] = Field(default_factory=dict)
    mask_paths: list[str] = Field(default_factory=list)
    slice_indices: list[int] = Field(default_factory=list)
    preview_images: list[str] = Field(default_factory=list)
    cache_hit: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)
    rationale_zh: str = ""
    limitation_zh: str = ""


class ReportEvidence(BaseModel):
    sentence: str
    label: str
    polarity: Literal["positive", "negative", "uncertain", "historical"]
    certainty: float = Field(default=1.0, ge=0.0, le=1.0)
    matched_term: str = ""
    sentence_index: int = Field(default=0, ge=0)
    source: Literal["rule", "radgraph_xl"] = "rule"


class ReportGraphNode(BaseModel):
    node_id: str
    text: str
    entity_type: Literal["anatomy", "observation"]
    assertion: Literal["definitely_present", "definitely_absent", "uncertain"]
    start_ix: int = Field(default=0, ge=0)
    end_ix: int = Field(default=0, ge=0)
    sentence_index: int = Field(default=0, ge=0)
    sentence: str = ""
    canonical_label: str | None = None


class ReportGraphEdge(BaseModel):
    source_id: str
    target_id: str
    relation: Literal["modify", "located_at", "suggestive_of"]


class ReportGraph(BaseModel):
    backend: Literal["modern-radgraph-xl", "radgraph-xl", "rule_fallback", "not_used"] = (
        "not_used"
    )
    model_type: str = "none"
    nodes: list[ReportGraphNode] = Field(default_factory=list)
    edges: list[ReportGraphEdge] = Field(default_factory=list)
    degraded: bool = False
    warning: str | None = None


class ModelAttributionEvidence(BaseModel):
    method: Literal["gradient_x_token"] = "gradient_x_token"
    target_label: str
    target_status: Literal["positive", "negative", "uncertain"]
    target_score: float = Field(ge=0.0, le=1.0)
    grid_shape: list[int] = Field(default_factory=list)
    slice_indices: list[int] = Field(default_factory=list)
    original_images: list[str] = Field(default_factory=list)
    overlay_images: list[str] = Field(default_factory=list)
    cache_hit: bool = False
    note: str = (
        "CT-CLIP模型归因图仅解释目标类别阳性分数的空间贡献；"
        "阴性或不确定标签同样展示，但高亮不表示检出病灶。"
    )


class CtAttributionArtifact(BaseModel):
    method: Literal["gradient_x_token"] = "gradient_x_token"
    artifact_path: str
    grid_shape: list[int] = Field(default_factory=list)
    preprocess: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0, ge=0.0)
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0.0)


class EvidenceFromImage(BaseModel):
    slice_range: list[int] = Field(default_factory=list)
    preview_images: list[str] = Field(default_factory=list)
    localized: bool = False
    note: str = ""
    grounding_type: Literal[
        "none", "anatomy_mask", "lesion_mask", "weak_heatmap"
    ] = "none"
    mask_paths: list[str] = Field(default_factory=list)
    bbox_2d: list[int] = Field(default_factory=list)
    bbox_3d: list[int] = Field(default_factory=list)
    anatomy_regions: list[str] = Field(default_factory=list)
    model_attribution: ModelAttributionEvidence | None = None


class RegionFinding(BaseModel):
    label: str
    region: str
    status: Literal["positive", "negative", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    slice_range: list[int] = Field(default_factory=list)
    bbox_2d: list[int] = Field(default_factory=list)
    bbox_3d: list[int] = Field(default_factory=list)
    mask_paths: list[str] = Field(default_factory=list)
    grounding_type: Literal[
        "none", "anatomy_mask", "lesion_mask", "weak_heatmap"
    ] = "none"
    statement_zh: str = ""


class AnatomyMaskResult(BaseModel):
    case_id: str
    anatomy_name: str
    mask_type: Literal["region", "anatomy"]
    mask_path: str
    native_shape: list[int] = Field(default_factory=list)
    ct_shape: list[int] = Field(default_factory=list)
    slice_range: list[int] = Field(default_factory=list)
    bbox_3d: list[int] = Field(default_factory=list)
    overlay_images: list[str] = Field(default_factory=list)
    alignment_method: Literal["affine", "normalized_index_resample", "unavailable"] = (
        "unavailable"
    )
    alignment_verified: bool = False


class LabelOutput(BaseModel):
    name: str
    name_zh: str = ""
    status: Literal["positive", "negative", "uncertain"]
    status_zh: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    source_scores: dict[str, float] = Field(default_factory=dict)
    evidence_from_report: list[ReportEvidence] = Field(default_factory=list)
    evidence_from_image: EvidenceFromImage = Field(default_factory=EvidenceFromImage)
    rag_support: bool = False
    rag_sources: list[str] = Field(default_factory=list)
    need_human_review: bool = True
    decision_source: Literal["model", "human_correction", "dataset_oracle"] = "model"
    original_status: Literal["positive", "negative", "uncertain"] | None = None
    correction_reason: str = ""
    diagnostic_tools: list[DiagnosticToolEvidence] = Field(default_factory=list)

    @field_validator("evidence_from_report", mode="before")
    @classmethod
    def upgrade_legacy_evidence(cls, value):
        upgraded = []
        for item in value or []:
            if isinstance(item, str):
                upgraded.append(
                    {
                        "sentence": item,
                        "label": "",
                        "polarity": "positive",
                        "certainty": 1.0,
                    }
                )
            else:
                upgraded.append(item)
        return upgraded

    @model_validator(mode="after")
    def add_chinese_display_fields(self) -> "LabelOutput":
        if not self.name_zh:
            self.name_zh = LABEL_ZH.get(self.name, self.name)
        if not self.status_zh:
            self.status_zh = STATUS_ZH[self.status]
        return self


class RetrievedDocument(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimilarCase(BaseModel):
    case_id: str
    score: float = Field(ge=0.0)
    matched_labels: list[str] = Field(default_factory=list)
    matched_labels_zh: list[str] = Field(default_factory=list)
    summary: str = ""
    patient_id: str = ""
    source: str = "CT-RATE training reports"
    source_split: str = "train"
    retrieval_strategy: Literal[
        "report_text", "predicted_conditions", "hybrid", "region_aware"
    ] = "report_text"
    matched_regions: list[str] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def add_chinese_matched_labels(self) -> "SimilarCase":
        if not self.matched_labels_zh:
            self.matched_labels_zh = [
                LABEL_ZH.get(label, label) for label in self.matched_labels
            ]
        return self


class ExecutionMetadata(BaseModel):
    input_mode: Literal["report_only", "ct_only", "report_and_ct"] = "report_only"
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    retrieval_attempts: int = Field(default=0, ge=0)
    retrieval_sufficient: bool = False
    rag_backend: str = "not_used"
    ct_cache_hit: bool | None = None
    ct_model_variant: str = "not_used"
    ct_input_name: str | None = None
    ct_input_size_bytes: int | None = Field(default=None, ge=0)
    ct_input_sha256: str | None = None
    ct_quality_degraded: bool = False
    ct_quality_reason: str | None = None
    ct_attribution_method: str = "not_used"
    ct_attribution_cache_hit: bool | None = None
    ct_attribution_latency_ms: float = Field(default=0.0, ge=0.0)
    qwen_vision_used: bool = False
    qwen_vision_image_count: int = Field(default=0, ge=0)
    qwen_vision_latency_ms: float = Field(default=0.0, ge=0.0)
    qwen_vision_fallback_reason: str | None = None
    qwen_grounding_region_count: int = Field(default=0, ge=0)
    qwen_grounding_heatmap_count: int = Field(default=0, ge=0)
    slice_vlm_model: str = "not_used"
    diagnostic_tool_count: int = Field(default=0, ge=0)
    diagnostic_tool_latency_ms: float = Field(default=0.0, ge=0.0)
    llm_calls: int = Field(default=0, ge=0)
    llm_fallbacks: int = Field(default=0, ge=0)
    llm_fallback_reasons: list[str] = Field(default_factory=list)
    node_timings_ms: dict[str, float] = Field(default_factory=dict)
    planned_tools: list[str] = Field(default_factory=list)
    failed_tools: list[str] = Field(default_factory=list)
    recovered_failures: int = Field(default=0, ge=0)
    degraded: bool = False
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0.0)


class HumanApproval(BaseModel):
    required: bool = False
    status: Literal["not_required", "pending", "approved", "rejected"] = "not_required"
    reasons: list[str] = Field(default_factory=list)


class LabelCorrection(BaseModel):
    label: str
    corrected_status: Literal["positive", "negative", "uncertain"]
    reason: str = ""


class CorrectionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    reviewer: str = Field(min_length=1, max_length=128)
    source: Literal["human", "dataset_weak_label"] = "human"
    corrections: list[LabelCorrection] = Field(min_length=1)


class SandboxCorrectionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class AppliedLabelCorrection(BaseModel):
    label: str
    before_status: Literal["positive", "negative", "uncertain"]
    after_status: Literal["positive", "negative", "uncertain"]
    reason: str = ""


class CorrectionEvent(BaseModel):
    created_at: str
    source: Literal["human", "dataset_weak_label"]
    reviewer: str
    items: list[AppliedLabelCorrection] = Field(default_factory=list)


class ToolPlanStep(BaseModel):
    tool: str
    reason: str
    required: bool = True


class ToolPlan(BaseModel):
    objective: str
    steps: list[ToolPlanStep] = Field(default_factory=list)
    generated_by: Literal["policy", "llm", "policy_fallback"] = "policy"
    fallback_reason: str | None = None


class ExecutionEvent(BaseModel):
    sequence: int = Field(ge=1)
    node: str
    tool: str
    status: Literal["running", "success", "recovered", "degraded"] = "success"
    duration_ms: float = Field(default=0.0, ge=0.0)
    attempts: int = Field(default=1, ge=1)
    summary: str = ""
    decision_summary: str = ""
    decision_basis: list[str] = Field(default_factory=list)
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None


class RetrievalAttemptTrace(BaseModel):
    attempt: int = Field(ge=1)
    queries: list[str] = Field(default_factory=list)
    backend: str = "not_used"
    sufficient: bool | None = None
    documents: list[RetrievedDocument] = Field(default_factory=list)


class RagTrace(BaseModel):
    query_history: list[list[str]] = Field(default_factory=list)
    attempts: list[RetrievalAttemptTrace] = Field(default_factory=list)
    final_sufficient: bool = False


class ModelReasoningStep(BaseModel):
    order: int = Field(ge=1)
    stage: str
    decision: str
    evidence: list[str] = Field(default_factory=list)
    uncertainty: str = ""


class ModelReasoningReport(BaseModel):
    generated_by: Literal["qwen", "deterministic_fallback", "not_used"] = "not_used"
    structured_steps_by: Literal["qwen", "audit_trace", "not_used"] = "not_used"
    summary_zh: str = ""
    raw_response_zh: str = ""
    steps: list[ModelReasoningStep] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    case_id: str
    labels: list[LabelOutput]
    ct_preview_images: list[str] = Field(default_factory=list)
    qwen_visual_images: list[str] = Field(default_factory=list)
    qwen_visual_reviews: list[QwenVisualLabelReview] = Field(default_factory=list)
    similar_cases: list[SimilarCase] = Field(default_factory=list)
    explanation_zh: str = ""
    disclaimer: str
    tool_trace: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    execution: ExecutionMetadata = Field(default_factory=ExecutionMetadata)
    region_findings: list[RegionFinding] = Field(default_factory=list)
    anatomy_masks: list[AnatomyMaskResult] = Field(default_factory=list)
    approval: HumanApproval = Field(default_factory=HumanApproval)
    agent_plan: ToolPlan | None = None
    execution_events: list[ExecutionEvent] = Field(default_factory=list)
    rag_trace: RagTrace = Field(default_factory=RagTrace)
    report_graph: ReportGraph = Field(default_factory=ReportGraph)
    model_reasoning: ModelReasoningReport = Field(default_factory=ModelReasoningReport)
    correction_history: list[CorrectionEvent] = Field(default_factory=list)
    diagnostic_evidence: list[DiagnosticToolEvidence] = Field(default_factory=list)

    @field_validator("labels")
    @classmethod
    def labels_are_unique(cls, labels: list[LabelOutput]) -> list[LabelOutput]:
        seen: set[str] = set()
        unique: list[LabelOutput] = []
        for label in labels:
            if label.name not in seen:
                seen.add(label.name)
                unique.append(label)
        return unique


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    case_id: str = Field(min_length=1, max_length=256)
    message: str = Field(min_length=1, max_length=4000)


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class ChatResponse(BaseModel):
    session_id: str
    case_id: str
    answer_zh: str
    intent: str
    tools_used: list[str] = Field(default_factory=list)
    execution_events: list[ExecutionEvent] = Field(default_factory=list)
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    reference_evaluation: dict[str, Any] = Field(default_factory=dict)
    used_remote_model: bool = False
    history: list[ConversationMessage] = Field(default_factory=list)


class AgentState(BaseModel):
    request: AnalyzeRequest
    parsed_report: ParsedReport | None = None
    report_predictions: list[LabelPrediction] = Field(default_factory=list)
    ct_predictions: list[LabelPrediction] = Field(default_factory=list)
    ct_preview_images: list[str] = Field(default_factory=list)
    qwen_visual_images: list[str] = Field(default_factory=list)
    qwen_visual_reviews: list[QwenVisualLabelReview] = Field(default_factory=list)
    qwen_vision_used: bool = False
    qwen_vision_latency_ms: float = Field(default=0.0, ge=0.0)
    qwen_vision_fallback_reason: str | None = None
    diagnostic_evidence: list[DiagnosticToolEvidence] = Field(default_factory=list)
    diagnostic_tool_latency_ms: float = Field(default=0.0, ge=0.0)
    fusion_predictions: list[LabelPrediction] = Field(default_factory=list)
    rag_queries: list[str] = Field(default_factory=list)
    rag_query_history: list[list[str]] = Field(default_factory=list)
    retrieved_docs: list[RetrievedDocument] = Field(default_factory=list)
    retrieval_history: list[RetrievalAttemptTrace] = Field(default_factory=list)
    similar_cases: list[SimilarCase] = Field(default_factory=list)
    evidence_by_label: dict[str, list[ReportEvidence]] = Field(default_factory=dict)
    report_graph: ReportGraph = Field(default_factory=ReportGraph)
    image_evidence_by_label: dict[str, EvidenceFromImage] = Field(default_factory=dict)
    consistency_warnings: list[str] = Field(default_factory=list)
    draft_response: AnalyzeResponse | None = None
    final_response: AnalyzeResponse | None = None
    tool_trace: list[str] = Field(default_factory=list)
    retrieval_sufficient: bool = True
    retrieval_attempts: int = 0
    rag_backend: str = "not_used"
    max_retrieval_attempts: int = 2
    ct_cache_hit: bool | None = None
    ct_input_name: str | None = None
    ct_input_size_bytes: int | None = Field(default=None, ge=0)
    ct_input_sha256: str | None = None
    ct_quality_degraded: bool = False
    ct_quality_reason: str | None = None
    ct_attribution_artifact: CtAttributionArtifact | None = None
    ct_attribution_cache_hit: bool | None = None
    ct_attribution_latency_ms: float = Field(default=0.0, ge=0.0)
    ct_peak_gpu_memory_mb: float | None = Field(default=None, ge=0.0)
    llm_calls: int = 0
    llm_fallbacks: int = 0
    llm_fallback_reasons: list[str] = Field(default_factory=list)
    node_timings_ms: dict[str, float] = Field(default_factory=dict)
    tool_plan: ToolPlan | None = None
    failed_tools: list[str] = Field(default_factory=list)
    recovered_failures: int = 0
    region_findings: list[RegionFinding] = Field(default_factory=list)
    anatomy_masks: list[AnatomyMaskResult] = Field(default_factory=list)
    approval: HumanApproval = Field(default_factory=HumanApproval)
    execution_events: list[ExecutionEvent] = Field(default_factory=list)
