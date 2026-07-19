from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    case_id: str = "demo_case"
    report_text: str
    question: str = "What abnormalities are present?"
    ct_volume_path: str | None = None
    ct_preview_images: list[str] = Field(default_factory=list)
    top_k_similar: int | None = None


class ParsedReport(BaseModel):
    findings: str = ""
    impression: str = ""
    full_report: str = ""


class LabelPrediction(BaseModel):
    name: str
    status: Literal["positive", "negative", "uncertain"] = "negative"
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["report", "ct", "fusion", "rule"] = "rule"


class EvidenceFromImage(BaseModel):
    slice_range: list[int] = Field(default_factory=list)
    preview_images: list[str] = Field(default_factory=list)
    note: str = ""


class LabelOutput(BaseModel):
    name: str
    status: Literal["positive", "negative", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    source_scores: dict[str, float] = Field(default_factory=dict)
    evidence_from_report: list[str] = Field(default_factory=list)
    evidence_from_image: EvidenceFromImage = Field(default_factory=EvidenceFromImage)
    rag_support: bool = False
    need_human_review: bool = True


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
    summary: str = ""


class AnalyzeResponse(BaseModel):
    case_id: str
    labels: list[LabelOutput]
    ct_preview_images: list[str] = Field(default_factory=list)
    similar_cases: list[SimilarCase] = Field(default_factory=list)
    explanation_zh: str = ""
    disclaimer: str
    tool_trace: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

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


class AgentState(BaseModel):
    request: AnalyzeRequest
    parsed_report: ParsedReport | None = None
    report_predictions: list[LabelPrediction] = Field(default_factory=list)
    ct_predictions: list[LabelPrediction] = Field(default_factory=list)
    ct_preview_images: list[str] = Field(default_factory=list)
    fusion_predictions: list[LabelPrediction] = Field(default_factory=list)
    rag_queries: list[str] = Field(default_factory=list)
    retrieved_docs: list[RetrievedDocument] = Field(default_factory=list)
    similar_cases: list[SimilarCase] = Field(default_factory=list)
    evidence_by_label: dict[str, list[str]] = Field(default_factory=dict)
    image_evidence_by_label: dict[str, EvidenceFromImage] = Field(default_factory=dict)
    consistency_warnings: list[str] = Field(default_factory=list)
    draft_response: AnalyzeResponse | None = None
    final_response: AnalyzeResponse | None = None
    tool_trace: list[str] = Field(default_factory=list)
    retrieval_sufficient: bool = True
