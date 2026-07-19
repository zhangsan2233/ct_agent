from datetime import datetime, timezone
import csv
from pathlib import Path

from chestct_agent.knowledge import STATUS_ZH
from chestct_agent.labels import LABEL_SPECS
from chestct_agent.schemas import (
    AnalyzeResponse,
    AppliedLabelCorrection,
    CorrectionEvent,
    CorrectionRequest,
    HumanApproval,
    LabelCorrection,
)


def load_ct_rate_reference_labels(data_dir: Path, case_id: str) -> set[str] | None:
    path = data_dir / "dataset" / "multi_abnormality_labels" / "valid_predicted_labels.csv"
    if not path.exists():
        return None
    volume_name = case_id if case_id.endswith(".nii.gz") else f"{case_id}.nii.gz"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("VolumeName") == volume_name:
                return {
                    spec.id
                    for spec in LABEL_SPECS
                    if row.get(spec.source_column) == "1"
                }
    return None


def dataset_correction_request(
    response: AnalyzeResponse, session_id: str, reference: set[str]
) -> CorrectionRequest:
    return CorrectionRequest(
        session_id=session_id,
        reviewer="ct-rate-hidden-weak-label",
        source="dataset_weak_label",
        corrections=[
            LabelCorrection(
                label=item.name,
                corrected_status="positive" if item.name in reference else "negative",
                reason="CT-RATE报告派生隐藏弱标签反馈",
            )
            for item in response.labels
        ],
    )


def apply_corrections(
    response: AnalyzeResponse, request: CorrectionRequest
) -> tuple[AnalyzeResponse, CorrectionEvent]:
    by_label = {item.name: item for item in response.labels}
    unknown = sorted({item.label for item in request.corrections} - set(by_label))
    if unknown:
        raise ValueError("Unknown correction labels: " + ", ".join(unknown))
    if len({item.label for item in request.corrections}) != len(request.corrections):
        raise ValueError("Correction labels must be unique.")
    complete_review = {item.label for item in request.corrections} == set(by_label)

    decision_source = (
        "human_correction" if request.source == "human" else "dataset_oracle"
    )
    applied: list[AppliedLabelCorrection] = []
    for correction in request.corrections:
        label = by_label[correction.label]
        before = label.status
        applied.append(
            AppliedLabelCorrection(
                label=label.name,
                before_status=before,
                after_status=correction.corrected_status,
                reason=correction.reason,
            )
        )
        label.status = correction.corrected_status
        label.status_zh = STATUS_ZH[correction.corrected_status]
        label.original_status = label.original_status or before
        label.decision_source = decision_source
        label.correction_reason = correction.reason
        label.need_human_review = request.source != "human" or not complete_review

    corrected_status = {item.label: item.after_status for item in applied}
    for finding in response.region_findings:
        if finding.label in corrected_status:
            finding.status = corrected_status[finding.label]

    event = CorrectionEvent(
        created_at=datetime.now(timezone.utc).isoformat(),
        source=request.source,
        reviewer=request.reviewer,
        items=applied,
    )
    response.correction_history.append(event)
    changed = [item for item in applied if item.before_status != item.after_status]
    response.warnings.append(
        (
            f"医生{request.reviewer}已提交逐标签纠错，共修改{len(changed)}项。"
            if request.source == "human"
            else "已进入CT-RATE隐藏弱标签训练沙箱；该结果发生标签泄漏，"
            f"仅用于演示纠错循环，共修改{len(changed)}项，不得作为模型评估结果。"
        )
    )
    if request.source == "human" and complete_review:
        response.approval = HumanApproval(
            required=False,
            status="approved",
            reasons=[f"{request.reviewer}已完成逐标签复核。"],
        )
    elif request.source == "human":
        response.approval = HumanApproval(
            required=True,
            status="pending",
            reasons=["本次仅复核了部分标签，仍需完成全部18类复核。"],
        )
    else:
        response.approval = HumanApproval(
            required=True,
            status="pending",
            reasons=["结果由数据集弱标签纠正，不代表临床金标准。"],
        )
    return response, event
