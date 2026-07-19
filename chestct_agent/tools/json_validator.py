from pydantic import ValidationError

from chestct_agent.knowledge import LABEL_ZH
from chestct_agent.schemas import AnalyzeResponse


def validate_response(response: AnalyzeResponse) -> tuple[AnalyzeResponse, list[str]]:
    warnings: list[str] = []
    try:
        validated = AnalyzeResponse.model_validate(response.model_dump())
    except ValidationError as exc:
        raise ValueError(f"Invalid AnalyzeResponse: {exc}") from exc

    if not validated.disclaimer:
        warnings.append("缺少使用范围声明。")
    for label in validated.labels:
        has_report_support = any(
            evidence.polarity in {"positive", "uncertain"}
            for evidence in label.evidence_from_report
        )
        has_image_support = label.evidence_from_image.grounding_type in {
            "lesion_mask",
            "weak_heatmap",
        }
        has_direct_evidence = has_report_support or has_image_support
        if label.status == "positive" and not has_direct_evidence:
            warnings.append(
                f"阳性结论缺少可定位的直接证据：{LABEL_ZH.get(label.name, label.name)}"
            )
    return validated, warnings
