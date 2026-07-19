from pydantic import ValidationError

from chestct_agent.schemas import AnalyzeResponse


def validate_response(response: AnalyzeResponse) -> tuple[AnalyzeResponse, list[str]]:
    warnings: list[str] = []
    try:
        validated = AnalyzeResponse.model_validate(response.model_dump())
    except ValidationError as exc:
        raise ValueError(f"Invalid AnalyzeResponse: {exc}") from exc

    if not validated.disclaimer:
        warnings.append("Missing disclaimer.")
    for label in validated.labels:
        if label.status == "positive" and not label.evidence_from_report and not label.evidence_from_image.preview_images:
            warnings.append(f"Positive label has no direct evidence: {label.name}")
    return validated, warnings

