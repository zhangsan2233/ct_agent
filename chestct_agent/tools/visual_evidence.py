from chestct_agent.schemas import EvidenceFromImage, LabelPrediction


def build_visual_evidence(
    predictions: list[LabelPrediction],
    preview_images: list[str],
) -> dict[str, EvidenceFromImage]:
    evidence: dict[str, EvidenceFromImage] = {}
    slice_range = [0, max(0, len(preview_images) - 1)] if preview_images else []
    for prediction in predictions:
        # Generic preview slices are only attached to a positive CT result. A
        # placeholder/uncertain prediction is not lesion-level image evidence.
        if prediction.status == "positive" and preview_images:
            evidence[prediction.name] = EvidenceFromImage(
                slice_range=slice_range,
                preview_images=preview_images[:3],
                note="Preview slices rendered from the original CT volume; not a lesion mask.",
            )
    return evidence
