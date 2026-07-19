import re
from pathlib import Path

from chestct_agent.schemas import EvidenceFromImage, LabelPrediction


SLICE_PATTERN = re.compile(r"slice_(\d+)", re.IGNORECASE)


def _slice_indices(preview_images: list[str]) -> list[int]:
    indices: list[int] = []
    for image_path in preview_images:
        match = SLICE_PATTERN.search(Path(image_path).name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(set(indices))


def build_visual_evidence(
    predictions: list[LabelPrediction],
    preview_images: list[str],
) -> dict[str, EvidenceFromImage]:
    evidence: dict[str, EvidenceFromImage] = {}
    indices = _slice_indices(preview_images)
    slice_range = [indices[0], indices[-1]] if indices else []
    for prediction in predictions:
        if prediction.status == "positive" and preview_images:
            evidence[prediction.name] = EvidenceFromImage(
                slice_range=slice_range,
                preview_images=preview_images[:3],
                localized=False,
                note=(
                    "这些是从原始 CT 生成的病例级预览切片。"
                    "CT-CLIP 当前只提供全体积评分，这些图片不能定位具体病灶。"
                ),
            )
    return evidence
