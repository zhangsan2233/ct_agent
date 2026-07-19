from chestct_agent.labels import LABEL_BY_ID
from chestct_agent.schemas import (
    AnatomyMaskResult,
    EvidenceFromImage,
    LabelPrediction,
    RegionFinding,
)


REGION_ALIASES = {
    "lung": {"lung", "lobe", "parenchyma"},
    "pleura": {"pleura", "pleural"},
    "heart": {"heart", "pericard", "coronary"},
    "mediastinum": {"mediastinum", "aorta", "lymph", "hilar"},
    "trachea and bronchie": {"trachea", "bronch", "airway"},
    "esophagus": {"esophagus", "hiatal", "diaphragm"},
}

LESION_MASK_ALIASES: dict[str, set[str]] = {
    "pulmonary_nodule": {"lung nodule"},
    "pleural_effusion": {"lung effusion"},
}


def _relevant_masks(label: str, masks: list[AnatomyMaskResult]) -> list[AnatomyMaskResult]:
    spec = LABEL_BY_ID[label]
    terms = " ".join(spec.anatomy_regions).lower()
    requested = {
        broad
        for broad, aliases in REGION_ALIASES.items()
        if any(alias in terms for alias in aliases)
    }
    matches = [
        mask
        for mask in masks
        if any(
            region in mask.anatomy_name.lower() or mask.anatomy_name.lower() in region
            for region in requested
        )
    ]
    return matches[:3]


def _lesion_masks(label: str, masks: list[AnatomyMaskResult]) -> list[AnatomyMaskResult]:
    aliases = LESION_MASK_ALIASES.get(label, set())
    return [
        mask
        for mask in masks
        if mask.mask_type == "anatomy" and mask.anatomy_name.lower() in aliases
    ][:3]


def ground_findings(
    predictions: list[LabelPrediction],
    masks: list[AnatomyMaskResult],
) -> tuple[list[RegionFinding], dict[str, EvidenceFromImage]]:
    findings: list[RegionFinding] = []
    image_evidence: dict[str, EvidenceFromImage] = {}
    for prediction in predictions:
        if prediction.status not in {"positive", "uncertain"}:
            continue
        lesion_masks = _lesion_masks(prediction.name, masks)
        relevant = lesion_masks or _relevant_masks(prediction.name, masks)
        if not relevant:
            continue
        primary = relevant[0]
        grounding_type = "lesion_mask" if lesion_masks else "anatomy_mask"
        if grounding_type == "lesion_mask":
            statement = (
                f"RadGenome病灶mask将该结论定位到“{primary.anatomy_name}”；"
                f"切片范围为 {primary.slice_range}，对齐验证状态为 {primary.alignment_verified}。"
            )
            note = (
                f"RadGenome病灶级分割；对齐方法：{primary.alignment_method}；"
                f"验证状态：{primary.alignment_verified}。"
            )
        else:
            statement = (
                f"该结论定位到相关解剖区域“{primary.anatomy_name}”；"
                "mask是解剖区域证据，不代表病灶轮廓。"
            )
            note = (
                "RadGenome解剖区域grounding，不是病灶分割。"
                f"对齐方法：{primary.alignment_method}；验证状态：{primary.alignment_verified}。"
            )
        findings.append(
            RegionFinding(
                label=prediction.name,
                region=primary.anatomy_name,
                status=prediction.status,
                confidence=prediction.confidence,
                slice_range=primary.slice_range,
                bbox_3d=primary.bbox_3d,
                mask_paths=[item.mask_path for item in relevant],
                grounding_type=grounding_type,
                statement_zh=statement,
            )
        )
        image_evidence[prediction.name] = EvidenceFromImage(
            slice_range=primary.slice_range,
            preview_images=[image for item in relevant for image in item.overlay_images][:3],
            localized=True,
            note=note,
            grounding_type=grounding_type,
            mask_paths=[item.mask_path for item in relevant],
            bbox_3d=primary.bbox_3d,
            anatomy_regions=[item.anatomy_name for item in relevant],
        )
    return findings, image_evidence
