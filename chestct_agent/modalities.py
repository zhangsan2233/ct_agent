"""Modality registry and routing for chest CT (production) and chest X-ray (schematic)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from chestct_agent.input_ingestion import InputIngestionError, ingest_ct_upload, ingest_cxr_upload
from chestct_agent.stage2_contract import DISCLAIMER
from chestct_agent.stage2_pipeline import Stage2Agent, Stage2Paths

if TYPE_CHECKING:
    pass

ModalityId = Literal["ct_chest", "cxr_chest", "mr_chest"]
ModalityStatus = Literal["production", "schematic", "interface_only"]


@dataclass(frozen=True)
class ModalitySpec:
    id: ModalityId
    title_zh: str
    dimensionality: Literal["2d", "3d"]
    body_region: str
    accepted_suffixes: tuple[str, ...]
    encoder: str
    status: ModalityStatus
    label_contract: str
    note_zh: str

    def as_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["accepted_suffixes"] = list(self.accepted_suffixes)
        data["runnable"] = self.status in {"production", "schematic"}
        return data


class ModalityNotReady(ValueError):
    pass


MODALITY_SPECS: tuple[ModalitySpec, ...] = (
    ModalitySpec(
        id="ct_chest",
        title_zh="胸部 CT",
        dimensionality="3d",
        body_region="chest",
        accepted_suffixes=(".nii", ".nii.gz"),
        encoder="frozen_ctclip_v2 + stage2_qlora",
        status="production",
        label_contract="stage2_eight_labels",
        note_zh="正式答辩链路：三维 CT-CLIP 证据 + Stage-2 JSON。",
    ),
    ModalitySpec(
        id="cxr_chest",
        title_zh="胸部 X 光",
        dimensionality="2d",
        body_region="chest",
        accepted_suffixes=(".png", ".jpg", ".jpeg", ".webp", ".bmp"),
        encoder="torchxrayvision_mapped + cxr_stage2_qlora",
        status="schematic",
        label_contract="stage2_eight_labels",
        note_zh="示意全链路：冻结公开 CXR 分类器映射 8 分 + 平行 CXR Stage-2 adapter + 同一 JSON/报告/纠错闭环。",
    ),
    ModalitySpec(
        id="mr_chest",
        title_zh="胸部 MR",
        dimensionality="3d",
        body_region="chest",
        accepted_suffixes=(".nii", ".nii.gz"),
        encoder="not_implemented",
        status="interface_only",
        label_contract="stage2_eight_labels",
        note_zh="仅注册接口，未接入编码器，调用应返回未就绪而不是伪装推理。",
    ),
)
MODALITY_BY_ID = {spec.id: spec for spec in MODALITY_SPECS}

def list_modalities() -> list[dict[str, Any]]:
    return [spec.as_public_dict() for spec in MODALITY_SPECS]


def get_modality(modality: str) -> ModalitySpec:
    spec = MODALITY_BY_ID.get(modality)  # type: ignore[arg-type]
    if spec is None:
        known = ", ".join(MODALITY_BY_ID)
        raise ModalityNotReady(f"未知模态 {modality}。可选：{known}。")
    return spec


def inspect_cxr_image(path: Path) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(path)
    image.load()
    gray = image.convert("L")
    extrema = gray.getextrema()
    histogram = gray.histogram()
    pixel_count = sum(histogram) or 1
    mean_intensity = sum(index * count for index, count in enumerate(histogram)) / pixel_count
    return {
        "path": str(path.resolve()),
        "width": image.size[0],
        "height": image.size[1],
        "mode": image.mode,
        "mean_intensity": round(mean_intensity, 2),
        "intensity_min": int(extrema[0]),
        "intensity_max": int(extrema[1]),
        "used_for": ["format_check", "preview", "encoder_input"],
        "used_as_diagnostic_encoder": True,
    }


def build_stage2_agent(
    root: Path,
    modality: str,
    device: str = "cuda:0",
) -> Stage2Agent:
    paths = Stage2Paths.for_modality(root, modality)  # type: ignore[arg-type]
    return Stage2Agent(paths, device=device)


def analyze_study(
    *,
    modality: str,
    case_id: str,
    image_path: Path | None,
    report_text: str,
    run_dir: Path | None = None,
    stage2_agent: Stage2Agent | None = None,
    enable_llm_2d_review: bool = False,
    root: Path | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    spec = get_modality(modality)
    if spec.status == "interface_only":
        raise ModalityNotReady(spec.note_zh)
    if image_path is None:
        raise InputIngestionError(f"{spec.title_zh} 需要影像文件。")
    if not report_text.strip():
        raise ValueError("Report text is empty.")
    if stage2_agent is None:
        if root is None:
            raise ModalityNotReady(f"{spec.title_zh} 需要 Stage-2 agent。")
        stage2_agent = build_stage2_agent(root, modality, device=device)
    if spec.id == "cxr_chest":
        result = stage2_agent.analyze_cxr(
            case_id=case_id,
            image_path=image_path,
            report_text=report_text,
            run_dir=run_dir,
        )
        result["modality"] = spec.as_public_dict()
        return result
    if spec.id == "ct_chest":
        result = stage2_agent.analyze(
            case_id=case_id,
            ct_path=image_path,
            report_text=report_text,
            run_dir=run_dir,
            enable_llm_2d_review=enable_llm_2d_review,
        )
        result["modality"] = spec.as_public_dict()
        result["warning"] = result.get("warning") or DISCLAIMER
        return result
    raise ModalityNotReady(f"模态 {spec.id} 没有可运行后端。")


def ingest_for_modality(modality: str, filename: str, data: bytes, case_id: str, upload_root: Path) -> Path:
    spec = get_modality(modality)
    if spec.id == "cxr_chest":
        return ingest_cxr_upload(filename, data, case_id, upload_root)
    if spec.id == "ct_chest":
        return ingest_ct_upload(filename, data, case_id, upload_root)
    raise ModalityNotReady(spec.note_zh)


def write_placeholder_cxr(path: Path, size: int = 256) -> Path:
    """Synthetic grayscale PNG for interface smoke tests only."""
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (size, size), 28)
    draw = ImageDraw.Draw(image)
    draw.ellipse((size * 0.12, size * 0.18, size * 0.48, size * 0.88), fill=170)
    draw.ellipse((size * 0.52, size * 0.18, size * 0.88, size * 0.88), fill=170)
    draw.rectangle((size * 0.46, size * 0.22, size * 0.54, size * 0.82), fill=90)
    image.save(path)
    return path
