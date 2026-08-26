"""Shared path resolution for CT and CXR Stage-2 inference."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

ModalityId = Literal["ct_chest", "cxr_chest"]

CT_ADAPTER_REL = Path("artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep/adapter")
CXR_ADAPTER_REL = Path("artifacts/llm_qlora/qwen3_5_9b_cxr_stage2/adapter")
CXR_ADAPTER_CANDIDATE_PREFIX = "artifacts/llm_qlora/qwen3_5_9b_cxr_stage2_candidate_"


def resolve_qwen_model_dir(root: Path) -> Path:
    candidates = (
        root / "models" / "Qwen3.5-9B",
        root / "models" / "qwen3_5_9B" / "Qwen3.5-9B",
    )
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def ct_adapter_dir(root: Path) -> Path:
    return root / CT_ADAPTER_REL


def cxr_adapter_dir(root: Path) -> Path:
    return root / CXR_ADAPTER_REL


def adapter_dir_for_modality(root: Path, modality: ModalityId) -> Path:
    if modality == "ct_chest":
        return ct_adapter_dir(root)
    if modality == "cxr_chest":
        return cxr_adapter_dir(root)
    raise ValueError(f"Unsupported modality: {modality}")


def model_version_tag(modality: str, adapter_dir: Path) -> str:
    return f"{modality}:{adapter_dir.name}"
