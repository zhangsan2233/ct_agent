"""Frozen TorchXRayVision encoder mapped to the Stage-2 eight-label contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chestct_agent.stage2_contract import (
    CXR_APPLICABILITY,
    DISCLAIMER,
    LABELS,
    LIMITED_LABELS,
    LIMITED_SCORE,
    SYSTEM_PROMPT,
    CXR_DISCLAIMER,
)

# TorchXRayVision pathology keys for densenet121-res224-all
TXV_ATELECTASIS = "Atelectasis"
TXV_EMPHYSEMA = "Emphysema"
TXV_LUNG_OPACITY = "Lung Opacity"
TXV_INFILTRATION = "Infiltration"
TXV_CONSOLIDATION = "Consolidation"
TXV_NODULE = "Nodule"
TXV_MASS = "Mass"
TXV_LUNG_LESION = "Lung Lesion"
TXV_FIBROSIS = "Fibrosis"


def _norm_key(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def map_pathology_scores(pathologies: dict[str, float]) -> dict[str, float]:
    """Map raw TorchXRayVision pathology probabilities to Stage-2 label scores."""
    lookup = {_norm_key(key): float(value) for key, value in pathologies.items()}

    def pick(*keys: str, default: float = 0.0) -> float:
        values = [lookup[_norm_key(key)] for key in keys if _norm_key(key) in lookup]
        return max(values, default=default) if values else default

    mapped = {
        "atelectasis": pick(TXV_ATELECTASIS),
        "emphysema": pick(TXV_EMPHYSEMA),
        "lung_opacity": pick(TXV_LUNG_OPACITY, TXV_INFILTRATION, TXV_CONSOLIDATION),
        "pulmonary_nodule": pick(TXV_NODULE, TXV_MASS, TXV_LUNG_LESION),
        "pulmonary_fibrotic_sequela": pick(TXV_FIBROSIS),
        "arterial_wall_calcification": LIMITED_SCORE,
        "coronary_artery_wall_calcification": LIMITED_SCORE,
        "lymphadenopathy": LIMITED_SCORE,
    }
    return {label: round(float(mapped[label]), 4) for label in LABELS}


@dataclass
class CxrEncoderRuntime:
    device: str = "cuda:0"
    model_name: str = "densenet121-res224-all"
    _model: Any = None
    _transform: Any = None

    def readiness_error(self) -> str | None:
        try:
            import torchxrayvision  # noqa: F401
        except ImportError:
            return "torchxrayvision is not installed (pip install torchxrayvision)."
        return None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        import torchxrayvision as xrv

        self._model = xrv.models.DenseNet(weights=self.model_name)
        self._model = self._model.to(self.device)
        self._model.eval()
        self._transform = xrv.datasets.XRayCenterCrop()

    def predict(self, image_path: Path) -> dict[str, Any]:
        import numpy as np
        import torch
        import torchxrayvision as xrv
        from PIL import Image

        error = self.readiness_error()
        if error:
            raise RuntimeError(error)
        self._ensure_loaded()
        image = Image.open(image_path).convert("L")
        array = np.array(image, dtype=np.float32)
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        tensor = xrv.datasets.normalize(tensor, self._model)
        tensor = self._transform(tensor)
        with torch.inference_mode():
            outputs = self._model(tensor.to(self.device))
        pathologies = {
            pathology: float(outputs[0, index].item())
            for index, pathology in enumerate(self._model.pathologies)
        }
        mapped = map_pathology_scores(pathologies)
        return {
            "pathologies": {key: round(value, 4) for key, value in pathologies.items()},
            "mapped_scores": mapped,
            "limited_labels": sorted(LIMITED_LABELS),
            "encoder": f"torchxrayvision/{self.model_name}",
        }
