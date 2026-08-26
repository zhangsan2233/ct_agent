"""Shared Stage-2 eight-label contract constants."""
from __future__ import annotations

from typing import Literal

LABELS = [
    "arterial_wall_calcification",
    "atelectasis",
    "coronary_artery_wall_calcification",
    "emphysema",
    "lung_opacity",
    "lymphadenopathy",
    "pulmonary_fibrotic_sequela",
    "pulmonary_nodule",
]

SYSTEM_PROMPT = (
    "You are ChestCT-Agent, a research-only chest CT evidence integrator. Return compact JSON only. "
    "Use only the supplied report impression and CT-CLIP scores; do not invent findings. Preserve uncertainty "
    "and require human review because labels are weak supervision."
)

DISCLAIMER = "Research-only weak-supervision output; not for clinical diagnosis. Human review is required."
CXR_DISCLAIMER = (
    "Research-only CXR schematic backend using frozen public classifier scores mapped to the Stage-2 "
    "eight-label contract; not for clinical diagnosis. Human review is required."
)

CXR_APPLICABILITY: dict[str, Literal["supported", "limited"]] = {
    "arterial_wall_calcification": "limited",
    "atelectasis": "supported",
    "coronary_artery_wall_calcification": "limited",
    "emphysema": "supported",
    "lung_opacity": "supported",
    "lymphadenopathy": "limited",
    "pulmonary_fibrotic_sequela": "supported",
    "pulmonary_nodule": "supported",
}

LIMITED_LABELS = frozenset(
    {
        "arterial_wall_calcification",
        "coronary_artery_wall_calcification",
        "lymphadenopathy",
    }
)
LIMITED_SCORE = 0.5
