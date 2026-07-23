from pathlib import Path

import nibabel as nib
import numpy as np

from chestct_agent.agent.planner import ToolPolicy
from chestct_agent.config import Settings
from chestct_agent.schemas import (
    AnalyzeRequest,
    DiagnosticToolEvidence,
    LabelPrediction,
    QwenVisualLabelReview,
)
from chestct_agent.tools.consistency_checker import apply_diagnostic_tool_evidence
from chestct_agent.tools.registry import TOOL_REGISTRY
from chestct_agent.tools.totalseg_diagnostics import TotalSegmentatorDiagnosticTool


def test_effusion_volume_thresholds_are_not_mask_presence(tmp_path: Path) -> None:
    settings = Settings(
        static_dir=tmp_path / "static",
        pleural_effusion_uncertain_ml=1.0,
        pleural_effusion_positive_ml=3.0,
        pericardial_effusion_uncertain_ml=1.0,
        pericardial_effusion_positive_ml=3.0,
    )
    ct = np.zeros((16, 16, 16), dtype=np.int16)
    ct_image = nib.Nifti1Image(ct, np.diag([2.0, 2.0, 2.0, 1.0]))
    segmentation = np.zeros_like(ct, dtype=np.uint8)
    segmentation[2:12, 2:12, 2:7] = 2  # 4.0 mL, positive pleural fluid.
    segmentation[4:9, 4:9, 8:13] = 3  # 1.0 mL, uncertain pericardial fluid.

    evidence = TotalSegmentatorDiagnosticTool(settings)._effusion_evidence(
        "synthetic",
        ct_image,
        ct.astype(np.float32),
        segmentation,
        tmp_path / "segmentation.nii.gz",
        False,
        10.0,
        {"pleural_effusion", "pericardial_effusion"},
    )
    by_label = {item.label: item for item in evidence}
    assert by_label["pleural_effusion"].verdict == "positive"
    assert by_label["pleural_effusion"].metrics["volume_ml"] == 4.0
    assert by_label["pericardial_effusion"].verdict == "uncertain"
    assert len(by_label["pleural_effusion"].preview_images) > 0


def test_nodule_candidate_requires_independent_visual_confirmation() -> None:
    prediction = LabelPrediction(
        name="pulmonary_nodule", status="negative", confidence=0.2, source="ct"
    )
    candidate = DiagnosticToolEvidence(
        label="pulmonary_nodule",
        tool="nodule_segmentation_tool",
        backend="TotalSegmentator",
        verdict="positive",
        confidence=0.95,
        coverage="complete",
    )
    without_visual, _ = apply_diagnostic_tool_evidence([prediction], [candidate], [])
    assert without_visual[0].status == "uncertain"

    visual = QwenVisualLabelReview(
        name="pulmonary_nodule",
        status="positive",
        confidence=0.9,
        backend="independent_slice_vlm",
        model="google/gemma-4-31b-it",
    )
    confirmed, _ = apply_diagnostic_tool_evidence(
        [prediction], [candidate], [visual], visual_minimum_confidence=0.85
    )
    assert confirmed[0].status == "positive"


def test_quantitative_effusion_tool_can_promote_uncertain_result() -> None:
    prediction = LabelPrediction(
        name="pleural_effusion", status="uncertain", confidence=0.48, source="ct"
    )
    volume = DiagnosticToolEvidence(
        label="pleural_effusion",
        tool="effusion_segmentation_tool",
        backend="TotalSegmentator",
        verdict="positive",
        confidence=0.9,
        coverage="complete",
        metrics={"volume_ml": 25.0},
    )
    updated, _ = apply_diagnostic_tool_evidence([prediction], [volume], [])
    assert updated[0].status == "positive"
    assert updated[0].confidence == 0.9


def test_small_nodule_candidate_downgrades_ctclip_positive() -> None:
    prediction = LabelPrediction(
        name="pulmonary_nodule", status="positive", confidence=0.73, source="ct"
    )
    small_candidate = DiagnosticToolEvidence(
        label="pulmonary_nodule",
        tool="nodule_segmentation_tool",
        backend="TotalSegmentator",
        verdict="uncertain",
        confidence=0.58,
        coverage="complete",
        metrics={"max_equivalent_diameter_mm": 2.2},
    )
    updated, _ = apply_diagnostic_tool_evidence([prediction], [small_candidate], [])
    assert updated[0].status == "uncertain"


def test_effusion_double_negative_can_override_screening_false_positive() -> None:
    prediction = LabelPrediction(
        name="pleural_effusion", status="positive", confidence=0.66, source="ct"
    )
    no_fluid = DiagnosticToolEvidence(
        label="pleural_effusion",
        tool="effusion_segmentation_tool",
        backend="TotalSegmentator",
        verdict="negative",
        confidence=0.9,
        coverage="complete",
        metrics={"volume_ml": 0.0},
    )
    visual = QwenVisualLabelReview(
        name="pleural_effusion",
        status="negative",
        confidence=0.9,
        backend="independent_slice_vlm",
        model="google/gemma-4-31b-it",
    )
    updated, _ = apply_diagnostic_tool_evidence(
        [prediction], [no_fluid], [visual], visual_minimum_confidence=0.85
    )
    assert updated[0].status == "negative"


def test_cardiac_measurement_uses_heart_and_lung_masks(tmp_path: Path) -> None:
    settings = Settings(
        static_dir=tmp_path / "static",
        cardiothoracic_ratio_positive=0.5,
    )
    ct = np.zeros((20, 20, 10), dtype=np.float32)
    segmentation = np.zeros_like(ct, dtype=np.uint8)
    segmentation[1:5, 2:8, 4] = 10
    segmentation[15:19, 2:8, 4] = 12
    segmentation[5:15, 5:15, 4] = 51
    ct_image = nib.Nifti1Image(ct, np.eye(4))

    evidence = TotalSegmentatorDiagnosticTool(settings)._cardiac_evidence(
        "synthetic-anatomy",
        ct_image,
        ct,
        segmentation,
        tmp_path / "total.nii.gz",
        False,
        10.0,
        {"cardiomegaly"},
    )
    by_label = {item.label: item for item in evidence}
    assert by_label["cardiomegaly"].verdict == "positive"
    assert by_label["cardiomegaly"].metrics["ct_cardiothoracic_ratio"] > 0.5


def test_ctclip_and_visual_positive_outvote_nodule_segmenter_miss() -> None:
    prediction = LabelPrediction(
        name="pulmonary_nodule", status="positive", confidence=0.76, source="ct"
    )
    segmenter_miss = DiagnosticToolEvidence(
        label="pulmonary_nodule",
        tool="nodule_segmentation_tool",
        backend="TotalSegmentator",
        verdict="negative",
        confidence=0.9,
        coverage="complete",
    )
    visual = QwenVisualLabelReview(
        name="pulmonary_nodule",
        status="positive",
        confidence=0.9,
        backend="independent_slice_vlm",
        model="google/gemma-4-31b-it",
    )
    updated, _ = apply_diagnostic_tool_evidence(
        [prediction], [segmenter_miss], [visual], visual_minimum_confidence=0.85
    )
    assert updated[0].status == "positive"


def test_small_pericardial_volume_triggers_review_not_diagnosis() -> None:
    prediction = LabelPrediction(
        name="pericardial_effusion", status="negative", confidence=0.2, source="ct"
    )
    borderline = DiagnosticToolEvidence(
        label="pericardial_effusion",
        tool="effusion_segmentation_tool",
        backend="TotalSegmentator",
        verdict="uncertain",
        confidence=0.6,
        coverage="complete",
        metrics={"volume_ml": 4.9},
    )
    updated, _ = apply_diagnostic_tool_evidence([prediction], [borderline], [])
    assert updated[0].status == "uncertain"
    assert updated[0].confidence == 0.5


def test_removed_pilot_tools_are_not_registered_or_agent_callable() -> None:
    request = AnalyzeRequest(case_id="case", ct_volume_path="case.nii.gz")
    allowed = set(ToolPolicy.allowed(request))
    assert "nodule_segmentation_tool" in allowed
    assert "effusion_segmentation_tool" in allowed
    assert "emphysema_quantification_tool" not in TOOL_REGISTRY
    assert "emphysema_quantification_tool" not in allowed
    assert "cardiac_measurement_tool" not in allowed
    assert "aortic_calcification_tool" not in TOOL_REGISTRY
    assert "aortic_calcification_tool" not in allowed
