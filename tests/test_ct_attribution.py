import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch

from chestct_agent.config import Settings
from chestct_agent.ctclip.runtime import CtClipRuntime, PATHOLOGIES
from chestct_agent.schemas import CtAttributionArtifact, LabelPrediction
from chestct_agent.tools.ct_attribution import CtAttributionTool
from chestct_agent.tools.ct_classifier import CtClassifierTool


def test_gradient_x_token_is_finite_and_label_specific():
    torch.manual_seed(7)
    tokens = torch.randn(1, 2, 2, 2, 3)
    projection_weight = torch.randn(4, 12)
    pooled = tokens.mean(dim=1).reshape(1, -1)
    latent = pooled @ projection_weight.T
    normalized = latent / latent.norm(dim=-1, keepdim=True)
    target_gradients = torch.tensor([[1.0, -0.5, 0.25, 0.75], [-0.25, 0.8, 1.1, -0.4]])

    attribution = CtClipRuntime._gradient_x_token(
        tokens,
        latent,
        normalized,
        projection_weight,
        target_gradients,
    )

    assert attribution.shape == (2, 2, 2, 2)
    assert torch.isfinite(attribution).all()
    assert attribution.min() >= 0
    assert attribution.max() <= 1
    assert not torch.allclose(attribution[0], attribution[1])


def test_attribution_renderer_outputs_all_ct_labels(tmp_path: Path):
    volume_path = tmp_path / "case.nii.gz"
    volume = np.linspace(-1000, 500, 8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), volume_path)

    labels = list(PATHOLOGIES.values())
    attributions = np.zeros((18, 24, 24, 24), dtype=np.float16)
    positive_index = labels.index("pulmonary_nodule")
    for temporal_index, spatial_index in ((7, 9), (11, 12), (16, 14)):
        attributions[
            positive_index,
            temporal_index,
            spatial_index : spatial_index + 3,
            spatial_index : spatial_index + 3,
        ] = 1.0

    preprocess = {
        "axis_order": "zxy",
        "original_shape": [8, 10, 12],
        "original_spacing": [1.0, 1.0, 1.0],
        "transposed_shape": [12, 8, 10],
        "resampled_shape": [12, 8, 10],
        "target_shape": [24, 24, 24],
        "crop_start": [0, 0, 0],
        "crop_shape": [12, 8, 10],
        "pad_before": [6, 8, 7],
        "pad_after": [6, 8, 7],
    }
    artifact_path = tmp_path / "attribution.npz"
    np.savez_compressed(
        artifact_path,
        attributions=attributions,
        labels=np.asarray(labels),
        method=np.asarray("gradient_x_token"),
        grid_shape=np.asarray([24, 24, 24]),
        preprocess_json=np.asarray(json.dumps(preprocess)),
    )
    artifact = CtAttributionArtifact(
        artifact_path=str(artifact_path),
        grid_shape=[24, 24, 24],
        preprocess=preprocess,
    )
    settings = Settings(
        openai_compatible_api_key="replace-me",
        artifact_dir=tmp_path / "artifacts",
        static_dir=tmp_path / "static",
        knowledge_dir=tmp_path / "knowledge",
        qdrant_path=tmp_path / "qdrant",
        calibration_path=tmp_path / "calibration.joblib",
        memory_db_path=tmp_path / "memory.sqlite3",
        ct_attribution_slices_per_label=3,
    )
    predictions = [
        LabelPrediction(
            name="pulmonary_nodule",
            status="positive",
            confidence=0.8,
            source="ct",
        ),
        LabelPrediction(
            name="pleural_effusion",
            status="uncertain",
            confidence=0.6,
            source="ct",
        ),
    ]

    evidence, warnings, cache_hit, _ = CtAttributionTool(settings).render(
        "case", str(volume_path), predictions, artifact
    )

    assert warnings == []
    assert cache_hit is False
    assert set(evidence) == {"pulmonary_nodule", "pleural_effusion"}
    assert len(evidence["pulmonary_nodule"].slice_indices) == 3
    assert len(evidence["pulmonary_nodule"].original_images) == 3
    assert len(evidence["pulmonary_nodule"].overlay_images) == 3
    assert evidence["pulmonary_nodule"].target_status == "positive"
    assert evidence["pulmonary_nodule"].target_score == 0.8
    assert len(evidence["pleural_effusion"].overlay_images) == 3
    assert evidence["pleural_effusion"].target_status == "uncertain"
    assert all(Path(path).exists() for path in evidence["pulmonary_nodule"].overlay_images)

    cached, cached_warnings, cached_hit, _ = CtAttributionTool(settings).render(
        "case", str(volume_path), predictions, artifact
    )
    assert cached_warnings == []
    assert cached_hit is True
    assert cached["pulmonary_nodule"].cache_hit is True
    assert all(
        0
        <= CtAttributionTool._native_slice_index(index, 24, preprocess)
        < preprocess["original_shape"][2]
        for index in range(24)
    )


def test_classifier_recovers_from_corrupt_attribution_cache(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    source = tmp_path / "source"
    (source / "CT_CLIP" / "ct_clip").mkdir(parents=True)
    (source / "transformer_maskgit" / "transformer_maskgit").mkdir(parents=True)
    (source / "CT_CLIP" / "ct_clip" / "ct_clip.py").write_text("", encoding="utf-8")
    (source / "transformer_maskgit" / "transformer_maskgit" / "ctvit.py").write_text(
        "", encoding="utf-8"
    )
    volume = tmp_path / "case.nii.gz"
    volume.write_bytes(b"volume")
    settings = Settings(
        openai_compatible_api_key="replace-me",
        artifact_dir=tmp_path / "artifacts",
        static_dir=tmp_path / "static",
        knowledge_dir=tmp_path / "knowledge",
        qdrant_path=tmp_path / "qdrant",
        calibration_path=tmp_path / "calibration.joblib",
        memory_db_path=tmp_path / "memory.sqlite3",
        ctclip_checkpoint=checkpoint,
        ctclip_source_dir=source,
        ctclip_python=Path(sys.executable),
    )
    tool = CtClassifierTool(settings)
    calls = 0

    def fake_predict(path: str):
        nonlocal calls
        calls += 1
        return {
            "probabilities": {"pulmonary_nodule": 0.8},
            "attributions": np.zeros((18, 24, 24, 24), dtype=np.float16),
            "method": "gradient_x_token",
            "grid_shape": [24, 24, 24],
            "preprocess": {},
            "attribution_latency_ms": 1.0,
        }

    monkeypatch.setattr(tool.runtime, "predict_with_attribution", fake_predict)
    first = tool.predict(str(volume), [])
    second = tool.predict(str(volume), [])
    assert first[2] is False
    assert second[2] is True
    assert calls == 1
    assert second[3] is not None

    Path(second[3].artifact_path).write_bytes(b"corrupt")
    recovered = tool.predict(str(volume), [])
    assert recovered[2] is False
    assert recovered[3] is not None
    assert calls == 2

    cache_before_checkpoint_change = tool._cache_path(str(volume))
    checkpoint.write_bytes(b"updated checkpoint")
    assert tool._cache_path(str(volume)) != cache_before_checkpoint_change
