from io import BytesIO
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from PIL import Image
import pytest

from chestct_agent.input_ingestion import InputIngestionError, ingest_cxr_upload
from chestct_agent.modalities import (
    ModalityNotReady,
    analyze_study,
    build_stage2_agent,
    ingest_for_modality,
    list_modalities,
    write_placeholder_cxr,
)
from chestct_agent.stage2_pipeline import LABELS, Stage2Paths

ROOT = Path(__file__).resolve().parents[1]


def _scratch() -> Path:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="modality_", dir=str(artifacts)))


def test_modality_registry_exposes_ct_cxr_and_reserved_mr():
    specs = {item["id"]: item for item in list_modalities()}
    assert specs["ct_chest"]["status"] == "production"
    assert specs["cxr_chest"]["status"] == "schematic"
    assert specs["mr_chest"]["status"] == "interface_only"
    assert specs["cxr_chest"]["encoder"].startswith("torchxrayvision")


def test_stage2_paths_use_parallel_adapters():
    scratch = _scratch()
    ct = Stage2Paths.for_modality(scratch, "ct_chest")
    cxr = Stage2Paths.for_modality(scratch, "cxr_chest")
    assert "ctclip_stage2" in str(ct.adapter_dir)
    assert "cxr_stage2" in str(cxr.adapter_dir)
    assert ct.adapter_dir != cxr.adapter_dir


def test_analyze_study_requires_report():
    scratch = _scratch()
    with pytest.raises(ValueError, match="Report text is empty"):
        analyze_study(
            modality="cxr_chest",
            case_id="no_report",
            image_path=write_placeholder_cxr(scratch / "cxr.png"),
            report_text="",
            root=ROOT,
        )


@patch("chestct_agent.stage2_pipeline.Stage2Agent._generate")
@patch("chestct_agent.stage2_pipeline.Stage2Agent._cxr_scores")
def test_cxr_full_pipeline_mocked(mock_scores, mock_generate):
    scratch = _scratch()
    image_path = write_placeholder_cxr(scratch / "cxr.png")
    mapped = {label: 0.1 + index / 20 for index, label in enumerate(LABELS)}
    mock_scores.return_value = (
        mapped,
        {"encoder": "torchxrayvision/test", "limited_labels": []},
    )
    import json

    labels = [
        {
            "name": name,
            "status": "positive" if name == "pulmonary_nodule" else "negative",
            "confidence": 0.9,
            "ctclip_score": mapped[name],
        }
        for name in LABELS
    ]
    mock_generate.return_value = json.dumps(
        {
            "case_id": "cxr_demo",
            "labels": labels,
            "need_human_review": True,
            "disclaimer": "test",
        },
        ensure_ascii=False,
    )
    adapter = scratch / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    paths = Stage2Paths.for_modality(scratch, "cxr_chest")
    paths = Stage2Paths(
        model_dir=scratch / "qwen",
        adapter_dir=adapter,
        ctclip_checkpoint=scratch / "ct.pt",
        ctclip_source=scratch / "ctclip",
        text_model_dir=scratch / "bert",
        modality="cxr_chest",
    )
    agent = MagicMock()
    agent.paths = paths
    agent.modality = "cxr_chest"
    agent.device = "cpu"
    agent.max_new_tokens = 128
    agent.readiness_errors.return_value = []
    agent.analyze_cxr.side_effect = None

    from chestct_agent.stage2_pipeline import Stage2Agent

    real = Stage2Agent.__new__(Stage2Agent)
    real.paths = paths
    real.device = "cpu"
    real.max_new_tokens = 128
    real.modality = "cxr_chest"
    real.model = None
    real.tokenizer = None
    real.ctclip = None
    real.cxr_encoder = None
    with patch.object(Stage2Agent, "_cxr_scores", mock_scores), patch.object(
        Stage2Agent, "_generate", mock_generate
    ), patch.object(Stage2Agent, "readiness_errors", return_value=[]):
        result = Stage2Agent(paths, "cpu").analyze_cxr(
            case_id="cxr_demo",
            image_path=image_path,
            report_text="There is a pulmonary nodule.",
            run_dir=scratch / "run",
        )
    assert result["stage2_json"] is not None
    assert result["report_zh"]
    assert result["ctclip_scores"] == mapped
    assert result["validation"]["schema_valid"] is True


def test_mr_is_registered_but_not_runnable():
    scratch = _scratch()
    with pytest.raises(ModalityNotReady):
        analyze_study(
            modality="mr_chest",
            case_id="mr",
            image_path=scratch / "missing.nii.gz",
            report_text="unused",
            root=ROOT,
        )


def test_ingest_cxr_upload_stores_png():
    scratch = _scratch()
    buffer = BytesIO()
    Image.new("L", (64, 64), 80).save(buffer, format="PNG")
    output = ingest_cxr_upload("patient chest.png", buffer.getvalue(), "case A", scratch)
    assert output.name == "cxr.png"
    assert output.is_file()


def test_ingest_for_modality_routes_cxr():
    scratch = _scratch()
    path = ingest_for_modality(
        "cxr_chest",
        "demo.png",
        write_placeholder_cxr(scratch / "src.png").read_bytes(),
        "route",
        scratch / "uploads",
    )
    assert path.name == "cxr.png"
