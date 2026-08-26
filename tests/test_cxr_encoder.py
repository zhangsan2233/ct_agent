from pathlib import Path
import tempfile

import pytest

from chestct_agent.cxr_encoder import LIMITED_LABELS, LIMITED_SCORE, map_pathology_scores
from chestct_agent.stage2_pipeline import LABELS


def test_map_pathology_scores_covers_eight_labels():
    raw = {
        "Atelectasis": 0.8,
        "Emphysema": 0.1,
        "Lung Opacity": 0.3,
        "Infiltration": 0.5,
        "Nodule": 0.7,
        "Fibrosis": 0.2,
    }
    mapped = map_pathology_scores(raw)
    assert set(mapped) == set(LABELS)
    assert mapped["atelectasis"] == 0.8
    assert mapped["emphysema"] == 0.1
    assert mapped["lung_opacity"] == 0.5
    assert mapped["pulmonary_nodule"] == 0.7
    assert mapped["pulmonary_fibrotic_sequela"] == 0.2
    for label in LIMITED_LABELS:
        assert mapped[label] == LIMITED_SCORE


def test_cxr_encoder_readiness_without_package():
    from chestct_agent.cxr_encoder import CxrEncoderRuntime

    runtime = CxrEncoderRuntime()
    error = runtime.readiness_error()
    if error is None:
        pytest.skip("torchxrayvision installed in this environment")
    assert "torchxrayvision" in error
