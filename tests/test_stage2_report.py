from chestct_agent.stage2_report import build_report_zh
from chestct_agent.stage2_pipeline import LABELS


def test_build_report_zh_includes_sections():
    stage2_json = {
        "case_id": "demo",
        "labels": [
            {"name": name, "status": "positive" if name == "pulmonary_nodule" else "negative", "confidence": 0.9, "ctclip_score": 0.7}
            for name in LABELS
        ],
        "need_human_review": True,
    }
    text = build_report_zh(stage2_json, modality="cxr_chest", report_text="nodule present", scores={name: 0.7 for name in LABELS})
    assert "主要结论" in text
    assert "肺结节" in text
    assert "人工复核" in text
