from chestct_agent.validated_memory_pipeline import LABELS, _normalize


def test_normalize_accepts_findings_and_chinese_labels():
    raw = {
        "findings": [
            {
                "disease": "动脉壁钙化",
                "decision": "阳性",
                "confidence": 0.81,
                "evidence": "纵隔窗可见高密度影",
            },
            {
                "disease": "肺不张",
                "decision": "阴性",
                "confidence": 0.76,
            },
        ]
    }

    normalized = _normalize(raw)

    assert normalized["arterial_wall_calcification"]["status"] == "positive"
    assert normalized["atelectasis"]["status"] == "negative"


def test_normalize_accepts_top_level_label_mapping_and_boolean_status():
    raw = {
        label: {
            "prediction": index % 2 == 0,
            "confidence": 0.8,
        }
        for index, label in enumerate(LABELS)
    }

    normalized = _normalize(raw)

    assert len(normalized) == len(LABELS)
    assert normalized[LABELS[0]]["status"] == "positive"
    assert normalized[LABELS[1]]["status"] == "negative"


def test_normalize_accepts_single_assessment_object():
    raw = {
        "label": "arterial_wall_calcification",
        "status": "negative",
        "confidence": 0.93,
        "visible_evidence": "未见明确动脉壁钙化",
    }

    normalized = _normalize(raw)

    assert list(normalized) == ["arterial_wall_calcification"]
    assert normalized["arterial_wall_calcification"]["confidence"] == 0.93
