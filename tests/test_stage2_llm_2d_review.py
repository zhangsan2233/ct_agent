from pathlib import Path
from unittest.mock import patch
import tempfile

from chestct_agent.stage2_llm_2d_review import (
    apply_audit_to_clip,
    build_agreement,
    format_review_zh,
    normalize_audits,
    normalize_review_sections,
    normalize_votes,
    write_markdown_report,
)
from chestct_agent.stage2_pipeline import LABELS, Stage2Agent, Stage2Paths


def test_normalize_votes_fills_missing_and_invalid():
    raw = [
        {"name": "emphysema", "vote": "visible", "confidence": 0.8, "evidence_zh": "可见"},
        {"name": "atelectasis", "vote": "bogus", "confidence": 2.0},
    ]
    votes = normalize_votes(raw)
    assert len(votes) == len(LABELS)
    emphysema = next(item for item in votes if item["name"] == "emphysema")
    atelectasis = next(item for item in votes if item["name"] == "atelectasis")
    assert emphysema["vote"] == "visible"
    assert atelectasis["vote"] == "insufficient_coverage"
    assert atelectasis["confidence"] == 1.0


def test_build_agreement_classifies_divergence():
    votes = normalize_votes(
        [
            {"name": "emphysema", "vote": "visible", "confidence": 0.9, "evidence_zh": ""},
            {"name": "atelectasis", "vote": "not_visible_on_slices", "confidence": 0.7, "evidence_zh": ""},
            {"name": "lung_opacity", "vote": "visible", "confidence": 0.6, "evidence_zh": ""},
            {"name": "pulmonary_nodule", "vote": "insufficient_coverage", "confidence": 0.2, "evidence_zh": ""},
        ]
    )
    scores = {
        "emphysema": 0.8,
        "atelectasis": 0.8,
        "lung_opacity": 0.2,
        "pulmonary_nodule": 0.1,
    }
    for name in LABELS:
        scores.setdefault(name, 0.1)
    agreement = build_agreement(votes, scores)
    assert "emphysema" in agreement["aligned_labels"]
    assert "lung_opacity" in agreement["llm_visible_ctclip_low"]
    assert "atelectasis" in agreement["llm_not_visible_ctclip_high"]
    assert "pulmonary_nodule" in agreement["insufficient_coverage_labels"]


def test_normalize_review_sections_fills_missing_fields():
    agreement = {"overall_match": "partial"}
    review, incomplete = normalize_review_sections({"match_summary_zh": "部分相符"}, agreement)
    assert review["match_summary_zh"] == "部分相符"
    assert review["agreement_decision"] == "partial_agree"
    assert incomplete is True
    assert "一、符合度" in format_review_zh(review)


def test_write_markdown_report_embeds_slice_images():
    with tempfile.TemporaryDirectory(dir=Path.cwd() / "artifacts") as tmp:
        run_dir = Path(tmp) / "run"
        slice_dir = run_dir / "llm_2d_slices"
        slice_dir.mkdir(parents=True)
        slice_path = slice_dir / "slice_0100_paired.jpg"
        slice_path.write_bytes(b"fake")
        votes = normalize_votes([])
        agreement = build_agreement(votes, {name: 0.1 for name in LABELS})
        review, _ = normalize_review_sections(None, agreement)
        report = write_markdown_report(
            run_dir,
            case_id="demo",
            votes=votes,
            agreement=agreement,
            review=review,
            review_zh=format_review_zh(review),
            slice_paths=[slice_path],
            ctclip_scores={name: 0.1 for name in LABELS},
        )
        text = report.read_text(encoding="utf-8")
        assert "llm_2d_slices/slice_0100_paired.jpg" in text
        assert "一、符合度" in text


def test_analyze_disabled_llm_2d_review_unchanged():
    with tempfile.TemporaryDirectory(dir=Path.cwd() / "artifacts") as tmp:
        base = Path(tmp)
        paths = Stage2Paths(
            model_dir=base / "model",
            adapter_dir=base / "adapter",
            ctclip_checkpoint=base / "ct.pt",
            ctclip_source=base / "ctclip",
            text_model_dir=base / "cxr",
        )
        agent = Stage2Agent(paths, device="cpu")
        ct_path = base / "case.nii.gz"
        ct_path.write_bytes(b"data")
        with (
            patch.object(agent, "readiness_errors", return_value=[]),
            patch.object(agent, "_ctclip_scores", return_value={name: 0.1 for name in LABELS}),
            patch.object(agent, "_release_ctclip"),
            patch.object(agent, "_generate", return_value='{"case_id":"c1","labels":[],"need_human_review":true}'),
        ):
            result = agent.analyze(
                case_id="c1",
                ct_path=ct_path,
                report_text="report",
                run_dir=base / "run",
                enable_llm_2d_review=False,
            )
        assert result["llm_2d_review"] == {"enabled": False}


def test_normalize_audits_and_flip_only_on_reject():
    raw = [
        {"name": "emphysema", "verdict": "confirm", "confidence": 0.9, "evidence_zh": "支持"},
        {"name": "atelectasis", "verdict": "disagree", "confidence": 0.8, "evidence_zh": "矛盾"},
        {"name": "lung_opacity", "verdict": "bogus"},
    ]
    audits = normalize_audits(raw)
    by_name = {item["name"]: item for item in audits}
    assert by_name["emphysema"]["verdict"] == "confirm"
    assert by_name["atelectasis"]["verdict"] == "reject"
    assert by_name["lung_opacity"]["verdict"] == "insufficient_coverage"
    scores = {name: 0.8 if name == "atelectasis" else 0.2 for name in LABELS}
    scores["emphysema"] = 0.9
    corrected = apply_audit_to_clip(audits, scores)
    assert corrected["emphysema"] is True
    assert corrected["atelectasis"] is False
    assert corrected["lung_opacity"] is False


def test_summarize_audits_counts_misjudgment_and_lift():
    from scripts.evaluate_stage2_llm_2d_clip_audit import summarize_audits

    labels = list(LABELS)
    audits_ok_reject = [{"name": name, "verdict": "insufficient_coverage", "confidence": 0.0, "evidence_zh": ""} for name in labels]
    audits_ok_reject[0] = {"name": labels[0], "verdict": "reject", "confidence": 0.9, "evidence_zh": "flip wrong clip"}
    row_good = {
        "ok": True,
        "ctclip_scores": {name: 0.9 if name == labels[0] else 0.1 for name in labels},
        "weak_positive": [],
        "audits": audits_ok_reject,
    }
    audits_harm = [{"name": name, "verdict": "insufficient_coverage", "confidence": 0.0, "evidence_zh": ""} for name in labels]
    audits_harm[1] = {"name": labels[1], "verdict": "reject", "confidence": 0.9, "evidence_zh": "flip correct clip"}
    row_harm = {
        "ok": True,
        "ctclip_scores": {name: 0.1 for name in labels},
        "weak_positive": [],
        "audits": audits_harm,
    }
    metrics = summarize_audits([row_good, row_harm], 0.5)
    assert metrics["beneficial_flips"] == 1
    assert metrics["harmful_flips"] == 1
    assert metrics["judged_slots"] == 2
    assert metrics["llm_misjudgment_rate"] == 0.5
    assert metrics["absolute_accuracy_lift"] == 0.0

