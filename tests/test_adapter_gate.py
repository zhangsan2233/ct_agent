from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def _scratch() -> Path:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="gate_", dir=str(artifacts)))


def test_adapter_gate_rejects_f1_regression():
    tmp_path = _scratch()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    out = tmp_path / "gate.json"
    baseline.write_text(json.dumps({"micro_f1": 0.8, "json_valid_rate": 1.0, "evidence_match_rate": 1.0}))
    candidate.write_text(json.dumps({"micro_f1": 0.75, "json_valid_rate": 1.0, "evidence_match_rate": 1.0}))
    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_modality_adapter_gate.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--baseline", str(baseline), "--candidate", str(candidate), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["promotion_allowed"] is False
    assert "micro_f1_regression" in report["reject_reasons"]
