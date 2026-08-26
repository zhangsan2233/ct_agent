"""Compare baseline vs candidate adapter metrics on a frozen evaluation set.

Promotion is rejected when micro-F1 drops or JSON/evidence completeness regresses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--modality", default="cxr_chest")
    args = parser.parse_args()
    baseline = load_metrics(args.baseline)
    candidate = load_metrics(args.candidate)
    base_f1 = float(baseline.get("micro_f1", baseline.get("micro_f1_vs_weak", 0.0)))
    cand_f1 = float(candidate.get("micro_f1", candidate.get("micro_f1_vs_weak", 0.0)))
    base_json = float(baseline.get("json_valid_rate", 1.0))
    cand_json = float(candidate.get("json_valid_rate", 1.0))
    base_evidence = float(baseline.get("evidence_match_rate", 1.0))
    cand_evidence = float(candidate.get("evidence_match_rate", 1.0))
    f1_ok = cand_f1 >= base_f1
    json_ok = cand_json >= base_json
    evidence_ok = cand_evidence >= base_evidence
    report = {
        "modality": args.modality,
        "baseline_micro_f1": base_f1,
        "candidate_micro_f1": cand_f1,
        "micro_f1_delta": round(cand_f1 - base_f1, 6),
        "baseline_json_valid_rate": base_json,
        "candidate_json_valid_rate": cand_json,
        "baseline_evidence_match_rate": base_evidence,
        "candidate_evidence_match_rate": cand_evidence,
        "promotion_allowed": f1_ok and json_ok and evidence_ok,
        "reject_reasons": [
            reason
            for reason, ok in (
                ("micro_f1_regression", f1_ok),
                ("json_valid_rate_regression", json_ok),
                ("evidence_match_regression", evidence_ok),
            )
            if not ok
        ],
        "notice": "Human approval still required even when promotion_allowed is true.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["promotion_allowed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
