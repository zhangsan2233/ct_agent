from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from chestct_agent.config import get_settings
from chestct_agent.tools.totalseg_diagnostics import TotalSegmentatorDiagnosticTool


DEFAULT_CASES = [
    "valid_1_a_1",
    "valid_4_a_1",
    "valid_5_a_1",
    "valid_6_a_1",
    "valid_10_a_1",
    "valid_11_a_1",
    "valid_14_a_1",
    "valid_22_a_1",
]
LABELS = [
    "pulmonary_nodule",
    "pleural_effusion",
    "pericardial_effusion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate independent CT specialist tools.")
    parser.add_argument("--manifest", default="artifacts/evaluation/multimodal_manifest.csv")
    parser.add_argument("--ctclip", default="artifacts/evaluation/ctclip_predictions.jsonl")
    parser.add_argument("--out", default="artifacts/evaluation/specialist_tool_metrics.json")
    parser.add_argument("--records", default="artifacts/evaluation/specialist_tool_records.jsonl")
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[str(item["case_id"])] = item
    return rows


def binary_metrics(truth: list[int], predicted: list[int]) -> dict[str, float]:
    return {
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "accuracy": float(sum(a == b for a, b in zip(truth, predicted, strict=True)) / len(truth)),
    }


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest).fillna("").set_index("case_id")
    ctclip = load_jsonl(Path(args.ctclip))
    tool = TotalSegmentatorDiagnosticTool(get_settings())
    records_path = Path(args.records)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    for index, case_id in enumerate(args.cases, start=1):
        if case_id not in manifest.index or case_id not in ctclip:
            print(f"[{index}/{len(args.cases)}] skip {case_id}: missing manifest or CT-CLIP row", flush=True)
            continue
        truth = {label for label in str(manifest.loc[case_id, "labels"]).split(";") if label}
        volume_path = ctclip[case_id]["ct_volume_path"]
        started = time.perf_counter()
        evidence, warnings, latency_ms = tool.analyze(case_id, volume_path, set(LABELS))
        elapsed = time.perf_counter() - started
        by_label = {item.label: item for item in evidence}
        for label in LABELS:
            item = by_label.get(label)
            records.append(
                {
                    "case_id": case_id,
                    "label": label,
                    "truth": int(label in truth),
                    "ctclip_score": float(ctclip[case_id]["probabilities"].get(label, 0.0)),
                    "tool_verdict": item.verdict if item else "unavailable",
                    "tool_confidence": float(item.confidence) if item else 0.0,
                    "metrics": item.metrics if item else {},
                    "tool_latency_ms": latency_ms,
                    "warnings": warnings,
                }
            )
        records_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
            encoding="utf-8",
        )
        verdicts = ", ".join(f"{label}={by_label[label].verdict}" for label in by_label)
        print(f"[{index}/{len(args.cases)}] {case_id} {elapsed:.1f}s | {verdicts}", flush=True)

    summary: dict[str, object] = {
        "cases": sorted({row["case_id"] for row in records}),
        "case_count": len({row["case_id"] for row in records}),
        "reference": "CT-RATE report-derived labels; these are weak clinical references, not pathology ground truth.",
        "per_label": {},
    }
    for label in LABELS:
        rows = [row for row in records if row["label"] == label]
        truth = [row["truth"] for row in rows]
        baseline = [int(row["ctclip_score"] >= 0.5) for row in rows]
        selective = [row for row in rows if row["tool_verdict"] in {"positive", "negative"}]
        tool_truth = [row["truth"] for row in selective]
        tool_pred = [int(row["tool_verdict"] == "positive") for row in selective]
        summary["per_label"][label] = {
            "positive_references": int(sum(truth)),
            "ctclip_at_0_5": binary_metrics(truth, baseline),
            "tool_selective": {
                **(binary_metrics(tool_truth, tool_pred) if selective else {}),
                "coverage": len(selective) / len(rows) if rows else 0.0,
                "decisions": len(selective),
                "uncertain_or_unavailable": len(rows) - len(selective),
            },
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
