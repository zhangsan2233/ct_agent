"""Batch-compare frozen CT-CLIP scores with independent 2D Qwen votes.

This script does not replace Stage-2 JSON. It reuses cached CT-CLIP scores from
the 100-case Stage-2 input JSONL, runs base-model vision votes only, and writes
aggregate agreement metrics. Weak labels are report-derived, not CT gold.
Do not present the numbers as clinical accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.stage2_llm_2d_review import (
    BaseQwenVisionRuntime,
    build_agreement,
    render_stage2_review_slices,
    run_independent_votes,
)
from chestct_agent.stage2_pipeline import LABELS

NOTICE = (
    "Agreement is against frozen CT-CLIP scores and report-derived weak labels. "
    "It is not radiologist-adjudicated CT ground truth or clinical performance."
)


def locate_ct(data_root: Path, case_id: str) -> Path | None:
    direct = list(data_root.glob(f"**/{case_id}.nii.gz"))
    if direct:
        return direct[0]
    parts = case_id.split("_")
    if len(parts) >= 2 and parts[0] == "train":
        patient = "_".join(parts[:2])
        nested = data_root / patient
        hits = list(nested.glob(f"**/{case_id}.nii.gz")) if nested.exists() else []
        if hits:
            return hits[0]
    return None


def load_cases(jsonl_path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        payload = json.loads(row["messages"][1]["content"])
        target = json.loads(row["messages"][2]["content"])
        case_id = str(payload.get("case_id") or row.get("metadata", {}).get("case_id") or "")
        scores = {name: float(payload["ctclip_scores"][name]) for name in LABELS}
        weak_positive = {
            item["name"]
            for item in target.get("labels", [])
            if isinstance(item, dict) and item.get("status") == "positive" and item.get("name") in LABELS
        }
        cases.append(
            {
                "case_id": case_id,
                "ctclip_scores": scores,
                "weak_positive": sorted(weak_positive),
            }
        )
    return cases


def load_done(out_jsonl: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not out_jsonl.exists():
        return done
    for line in out_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = row.get("case_id")
        if case_id:
            done[case_id] = row
    return done


def binary_kappa(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pred_pos = sum(a for a, _ in pairs) / n
    true_pos = sum(b for _, b in pairs) / n
    pe = pred_pos * true_pos + (1 - pred_pos) * (1 - true_pos)
    if pe >= 1:
        return 1.0 if po >= 1 else 0.0
    return (po - pe) / (1 - pe)


def chance_agreement(llm_pos_rate: float, ct_pos_rate: float) -> float:
    return llm_pos_rate * ct_pos_rate + (1 - llm_pos_rate) * (1 - ct_pos_rate)


def summarize(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    vote_counts = Counter()
    comparable_pairs: list[tuple[int, int]] = []
    llm_vs_weak: list[tuple[int, int]] = []
    ct_vs_weak: list[tuple[int, int]] = []
    per_label = {
        name: {
            "votes": Counter(),
            "comparable": 0,
            "agree_ctclip": 0,
            "llm_visible_ctclip_low": 0,
            "llm_not_visible_ctclip_high": 0,
            "insufficient": 0,
            "ct_pos": 0,
            "weak_pos": 0,
            "llm_visible": 0,
        }
        for name in LABELS
    }
    ok_cases = 0
    failed_cases = 0
    shuffled_pairs: list[tuple[int, int]] = []
    rng = random.Random(20260823)
    llm_bits: list[int] = []
    ct_bits: list[int] = []

    for row in rows:
        if row.get("ok") is not True:
            failed_cases += 1
            continue
        ok_cases += 1
        scores = row["ctclip_scores"]
        votes = {item["name"]: item for item in row["votes"]}
        weak = set(row.get("weak_positive") or [])
        for name in LABELS:
            vote = votes.get(name, {}).get("vote", "insufficient_coverage")
            vote_counts[vote] += 1
            stats = per_label[name]
            stats["votes"][vote] += 1
            ct_pos = float(scores.get(name, 0.0)) >= threshold
            weak_pos = name in weak
            stats["ct_pos"] += int(ct_pos)
            stats["weak_pos"] += int(weak_pos)
            ct_vs_weak.append((int(ct_pos), int(weak_pos)))
            if vote == "insufficient_coverage":
                stats["insufficient"] += 1
                continue
            llm_pos = vote == "visible"
            stats["llm_visible"] += int(llm_pos)
            stats["comparable"] += 1
            agree = llm_pos == ct_pos
            stats["agree_ctclip"] += int(agree)
            if llm_pos and not ct_pos:
                stats["llm_visible_ctclip_low"] += 1
            if (not llm_pos) and ct_pos:
                stats["llm_not_visible_ctclip_high"] += 1
            comparable_pairs.append((int(llm_pos), int(ct_pos)))
            llm_vs_weak.append((int(llm_pos), int(weak_pos)))
            llm_bits.append(int(llm_pos))
            ct_bits.append(int(ct_pos))

    if llm_bits:
        shuffled = llm_bits[:]
        rng.shuffle(shuffled)
        shuffled_pairs = list(zip(shuffled, ct_bits))

    n_slots = ok_cases * len(LABELS)
    comparable = len(comparable_pairs)
    agree = sum(a == b for a, b in comparable_pairs)
    llm_pos_rate = (sum(a for a, _ in comparable_pairs) / comparable) if comparable else 0.0
    ct_pos_rate = (sum(b for _, b in comparable_pairs) / comparable) if comparable else 0.0
    per_label_out = {}
    for name, stats in per_label.items():
        n = stats["comparable"]
        per_label_out[name] = {
            "vote_counts": dict(stats["votes"]),
            "insufficient_rate": (stats["insufficient"] / ok_cases) if ok_cases else 0.0,
            "comparable": n,
            "agreement_with_ctclip": (stats["agree_ctclip"] / n) if n else None,
            "llm_visible_ctclip_low": stats["llm_visible_ctclip_low"],
            "llm_not_visible_ctclip_high": stats["llm_not_visible_ctclip_high"],
            "ctclip_positive_rate": (stats["ct_pos"] / ok_cases) if ok_cases else 0.0,
            "weak_label_positive_rate": (stats["weak_pos"] / ok_cases) if ok_cases else 0.0,
            "llm_visible_rate_among_comparable": (stats["llm_visible"] / n) if n else None,
        }
    return {
        "notice": NOTICE,
        "evaluated_cases": ok_cases,
        "failed_cases": failed_cases,
        "label_slots": n_slots,
        "vote_counts": dict(vote_counts),
        "insufficient_rate": (vote_counts["insufficient_coverage"] / n_slots) if n_slots else 0.0,
        "comparable_slots": comparable,
        "coverage_rate": (comparable / n_slots) if n_slots else 0.0,
        "llm_vs_ctclip": {
            "agreement": (agree / comparable) if comparable else None,
            "cohen_kappa": binary_kappa(comparable_pairs),
            "llm_positive_rate": llm_pos_rate,
            "ctclip_positive_rate": ct_pos_rate,
            "chance_agreement_independent": chance_agreement(llm_pos_rate, ct_pos_rate),
            "shuffled_llm_agreement": (
                sum(a == b for a, b in shuffled_pairs) / len(shuffled_pairs) if shuffled_pairs else None
            ),
            "majority_baseline_agreement": max(ct_pos_rate, 1 - ct_pos_rate) if comparable else None,
        },
        "llm_vs_weak_labels": {
            "agreement": (
                sum(a == b for a, b in llm_vs_weak) / len(llm_vs_weak) if llm_vs_weak else None
            ),
            "cohen_kappa": binary_kappa(llm_vs_weak),
        },
        "ctclip_vs_weak_labels": {
            "agreement": (
                sum(a == b for a, b in ct_vs_weak) / len(ct_vs_weak) if ct_vs_weak else None
            ),
            "cohen_kappa": binary_kappa(ct_vs_weak),
        },
        "per_label": per_label_out,
        "interpretation_keys": {
            "kappa_near_zero": "LLM votes are consistent with chance relative to CT-CLIP.",
            "agreement_below_majority_baseline": "Always copying the CT-CLIP majority class would beat the LLM.",
            "high_insufficient_rate": "LLM often abstains; 2D slices are weakly informative.",
            "llm_weak_kappa_near_zero": "2D votes do not track report-derived weak labels.",
        },
    }


def run_eval(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cases = load_cases(args.inputs)
    if args.limit:
        cases = cases[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.out_dir / "votes.jsonl"
    done = load_done(out_jsonl)
    pending = [case for case in cases if case["case_id"] not in done]
    print(f"loaded={len(cases)} already_done={len(done)} pending={len(pending)}", flush=True)
    runtime = BaseQwenVisionRuntime(model_dir=args.model_dir, device=args.device)
    if pending:
        runtime.load()
    try:
        with out_jsonl.open("a", encoding="utf-8") as handle:
            for index, case in enumerate(pending, start=1):
                case_id = case["case_id"]
                started = time.perf_counter()
                ct_path = locate_ct(args.data_root, case_id)
                record: dict[str, Any] = {
                    "case_id": case_id,
                    "ctclip_scores": case["ctclip_scores"],
                    "weak_positive": case["weak_positive"],
                }
                try:
                    if ct_path is None:
                        raise FileNotFoundError(f"CT volume not found for {case_id}")
                    slice_dir = args.out_dir / "slices" / case_id
                    slice_paths = render_stage2_review_slices(
                        case_id, ct_path, slice_dir, max_images=args.max_images
                    )
                    votes = run_independent_votes(runtime, slice_paths)
                    agreement = build_agreement(votes, case["ctclip_scores"], positive_threshold=args.threshold)
                    record.update(
                        {
                            "ok": True,
                            "ct_found": True,
                            "slice_count": len(slice_paths),
                            "votes": votes,
                            "agreement": agreement,
                            "elapsed_seconds": round(time.perf_counter() - started, 3),
                        }
                    )
                except Exception as exc:
                    record.update(
                        {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "elapsed_seconds": round(time.perf_counter() - started, 3),
                        }
                    )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                status = "ok" if record.get("ok") else record.get("error")
                print(f"[{index}/{len(pending)}] {case_id}: {status}", flush=True)
    finally:
        runtime.release()
    all_rows = list(load_done(out_jsonl).values())
    selected_ids = {case["case_id"] for case in cases}
    all_rows = [row for row in all_rows if row.get("case_id") in selected_ids]
    metrics = summarize(all_rows, args.threshold)
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {metrics_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=ROOT / "artifacts" / "llm_eval" / "stage2_demo_100" / "inputs" / "with_ctclip_100.jsonl",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/root/summer_zhl/data/train_fixed"))
    parser.add_argument("--model-dir", type=Path, default=Path("/root/summer_zhl/models/Qwen3.5-9B"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "llm_eval" / "stage2_llm_2d_agreement_100",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        rows = list(load_done(args.out_dir / "votes.jsonl").values())
        metrics = summarize(rows, args.threshold)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    run_eval(args)


if __name__ == "__main__":
    main()
