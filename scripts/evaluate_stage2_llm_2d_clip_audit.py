"""Audit frozen CT-CLIP binary calls with 2D Qwen (scores + slices, no independent diagnosis).

LLM verdicts are confirm / reject / insufficient_coverage. Reject flips the CT-CLIP
binary call; otherwise the CT-CLIP call is kept. Weak labels are report-derived,
not CT gold. Do not present these numbers as clinical accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.stage2_llm_2d_review import (
    BaseQwenVisionRuntime,
    apply_audit_to_clip,
    render_stage2_review_slices,
    run_clip_audits,
)
from chestct_agent.stage2_pipeline import LABELS

NOTICE = (
    "LLM audits frozen CT-CLIP using 2D slices. Reference labels are report-derived weak labels, "
    "not radiologist-adjudicated CT ground truth."
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
        cases.append({"case_id": case_id, "ctclip_scores": scores, "weak_positive": sorted(weak_positive)})
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


def summarize_audits(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    ok_cases = sum(1 for row in rows if row.get("ok") is True)
    failed_cases = sum(1 for row in rows if row.get("ok") is not True)
    n_slots = ok_cases * len(LABELS)
    verdicts = Counter()
    judged = 0
    misjudged = 0
    beneficial_flip = 0
    harmful_flip = 0
    wasted_confirm_error = 0
    correct_confirm = 0
    clip_correct = 0
    corrected_correct = 0
    per_label: dict[str, dict[str, Any]] = {
        name: Counter(
            {
                "confirm": 0,
                "reject": 0,
                "insufficient_coverage": 0,
                "misjudged": 0,
                "judged": 0,
                "beneficial_flip": 0,
                "harmful_flip": 0,
                "clip_correct": 0,
                "corrected_correct": 0,
            }
        )
        for name in LABELS
    }

    for row in rows:
        if row.get("ok") is not True:
            continue
        scores = row["ctclip_scores"]
        weak = set(row.get("weak_positive") or [])
        audits = {item["name"]: item for item in row["audits"]}
        corrected = apply_audit_to_clip(row["audits"], scores, positive_threshold=threshold)
        for name in LABELS:
            clip_pos = float(scores.get(name, 0.0)) >= threshold
            weak_pos = name in weak
            verdict = audits.get(name, {}).get("verdict", "insufficient_coverage")
            if verdict not in {"confirm", "reject", "insufficient_coverage"}:
                verdict = "insufficient_coverage"
            verdicts[verdict] += 1
            stats = per_label[name]
            stats[verdict] += 1
            clip_ok = clip_pos == weak_pos
            corr_ok = corrected[name] == weak_pos
            stats["clip_correct"] += int(clip_ok)
            stats["corrected_correct"] += int(corr_ok)
            clip_correct += int(clip_ok)
            corrected_correct += int(corr_ok)
            if verdict == "insufficient_coverage":
                continue
            judged += 1
            stats["judged"] += 1
            if verdict == "confirm":
                if clip_ok:
                    correct_confirm += 1
                else:
                    misjudged += 1
                    wasted_confirm_error += 1
                    stats["misjudged"] += 1
            elif verdict == "reject":
                if clip_ok:
                    misjudged += 1
                    harmful_flip += 1
                    stats["misjudged"] += 1
                    stats["harmful_flip"] += 1
                else:
                    beneficial_flip += 1
                    stats["beneficial_flip"] += 1

    clip_acc = (clip_correct / n_slots) if n_slots else None
    corr_acc = (corrected_correct / n_slots) if n_slots else None
    clip_err = (1 - clip_acc) if clip_acc is not None else None
    corr_err = (1 - corr_acc) if corr_acc is not None else None
    relative_error_reduction = None
    if clip_err not in (None, 0):
        relative_error_reduction = (clip_err - corr_err) / clip_err
    per_label_out = {}
    for name, stats in per_label.items():
        n_judged = int(stats["judged"])
        per_label_out[name] = {
            "verdict_counts": {
                "confirm": int(stats["confirm"]),
                "reject": int(stats["reject"]),
                "insufficient_coverage": int(stats["insufficient_coverage"]),
            },
            "insufficient_rate": (int(stats["insufficient_coverage"]) / ok_cases) if ok_cases else 0.0,
            "judged": n_judged,
            "misjudgment_rate": (int(stats["misjudged"]) / n_judged) if n_judged else None,
            "beneficial_flips": int(stats["beneficial_flip"]),
            "harmful_flips": int(stats["harmful_flip"]),
            "clip_accuracy": (int(stats["clip_correct"]) / ok_cases) if ok_cases else None,
            "corrected_accuracy": (int(stats["corrected_correct"]) / ok_cases) if ok_cases else None,
        }
    return {
        "notice": NOTICE,
        "evaluated_cases": ok_cases,
        "failed_cases": failed_cases,
        "label_slots": n_slots,
        "verdict_counts": dict(verdicts),
        "insufficient_rate": (verdicts["insufficient_coverage"] / n_slots) if n_slots else 0.0,
        "judged_slots": judged,
        "coverage_rate": (judged / n_slots) if n_slots else 0.0,
        "llm_misjudgment_rate": (misjudged / judged) if judged else None,
        "confirm_correct": correct_confirm,
        "confirm_when_clip_wrong": wasted_confirm_error,
        "beneficial_flips": beneficial_flip,
        "harmful_flips": harmful_flip,
        "net_flips": beneficial_flip - harmful_flip,
        "clip_accuracy_vs_weak": clip_acc,
        "corrected_accuracy_vs_weak": corr_acc,
        "absolute_accuracy_lift": (corr_acc - clip_acc) if clip_acc is not None and corr_acc is not None else None,
        "relative_error_reduction": relative_error_reduction,
        "per_label": per_label_out,
    }


def run_eval(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cases = load_cases(args.inputs)
    if args.limit:
        cases = cases[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.out_dir / "audits.jsonl"
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
                    audits = run_clip_audits(
                        runtime,
                        slice_paths,
                        case["ctclip_scores"],
                        positive_threshold=args.threshold,
                    )
                    record.update(
                        {
                            "ok": True,
                            "slice_count": len(slice_paths),
                            "audits": audits,
                            "corrected_binary": apply_audit_to_clip(
                                audits, case["ctclip_scores"], positive_threshold=args.threshold
                            ),
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
    metrics = summarize_audits(all_rows, args.threshold)
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
        default=ROOT / "artifacts" / "llm_eval" / "stage2_llm_2d_clip_audit_100",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        rows = list(load_done(args.out_dir / "audits.jsonl").values())
        print(json.dumps(summarize_audits(rows, args.threshold), ensure_ascii=False, indent=2))
        return
    run_eval(args)


if __name__ == "__main__":
    main()
