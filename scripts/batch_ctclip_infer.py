"""Run CT-CLIP once over a fixed manifest, keeping one model in GPU memory.

The output is JSONL intentionally: a completed case remains available if a later
volume is corrupt or a batch job is interrupted.  Re-running skips successful
case IDs unless --overwrite is supplied.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.ctclip import CtClipRuntime


DEFAULT_LABELS = [
    "arterial_wall_calcification", "atelectasis",
    "coronary_artery_wall_calcification", "emphysema", "lung_opacity",
    "lymphadenopathy", "pulmonary_fibrotic_sequela", "pulmonary_nodule",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--text-model-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not row.get("error") and row.get("case_id"):
            completed.add(str(row["case_id"]))
    return completed


def main() -> None:
    args = parse_args()
    os.environ["CTCLIP_TEXT_MODEL_DIR"] = str(args.text_model_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8", newline="")))
    labels = [name for name in DEFAULT_LABELS if name in rows[0]] if rows else DEFAULT_LABELS
    done = set() if args.overwrite else read_completed(args.out)
    mode = "w" if args.overwrite else "a"
    runtime = CtClipRuntime(args.checkpoint, args.source_dir, args.device, not args.no_fp16)
    started = time.time()
    succeeded = failed = skipped = 0
    with args.out.open(mode, encoding="utf-8", newline="") as handle:
        for index, row in enumerate(rows, start=1):
            case_id = row["case_id"]
            if case_id in done:
                skipped += 1
                continue
            result = {
                "case_id": case_id,
                "study_id": row.get("study_id", ""),
                "volume_name": row.get("volume_name", ""),
                "ct_volume_path": row["ct_volume_path"],
                "ground_truth": {label: int(row[label]) for label in labels},
                "report_impression": row.get("report_impression", ""),
                "probabilities": None,
                "seconds": None,
                "error": None,
            }
            begin = time.perf_counter()
            try:
                path = Path(result["ct_volume_path"])
                if not path.is_file():
                    raise FileNotFoundError(f"NIfTI not found: {path}")
                result["probabilities"] = runtime.predict(str(path))
                succeeded += 1
            except Exception as exc:  # Preserve failures and continue with the next patient.
                result["error"] = f"{type(exc).__name__}: {exc}"
                result["traceback"] = traceback.format_exc(limit=4)
                failed += 1
            result["seconds"] = round(time.perf_counter() - begin, 4)
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {case_id}: " + ("ok" if not result["error"] else result["error"]), flush=True)
    summary = {
        "manifest": str(args.manifest), "output": str(args.out), "cases_in_manifest": len(rows),
        "succeeded_this_run": succeeded, "failed_this_run": failed, "skipped_existing": skipped,
        "wall_seconds": round(time.time() - started, 3), "labels": labels,
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
