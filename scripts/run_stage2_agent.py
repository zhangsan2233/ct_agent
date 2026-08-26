"""Unified single-case and batch CLI for the frozen CT-CLIP + Stage-2 agent."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.stage2_pipeline import Stage2Agent, Stage2Paths


def add_model_args(parser: argparse.ArgumentParser) -> None:
    defaults = Stage2Paths.defaults(ROOT)
    parser.add_argument("--model-dir", type=Path, default=defaults.model_dir)
    parser.add_argument("--adapter-dir", type=Path, default=defaults.adapter_dir)
    parser.add_argument("--ctclip-checkpoint", type=Path, default=defaults.ctclip_checkpoint)
    parser.add_argument("--ctclip-source", type=Path, default=defaults.ctclip_source)
    parser.add_argument("--text-model-dir", type=Path, default=defaults.text_model_dir)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--llm-2d-review",
        action="store_true",
        help="Enable experimental base-model 2D slice review alongside frozen CT-CLIP (does not change primary JSON).",
    )


def agent_from(args: argparse.Namespace) -> Stage2Agent:
    paths = Stage2Paths(args.model_dir, args.adapter_dir, args.ctclip_checkpoint, args.ctclip_source, args.text_model_dir)
    return Stage2Agent(paths, args.device, args.max_new_tokens)


def read_report(args: argparse.Namespace) -> str:
    if bool(args.report) == bool(args.report_file):
        raise SystemExit("Supply exactly one of --report or --report-file.")
    return args.report if args.report is not None else args.report_file.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("single", help="Run one CT and report")
    single.add_argument("--ct", required=True, type=Path)
    single.add_argument("--case-id", required=True)
    single.add_argument("--report")
    single.add_argument("--report-file", type=Path)
    single.add_argument("--runs-dir", type=Path, default=ROOT / "artifacts" / "agent_runs")
    add_model_args(single)
    batch = commands.add_parser("batch", help="Run CSV rows with case_id,ct_path and report_text or report_path")
    batch.add_argument("--manifest", required=True, type=Path)
    batch.add_argument("--runs-dir", type=Path, default=ROOT / "artifacts" / "agent_runs")
    batch.add_argument("--out", type=Path, default=ROOT / "artifacts" / "agent_runs" / "batch_results.jsonl")
    add_model_args(batch)
    args = parser.parse_args()
    agent = agent_from(args)
    if args.command == "single":
        run_dir = args.runs_dir / args.case_id
        result = agent.analyze(
            case_id=args.case_id,
            ct_path=args.ct,
            report_text=read_report(args),
            run_dir=run_dir,
            enable_llm_2d_review=args.llm_2d_review,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    with args.out.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, start=1):
            case_id = row.get("case_id", "").strip()
            try:
                report = row.get("report_text", "").strip()
                if not report and row.get("report_path"):
                    report = Path(row["report_path"]).read_text(encoding="utf-8")
                result = agent.analyze(
                    case_id=case_id,
                    ct_path=Path(row["ct_path"]),
                    report_text=report,
                    run_dir=args.runs_dir / case_id,
                    enable_llm_2d_review=args.llm_2d_review,
                )
                record = {"case_id": case_id, "ok": True, "result": result}
            except Exception as exc:
                record = {"case_id": case_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{index}/{len(rows)}] {case_id}: {'ok' if record['ok'] else record['error']}", flush=True)


if __name__ == "__main__":
    main()
