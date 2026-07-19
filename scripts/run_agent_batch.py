import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.agent.graph import ChestCtAgent
from chestct_agent.config import get_settings
from chestct_agent.schemas import AgentState, AnalyzeRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable ChestCT-Agent batch inference.")
    parser.add_argument(
        "--manifest",
        default="artifacts/evaluation/multimodal_manifest.csv",
    )
    parser.add_argument("--mode", choices=("report_only", "ct_only", "multimodal"), default="multimodal")
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--question", default="What abnormalities are present?")
    return parser.parse_args()


def _completed_cases(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("case_id") and not item.get("_error"):
            completed.add(str(item["case_id"]))
    return completed


async def run_batch(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest).fillna("")
    if args.limit > 0:
        manifest = manifest.head(args.limit)
    output_path = Path(args.out or f"artifacts/evaluation/{args.mode}_predictions.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.resume:
        output_path.write_text("", encoding="utf-8")
    completed = _completed_cases(output_path) if args.resume else set()

    settings = get_settings()
    if args.no_llm:
        settings = settings.model_copy(update={"openai_compatible_api_key": "replace-me"})
    agent = ChestCtAgent(settings)

    with output_path.open("a", encoding="utf-8") as handle:
        for index, row in manifest.iterrows():
            case_id = str(row["case_id"])
            if case_id in completed:
                continue
            report_text = str(row.get("report_text", "")) if args.mode != "ct_only" else ""
            ct_path = str(row.get("ct_volume_path", "")) if args.mode != "report_only" else ""
            started = time.perf_counter()
            try:
                request = AnalyzeRequest(
                    case_id=case_id,
                    report_text=report_text,
                    ct_volume_path=ct_path or None,
                    question=args.question,
                )
                response = await agent.run(AgentState(request=request))
                payload = response.model_dump(mode="json")
                payload["_evaluation_mode"] = args.mode
                payload["_wall_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            except Exception as exc:
                payload = {
                    "case_id": case_id,
                    "_evaluation_mode": args.mode,
                    "_error": f"{type(exc).__name__}: {exc}",
                }
                if args.fail_fast:
                    raise
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index + 1}/{len(manifest)}] {case_id}")

    print(f"Wrote predictions to {output_path}")


def main() -> None:
    asyncio.run(run_batch(parse_args()))


if __name__ == "__main__":
    main()
