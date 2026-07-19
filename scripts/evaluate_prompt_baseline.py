import argparse
import asyncio
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.config import get_settings
from chestct_agent.evaluation import classification_metrics
from chestct_agent.labels import LABEL_IDS
from chestct_agent.llm import QwenClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen zero-shot report classification.")
    parser.add_argument("--manifest", default="artifacts/evaluation/multimodal_manifest.csv")
    parser.add_argument("--splits", default="artifacts/evaluation/patient_splits.csv")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--predictions", default="artifacts/evaluation/qwen_prompt_predictions.jsonl")
    parser.add_argument("--out", default="artifacts/evaluation/qwen_prompt_metrics.json")
    return parser.parse_args()


def existing_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            result[str(item["case_id"])] = item
    return result


def normalize_probabilities(value: dict, labels: list[str]) -> dict[str, float]:
    candidate = value.get("probabilities")
    if not isinstance(candidate, dict) and any(label in value for label in labels):
        candidate = value
    if not isinstance(candidate, dict) and isinstance(value.get("labels"), list):
        candidate = {
            str(item.get("name")): item.get("confidence", item.get("probability", 0.0))
            for item in value["labels"]
            if isinstance(item, dict) and item.get("name")
        }
    if not isinstance(candidate, dict):
        return {}
    result = {}
    for label in labels:
        try:
            result[label] = min(max(float(candidate.get(label, 0.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            result[label] = 0.0
    return result


async def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest).fillna("")
    splits = pd.read_csv(args.splits)
    frame = frame.merge(splits[["case_id", "patient_id", "evaluation_split"]], on="case_id")
    frame = frame[frame["evaluation_split"].eq("test")]
    frame = frame.sort_values("case_id").drop_duplicates("patient_id").head(args.limit)
    settings = get_settings()
    client = QwenClient(settings)
    if not client.is_configured:
        raise SystemExit("Qwen API is not configured; refusing to report fallback as prompt results.")
    prediction_path = Path(args.predictions)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_predictions(prediction_path)
    labels = list(LABEL_IDS)
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))

    async def classify(row: pd.Series) -> dict:
        case_id = str(row["case_id"])
        async with semaphore:
            fallback = {"probabilities": {label: 0.0 for label in labels}}
            call = await client.chat_json(
                system=(
                    "Classify a chest CT radiology report into exactly the supplied labels. "
                    "Respect negation and uncertainty. Return JSON only with a probabilities object "
                    "mapping every supplied label to a number from 0 to 1. Do not add labels."
                ),
                user=json.dumps(
                    {"labels": labels, "report": str(row["report_text"])}, ensure_ascii=False
                ),
                fallback=fallback,
            )
            probabilities = normalize_probabilities(call.value, labels)
            return {
                "case_id": case_id,
                "used_remote": call.used_remote,
                "fallback_reason": call.fallback_reason,
                "parse_valid": len(probabilities) == len(labels),
                "probabilities": probabilities,
                "raw_response": call.value,
            }

    pending = [
        asyncio.create_task(classify(row))
        for _, row in frame.iterrows()
        if str(row["case_id"]) not in completed
    ]
    with prediction_path.open("a", encoding="utf-8") as handle:
        for task in asyncio.as_completed(pending):
            item = await task
            case_id = str(item["case_id"])
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            completed[case_id] = item
            print(
                f"Completed {len(completed)}/{len(frame)}: {case_id}, "
                f"remote={item['used_remote']}"
            )

    y_true = np.zeros((len(frame), len(labels)), dtype=int)
    y_score = np.zeros_like(y_true, dtype=float)
    fallback_count = 0
    for row_index, (_, row) in enumerate(frame.iterrows()):
        positives = {label for label in str(row["labels"]).split(";") if label}
        item = completed[str(row["case_id"])]
        fallback_count += int(
            not item.get("used_remote", False) or not item.get("parse_valid", False)
        )
        probabilities = item.get("probabilities", {})
        for column, label in enumerate(labels):
            y_true[row_index, column] = int(label in positives)
            try:
                y_score[row_index, column] = min(max(float(probabilities.get(label, 0.0)), 0.0), 1.0)
            except (TypeError, ValueError):
                y_score[row_index, column] = 0.0
    metrics = classification_metrics(y_true, y_score, np.full(len(labels), 0.5), labels)
    result = {
        "model": settings.agent_model,
        "cases": len(frame),
        "patient_level_one_report_per_patient": True,
        "reference": "CT-RATE report-derived weak labels",
        "fallback_count": fallback_count,
        "metrics": metrics,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "metrics"}, indent=2))
    print({key: metrics[key] for key in ("micro_f1", "macro_f1", "macro_auroc", "macro_auprc")})
    print(f"Wrote {output}")


if __name__ == "__main__":
    asyncio.run(main())
