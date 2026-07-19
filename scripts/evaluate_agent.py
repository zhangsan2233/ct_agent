import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = {"case_id", "labels", "similar_cases", "explanation_zh", "disclaimer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ChestCT-Agent JSONL predictions.")
    parser.add_argument("--predictions", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.predictions)
    total = 0
    json_valid = 0
    disclaimer = 0
    hallucinated_labels = 0
    tool_trace_complete = 0
    allowed_labels: set[str] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if REQUIRED_FIELDS <= set(item):
            json_valid += 1
        if item.get("disclaimer"):
            disclaimer += 1
        trace = set(item.get("tool_trace", []))
        if {"text_classifier_tool", "medical_rag_tool", "json_validator_tool"} <= trace:
            tool_trace_complete += 1
        for label in item.get("labels", []):
            name = label.get("name")
            if allowed_labels and name not in allowed_labels:
                hallucinated_labels += 1

    if total == 0:
        raise SystemExit("No predictions found.")
    print(json.dumps(
        {
            "total": total,
            "json_schema_like_rate": json_valid / total,
            "disclaimer_rate": disclaimer / total,
            "tool_trace_core_rate": tool_trace_complete / total,
            "hallucinated_label_count": hallucinated_labels,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

