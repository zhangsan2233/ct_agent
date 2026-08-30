"""Compare Agent+tools with Agent+tools+Memory on one frozen patient cohort."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.memory_gate import gate_memory_change, load_tool_thresholds


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[str(item["case_id"])] = item
    return rows


def metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "label_accuracy": float(np.mean(truth == predicted)),
        "wrong_label_decisions": int(np.sum(truth != predicted)),
        "micro_precision": float(
            precision_score(truth, predicted, average="micro", zero_division=0)
        ),
        "micro_recall": float(recall_score(truth, predicted, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(truth, predicted, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "exact_match_cases": int(np.sum(np.all(truth == predicted, axis=1))),
    }


def status_map(value: dict | list) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(label): str(item["status"]) for label, item in value.items()}
    return {str(item["label"]): str(item["status"]) for item in value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--memory-rechecks", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline_predictions)
    rechecks = load_jsonl(args.memory_rechecks)
    thresholds = load_tool_thresholds(args.calibration)
    case_ids = sorted(set(baseline) & set(rechecks))
    if not case_ids:
        raise SystemExit("No overlapping baseline and Memory cases.")

    accepted = beneficial = harmful = 0
    final_by_case = {}
    for case_id in case_ids:
        base = baseline[case_id]
        review = copy.deepcopy(rechecks[case_id])
        initial = status_map(review["initial"])
        final = status_map(review["final"])
        tool_scores = {
            str(item["label"]): float(item["ctclip_score"])
            for item in base.get("tool_packet", {}).get("findings", [])
        }
        for change in review.get("changes", []):
            if not change.get("accepted"):
                continue
            label = str(change["label"])
            decision = gate_memory_change(
                before_status=initial[label],
                proposed_status=str(change.get("proposed") or change.get("final")),
                confidence=float(change.get("confidence", 0.0)),
                memory_ids=[str(value) for value in change.get("memory_ids_used", [])],
                supporting_slice_indices=[
                    int(value) for value in change.get("supporting_slice_indices", [])
                ],
                visible_evidence=str(change.get("visible_evidence") or ""),
                tool_score=tool_scores.get(label),
                tool_threshold=thresholds.get(label),
            )
            final[label] = decision.final_status
            if not decision.accepted:
                continue
            accepted += 1
            truth = int(review["truth"][label])
            before = int(initial[label] == "positive")
            after = int(final[label] == "positive")
            beneficial += int(before != truth and after == truth)
            harmful += int(before == truth and after != truth)
        final_by_case[case_id] = final

    truth = np.asarray(
        [[int(rechecks[case_id]["truth"][label]) for label in args.labels] for case_id in case_ids]
    )
    baseline_matrix = np.asarray(
        [
            [
                int(status_map(rechecks[case_id]["initial"])[label] == "positive")
                for label in args.labels
            ]
            for case_id in case_ids
        ]
    )
    memory_matrix = np.asarray(
        [
            [int(final_by_case[case_id][label] == "positive") for label in args.labels]
            for case_id in case_ids
        ]
    )
    baseline_metrics = metrics(truth, baseline_matrix)
    memory_metrics = metrics(truth, memory_matrix)
    output = {
        "cases": len(case_ids),
        "labels": args.labels,
        "baseline": baseline_metrics,
        "with_memory": memory_metrics,
        "delta": {
            key: float(memory_metrics[key]) - float(baseline_metrics[key])
            for key in ("label_accuracy", "micro_precision", "micro_recall", "micro_f1", "macro_f1")
        },
        "accepted_changes": accepted,
        "beneficial_changes": beneficial,
        "harmful_changes": harmful,
        "calibration": str(args.calibration),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
