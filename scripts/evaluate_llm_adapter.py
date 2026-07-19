"""Generate structured outputs from a QLoRA adapter and score JSON/label validity.

The reference labels in the SFT file are report-derived weak labels, not CT
ground truth.  This script therefore evaluates SFT-format adherence and weak
label agreement only; it must not be presented as diagnostic accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


LABEL_ALIASES = {"lung_nodule": "pulmonary_nodule"}


def parse_json(text: str) -> tuple[dict | None, str | None]:
    """Extract the first JSON object, tolerating a Markdown code fence."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        candidate = candidate.rsplit("```", 1)[0].strip()
    start = candidate.find("{")
    if start < 0:
        return None, "no_json_object"
    try:
        result, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as error:
        return None, f"json_decode_error:{error.msg}"
    return result if isinstance(result, dict) else None, None


def reference_positive_labels(example: dict) -> set[str]:
    """Read either an SFT target or an explicit evaluation reference."""
    if "ground_truth" in example:
        return {name for name, value in example["ground_truth"].items() if int(value) == 1}
    target = json.loads(example["messages"][2]["content"])
    return {item["name"] for item in target["labels"] if item.get("status") == "positive"}


def predicted_positive_labels(result: dict) -> tuple[set[str], str | None]:
    labels = result.get("labels")
    if not isinstance(labels, list):
        return set(), "labels_missing_or_not_list"
    names: set[str] = set()
    for item in labels:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return set(), "invalid_label_item"
        if item.get("status") == "positive":
            names.add(LABEL_ALIASES.get(item["name"], item["name"]))
    return names, None


def input_ct_scores(messages: list[dict]) -> dict[str, float] | None:
    """Read CT evidence from either the Stage-1 or compact Stage-2 schema."""
    try:
        payload = json.loads(messages[1]["content"])
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        return None
    scores = payload.get("ctclip_scores", payload.get("ct_model_scores"))
    if not isinstance(scores, dict):
        return None
    return {LABEL_ALIASES.get(str(name), str(name)): float(value) for name, value in scores.items()}


def ct_evidence_stats(result: dict, scores: dict[str, float] | None, labels: list[str]) -> Counter:
    stats: Counter = Counter()
    if not scores:
        return stats
    stats["ct_score_slots_expected"] += len(labels)
    output_labels = result.get("labels")
    if not isinstance(output_labels, list):
        return stats
    by_name = {
        LABEL_ALIASES.get(item.get("name"), item.get("name")): item
        for item in output_labels if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    for label in labels:
        item = by_name.get(label)
        if item is None:
            continue
        value = item.get("ctclip_score")
        if value is None and isinstance(item.get("source_scores"), dict):
            value = item["source_scores"].get("ct_model")
        if not isinstance(value, (int, float)):
            continue
        stats["ct_score_slots_present"] += 1
        if label in scores and abs(float(value) - scores[label]) <= 1e-4:
            stats["ct_score_slots_matched"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--valid-file", default="artifacts/llm_sft/valid.jsonl", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not args.model_path.is_dir() or not args.adapter_path.is_dir() or not args.valid_file.is_file():
        raise SystemExit("Model, adapter, or validation file is missing.")
    if args.limit < 1 or args.generation_batch_size < 1:
        raise SystemExit("--limit and --generation-batch-size must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
                              StoppingCriteria, StoppingCriteriaList)

    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_use_double_quant=True)
    base = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=True,
        quantization_config=quantization, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(args.adapter_path), local_files_only=True)
    model.eval()

    class CompleteJsonStoppingCriteria(StoppingCriteria):
        """Stop greedy decoding once a structurally usable JSON response closes."""

        def __init__(self, prompt_tokens: int) -> None:
            self.prompt_tokens = prompt_tokens

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            generated_tokens = input_ids.shape[1] - self.prompt_tokens
            if generated_tokens < 32:
                return False
            # A complete JSON object can only end at a closing brace.  Checking
            # only there avoids repeatedly decoding an ever-growing response.
            if "}" not in tokenizer.decode(input_ids[0, -1:], skip_special_tokens=True):
                return False
            candidate = tokenizer.decode(input_ids[0, self.prompt_tokens:], skip_special_tokens=True)
            parsed, _ = parse_json(candidate)
            return parsed is not None and predicted_positive_labels(parsed)[1] is None

    examples = [json.loads(line) for line in args.valid_file.read_text(encoding="utf-8").splitlines()[:args.limit]]
    labels = sorted({label for example in examples for label in reference_positive_labels(example)} |
                    {label for example in examples for label in example.get("evaluation_labels", [])})
    totals: dict[str, Counter] = {label: Counter() for label in labels}
    errors: Counter = Counter()
    ct_totals: Counter = Counter()
    records: list[dict] = []
    for batch_start in range(0, len(examples), args.generation_batch_size):
        batch = examples[batch_start:batch_start + args.generation_batch_size]
        batch_messages = [example["messages"][:2] for example in batch]
        # Qwen3.5 defaults to a long hidden/reasoning-style response.  This
        # task is constrained to machine-readable JSON, so disable it at the
        # template level rather than wasting the output budget on reasoning.
        prompts = [processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ) for messages in batch_messages]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        generation_args = {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                           "pad_token_id": tokenizer.pad_token_id}
        # StoppingCriteria stops an entire batch when any sequence is complete.
        # Use it only for single-record interactive generation; fixed-budget
        # batched decoding is deterministic and much faster for full evaluation.
        if len(batch) == 1:
            generation_args["stopping_criteria"] = StoppingCriteriaList(
                [CompleteJsonStoppingCriteria(inputs["input_ids"].shape[1])]
            )
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_args)
        prompt_width = inputs["input_ids"].shape[1]
        for offset, (example, messages) in enumerate(zip(batch, batch_messages, strict=True), start=1):
            index = batch_start + offset
            completion = tokenizer.decode(generated[offset - 1, prompt_width:], skip_special_tokens=True)
            parsed, error = parse_json(completion)
            case_id = json.loads(messages[1]["content"])["case_id"]
            item: dict = {"case_id": case_id, "raw_completion": completion, "json_valid": parsed is not None}
            if parsed is None:
                errors[error or "unknown"] += 1
                item["error"] = error
            else:
                predicted, structure_error = predicted_positive_labels(parsed)
                if structure_error:
                    errors[structure_error] += 1
                    item["error"] = structure_error
                truth = reference_positive_labels(example)
                item["prediction"] = parsed
                item["positive_labels_predicted"] = sorted(predicted)
                item["positive_labels_reference"] = sorted(truth)
                if not structure_error:
                    for label in labels:
                        if label in predicted and label in truth:
                            totals[label]["tp"] += 1
                        elif label in predicted:
                            totals[label]["fp"] += 1
                        elif label in truth:
                            totals[label]["fn"] += 1
                        else:
                            totals[label]["tn"] += 1
                    ct_totals.update(ct_evidence_stats(parsed, input_ct_scores(messages), labels))
            records.append(item)
            print(f"[{index}/{len(examples)}] {case_id}: json_valid={item['json_valid']}", flush=True)

    per_label = {}
    micro = Counter()
    for label, counts in totals.items():
        micro.update(counts)
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        per_label[label] = {**counts, "precision": precision, "recall": recall, "f1": f1}
    precision = micro["tp"] / (micro["tp"] + micro["fp"]) if micro["tp"] + micro["fp"] else 0.0
    recall = micro["tp"] / (micro["tp"] + micro["fn"]) if micro["tp"] + micro["fn"] else 0.0
    summary = {
        "evaluated_cases": len(examples), "valid_json_cases": sum(record["json_valid"] for record in records),
        "errors": dict(errors), "micro": {**micro, "precision": precision, "recall": recall,
                                             "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0},
        "per_label": per_label,
        "ct_evidence": {
            **ct_totals,
            "score_field_coverage": (
                ct_totals["ct_score_slots_present"] / ct_totals["ct_score_slots_expected"]
                if ct_totals["ct_score_slots_expected"] else None
            ),
            "score_value_fidelity": (
                ct_totals["ct_score_slots_matched"] / ct_totals["ct_score_slots_expected"]
                if ct_totals["ct_score_slots_expected"] else None
            ),
        },
        "notice": "Agreement is against report-derived weak labels, not radiologist-adjudicated CT ground truth.",
    }
    (args.output_dir / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
