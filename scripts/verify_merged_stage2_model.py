"""Verify a merged Stage-2 model using saved CT-CLIP evidence from one acceptance run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.stage2_pipeline import SYSTEM_PROMPT, extract_json, validate_stage2_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-model-dir", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--report-file", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    for path in (args.merged_model_dir, args.source_result, args.report_file):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    case_id = source["input"]["case_id"]
    scores = source["ctclip_scores"]
    report = args.report_file.read_text(encoding="utf-8").strip()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.merged_model_dir, local_files_only=True, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.merged_model_dir,
        device_map="auto",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    payload = {
        "case_id": case_id,
        "task": "Integrate report and CT-CLIP evidence into the required compact JSON.",
        "report_impression": report[:3500],
        "ctclip_scores": scores,
        "ctclip_score_definition": "probability of the finding being present",
        "label_provenance": "research-only; report-derived weak labels, not radiologist-adjudicated CT ground truth",
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(generated[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    parsed, parse_error = extract_json(raw)
    errors = ([parse_error] if parse_error else []) + validate_stage2_json(parsed, case_id, scores)
    output = {
        "case_id": case_id,
        "model_kind": "merged_qwen3_5_9b_stage2",
        "merged_model_dir": str(args.merged_model_dir),
        "ctclip_scores": scores,
        "stage2_json": parsed,
        "validation": {"parseable_json": parse_error is None, "schema_valid": not errors, "errors": errors},
        "raw_stage2_output": raw,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["validation"], ensure_ascii=False))


if __name__ == "__main__":
    main()
