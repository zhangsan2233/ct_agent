"""Single-case defence demo: NIfTI + report -> CT-CLIP scores + Stage-2 JSON.

Run on the prepared GPU server.  The script is offline by design and never
downloads model files.  Output is research-only, not a clinical diagnosis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.ctclip import CtClipRuntime

LABELS = ["arterial_wall_calcification", "atelectasis", "coronary_artery_wall_calcification", "emphysema", "lung_opacity", "lymphadenopathy", "pulmonary_fibrotic_sequela", "pulmonary_nodule"]
SYSTEM = ("You are ChestCT-Agent, a research-only chest CT evidence integrator. Return compact JSON only. "
          "Use only the supplied report impression and CT-CLIP scores; do not invent findings. Preserve uncertainty "
          "and require human review because labels are weak supervision.")


def parse_json(text: str):
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        start = candidate.index("{")
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
        return value if isinstance(value, dict) else None
    except (ValueError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", required=True, type=Path, help="Input .nii or .nii.gz CT volume")
    parser.add_argument("--report", help="Report impression text")
    parser.add_argument("--report-file", type=Path, help="UTF-8 text file containing report impression")
    parser.add_argument("--case-id", help="Optional displayed case ID")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "Qwen3.5-9B")
    parser.add_argument("--adapter-dir", type=Path, default=ROOT / "artifacts" / "llm_qlora" / "qwen3_5_9b_ctclip_stage2_500_2ep" / "adapter")
    parser.add_argument("--ctclip-checkpoint", type=Path, default=ROOT / "models" / "ctclip" / "CT-CLIP_v2.pt")
    parser.add_argument("--ctclip-source", type=Path, default=ROOT / "external" / "CT-CLIP-main")
    parser.add_argument("--text-model-dir", type=Path, default=ROOT / "models" / "cxrbert")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=Path("demo_result.json"))
    args = parser.parse_args()
    if bool(args.report) == bool(args.report_file):
        raise SystemExit("Supply exactly one of --report or --report-file.")
    if not args.ct.is_file():
        raise SystemExit(f"CT file not found: {args.ct}")
    report = args.report if args.report is not None else args.report_file.read_text(encoding="utf-8").strip()
    os.environ.update({"CTCLIP_TEXT_MODEL_DIR": str(args.text_model_dir), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    ctclip = CtClipRuntime(args.ctclip_checkpoint, args.ctclip_source, args.device, use_fp16=True)
    raw_scores = ctclip.predict(str(args.ct))
    scores = {label: round(float(raw_scores[label]), 4) for label in LABELS}
    # CT-CLIP is only needed for one forward pass.  Reclaim its VRAM before
    # loading the 4-bit language model so a 24 GB demonstration GPU is enough.
    del raw_scores, ctclip
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    payload = {"case_id": args.case_id or args.ct.name, "task": "Integrate report and CT-CLIP evidence into the required compact JSON.", "report_impression": report[:3500], "ctclip_scores": scores, "ctclip_score_definition": "probability of the finding being present", "label_provenance": "research-only; report-derived weak labels, not radiologist-adjudicated CT ground truth"}
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    # Match the loader used in the held-out evaluation.  Qwen3.5 uses a custom
    # image-text architecture even though this Stage-2 prompt is text-only.
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForImageTextToText.from_pretrained(args.model_dir, quantization_config=qconfig, device_map="auto", local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter_dir, local_files_only=True).eval()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(generated[0][encoded.input_ids.shape[1]:], skip_special_tokens=True)
    result = {"case_id": payload["case_id"], "ct_path": str(args.ct), "ctclip_scores": scores, "stage2_json": parse_json(answer), "raw_stage2_output": answer, "warning": "Research-only output. It is not a clinical diagnosis and requires human review."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
