"""Merge the final Stage-2 QLoRA adapter into a standalone local Qwen3.5 model.

The input base model and adapter must already be present on disk.  This script never
downloads weights.  It loads the base model in BF16 (not 4-bit) because LoRA merging
into quantized weights is not a reliable export path, then writes a sharded SafeTensors
model that can be loaded without PEFT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    for name, path in {"base model": args.model_dir, "adapter": args.adapter_dir}.items():
        if not path.is_dir():
            raise SystemExit(f"{name} directory not found: {path}")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.out_dir}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        device_map=args.device_map,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base, args.adapter_dir, local_files_only=True, low_cpu_mem_usage=True
    )
    merged = model.merge_and_unload(safe_merge=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(
        args.out_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor.save_pretrained(args.out_dir)
    metadata = {
        "base_model_dir": str(args.model_dir.resolve()),
        "merged_adapter_dir": str(args.adapter_dir.resolve()),
        "dtype": "bfloat16",
        "safe_serialization": True,
        "max_shard_size": args.max_shard_size,
        "note": "Merged locally from Qwen3.5-9B plus the final Stage-2 CT-CLIP evidence adapter.",
    }
    (args.out_dir / "merge_provenance.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
