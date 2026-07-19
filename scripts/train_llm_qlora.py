"""Text-evidence QLoRA SFT for the local Qwen3.5-9B model.

The script deliberately accepts a *local* base-model directory only. It never
attempts to fetch a model from Hugging Face, which keeps model access explicit
and reproducible on the training server.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path, help="Local Qwen-Instruct model directory")
    parser.add_argument("--adapter-path", type=Path,
                        help="Optional existing LoRA adapter to continue training from.")
    parser.add_argument("--train-file", default="artifacts/llm_sft/train.jsonl", type=Path)
    parser.add_argument("--valid-file", default="artifacts/llm_sft/valid.jsonl", type=Path)
    parser.add_argument("--output-dir", default="artifacts/llm_qlora", type=Path)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override epoch-based training and stop after this many optimizer steps.")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # RTX 4090 does not expose the P2P/IB paths assumed by Accelerate's
    # multi-GPU probe. This keeps even a single-GPU smoke run portable.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    if not args.model_path.is_dir():
        raise SystemExit(f"Local model directory not found: {args.model_path}")
    if args.adapter_path is not None and not args.adapter_path.is_dir():
        raise SystemExit(f"LoRA adapter directory not found: {args.adapter_path}")
    for path in (args.train_file, args.valid_file):
        if not path.is_file():
            raise SystemExit(f"Dataset file not found: {path}")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
        Trainer, TrainingArguments,
    )

    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # In DDP, each rank owns one 4-bit model replica on its assigned GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank} if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "auto"
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        # With `accelerate launch --multi_gpu`, each DDP process must own one
        # complete 4-bit replica. `device_map="auto"` would otherwise shard a
        # replica across both cards and conflict with DDP.
        device_map=device_map,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    data = load_dataset("json", data_files={"train": str(args.train_file), "validation": str(args.valid_file)})
    def tokenise(example: dict) -> dict:
        # Stage 1 is text-only: no synthetic image is inserted. The processor
        # retains Qwen3.5 compatibility and allows stage 2 to add CT previews.
        prompt = processor.apply_chat_template(example["messages"][:2], tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(full, truncation=True, max_length=args.max_length, add_special_tokens=False)
        prefix = tokenizer(prompt, truncation=True, max_length=args.max_length, add_special_tokens=False)["input_ids"]
        labels = list(encoded["input_ids"])
        labels[:min(len(labels), len(prefix))] = [-100] * min(len(labels), len(prefix))
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": labels}
    columns = data["train"].column_names
    data = data.map(tokenise, remove_columns=columns, desc="Tokenising chat records")
    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=True)
    else:
        lora = LoraConfig(
            r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                "in_proj_a", "in_proj_b", "out_proj", "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=args.logging_steps,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=2, bf16=True, tf32=True, gradient_checkpointing=True,
        report_to="none", remove_unused_columns=False, ddp_find_unused_parameters=False,
    )
    def collate(features: list[dict]) -> dict:
        max_length = max(len(row["input_ids"]) for row in features)
        pad_id = tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([row["input_ids"] + [pad_id] * (max_length - len(row["input_ids"])) for row in features]),
            "attention_mask": torch.tensor([row["attention_mask"] + [0] * (max_length - len(row["attention_mask"])) for row in features]),
            "labels": torch.tensor([row["labels"] + [-100] * (max_length - len(row["labels"])) for row in features]),
        }
    trainer = Trainer(model=model, args=training_args, train_dataset=data["train"], eval_dataset=data["validation"], data_collator=collate)
    trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))


if __name__ == "__main__":
    main()
