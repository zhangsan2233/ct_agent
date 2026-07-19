"""Resume-safe SCP-over-SSH uploader for Qwen3.5-9B and LLM SFT assets."""
from __future__ import annotations

import argparse
import os
import shlex
import time
from pathlib import Path

from upload_ctclip_validation_remote import connect, upload_file_with_retries


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", default="root")
    parser.add_argument("--password-env", default="REMOTE_SSH_PASSWORD")
    parser.add_argument("--remote-root", default="/root/summer_zhl")
    parser.add_argument("--model-dir", default=PROJECT_ROOT / "models" / "qwen3_5_9B" / "Qwen3.5-9B", type=Path)
    parser.add_argument("--sft-dir", default=PROJECT_ROOT / "artifacts" / "llm_sft", type=Path)
    parser.add_argument("--start-index", type=int, default=1, help="1-based file index for resume")
    args = parser.parse_args()
    if not os.environ.get(args.password_env):
        raise SystemExit(f"Set {args.password_env}; do not pass the password as an argument.")
    model_files = sorted(path for path in args.model_dir.rglob("*") if path.is_file())
    required_model = {"config.json", "model.safetensors.index.json", "tokenizer.json", "preprocessor_config.json"}
    missing = required_model - {path.name for path in model_files}
    if missing:
        raise SystemExit(f"Model directory is incomplete: {sorted(missing)}")
    files: list[tuple[Path, str]] = [
        (path, f"{args.remote_root}/models/Qwen3.5-9B/{path.relative_to(args.model_dir).as_posix()}")
        for path in model_files
    ]
    for name in ("train.jsonl", "valid.jsonl", "manifest.json"):
        path = args.sft_dir / name
        if not path.is_file():
            raise SystemExit(f"Missing SFT artifact: {path}")
        files.append((path, f"{args.remote_root}/artifacts/llm_sft/{name}"))
    for relative in (
        "scripts/train_llm_qlora.py", "scripts/build_llm_sft_dataset.py", "scripts/evaluate_llm_adapter.py",
        "scripts/build_ctclip_ablation_eval.py",
        "scripts/prepare_ctclip_stage2_train_manifest.py",
        "requirements-llm-train.txt", "docs/LLM_QLORA_TRAINING.md",
    ):
        path = PROJECT_ROOT / relative
        files.append((path, f"{args.remote_root}/{relative}"))
    stage2_manifest = PROJECT_ROOT / "artifacts" / "ctclip_stage2" / "train_manifest_1000.csv"
    if stage2_manifest.is_file():
        files.append((stage2_manifest, f"{args.remote_root}/artifacts/ctclip_stage2/train_manifest_1000.csv"))
    # Required only for the CT-CLIP paired ablation evaluation.  They are
    # small tabular metadata files, not CT volumes or model weights.
    for relative in (
        "data/dataset/radiology_text_reports/validation_reports.csv",
        "data/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
    ):
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise SystemExit(f"Missing ablation input: {path}")
        files.append((path, f"{args.remote_root}/{relative}"))
    if not 1 <= args.start_index <= len(files) + 1:
        raise SystemExit(f"--start-index must be 1..{len(files) + 1}")
    client = connect(args)
    connection = [client]
    try:
        parents = sorted({str(Path(remote).parent).replace("\\", "/") for _, remote in files})
        command = "mkdir -p " + " ".join(shlex.quote(parent) for parent in parents)
        _, stdout, stderr = client.exec_command(command)
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError(stderr.read().decode("utf-8", errors="replace"))
        print(f"Files total: {len(files)}; starting at {args.start_index}", flush=True)
        for index, (local, remote) in enumerate(files[args.start_index - 1:], start=args.start_index):
            started = time.monotonic()
            status = upload_file_with_retries(args, local, remote, persistent_client=connection)
            print(f"[{index}/{len(files)}] {local.name}: {status} ({time.monotonic() - started:.1f}s)", flush=True)
    finally:
        if connection[0] is not None:
            connection[0].close()
    print("Upload complete.", flush=True)


if __name__ == "__main__":
    main()
