"""Download only the CT-RATE NIfTI volumes named in a fixed test manifest.

Requires a Hugging Face token that has been granted access to ibrahimhamamci/CT-RATE.
Read it from HF_TOKEN or from a local .env file; neither is printed or uploaded.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def dotenv_token(name: str) -> str | None:
    token = os.environ.get(name)
    if token:
        return token
    dotenv = Path(".env")
    if not dotenv.exists():
        return None
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key == name:
            return value.strip().strip("\"'")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--repo-id", default="ibrahimhamamci/CT-RATE")
    parser.add_argument("--token-env-name", default="HF_TOKEN")
    args = parser.parse_args()
    token = dotenv_token(args.token_env_name)
    if not token:
        raise SystemExit("No Hugging Face token found. Set HF_TOKEN or add HF_TOKEN=... to .env.")
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8", newline="")))
    if not rows:
        raise SystemExit("Manifest is empty.")
    for number, row in enumerate(rows, start=1):
        # CT-RATE repository layout is dataset/valid_fixed/...; --data-dir keeps
        # the resulting local path as data/dataset/valid_fixed/..., as expected.
        remote_name = "dataset/valid_fixed/" + "/".join(row["ct_volume_path"].split("/valid_fixed/", 1)[1].split("/"))
        print(f"[{number}/{len(rows)}] {remote_name}", flush=True)
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=remote_name,
            local_dir=str(args.data_dir),
            token=token,
        )
    print(f"Downloaded {len(rows)} manifest volumes to {args.data_dir.resolve()}")


if __name__ == "__main__":
    main()
