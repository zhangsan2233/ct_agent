import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


METADATA_PATTERNS = [
    "dataset/radiology_text_reports/*",
    "dataset/multi_abnormality_labels/*",
    "dataset/metadata/*",
]

METADATA_FILES = [
    "dataset/radiology_text_reports/train_reports.csv",
    "dataset/radiology_text_reports/validation_reports.csv",
    "dataset/multi_abnormality_labels/train_predicted_labels.csv",
    "dataset/multi_abnormality_labels/valid_predicted_labels.csv",
    "dataset/metadata/train_metadata.csv",
    "dataset/metadata/validation_metadata.csv",
    "dataset/metadata/Metadata_Attributes.xlsx",
    "dataset/metadata/no_chest_train.txt",
    "dataset/metadata/no_chest_valid.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CT-RATE metadata/report/label files.")
    parser.add_argument("--repo-id", default="ibrahimhamamci/CT-RATE")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Use snapshot_download instead of downloading known metadata files one by one.",
    )
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument(
        "--token-env-name",
        default="HF_TOKEN",
        help="Environment/.env variable that stores the Hugging Face token.",
    )
    return parser.parse_args()


def read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_token(name: str) -> str | None:
    import os

    return os.environ.get(name) or read_dotenv(Path(".env")).get(name)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    token = get_token(args.token_env_name)

    if args.metadata_only and not args.snapshot:
        for repo_file in METADATA_FILES:
            print(f"Downloading {repo_file} ...", flush=True)
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=repo_file,
                local_dir=str(data_dir),
                token=token,
            )
        print(f"Downloaded metadata/report/label files to {data_dir.resolve()}")
        return

    patterns = METADATA_PATTERNS if args.metadata_only else args.allow_pattern
    if not patterns:
        raise SystemExit("Use --metadata-only or provide one or more --allow-pattern values.")
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=patterns,
        token=token,
    )
    print(path)


if __name__ == "__main__":
    main()
