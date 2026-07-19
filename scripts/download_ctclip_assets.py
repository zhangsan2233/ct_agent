import argparse
import os
from pathlib import Path
import shutil
import json
import urllib.request

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


SOURCE_TREE_URL = (
    "https://api.github.com/repos/ibrahimethemhamamci/CT-CLIP/git/trees/main?recursive=1"
)
SOURCE_RAW_ROOT = "https://raw.githubusercontent.com/ibrahimethemhamamci/CT-CLIP/main"
REPO_ID = "ibrahimhamamci/CT-RATE"
CHECKPOINT_FILE = "models/CT-CLIP-Related/CT-CLIP_v2.pt"


def read_env_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def download_source(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(SOURCE_TREE_URL, headers={"User-Agent": "ChestCT-Agent"})
    with urllib.request.urlopen(request) as response:
        tree = json.load(response)["tree"]
    prefixes = ("CT_CLIP/ct_clip/", "transformer_maskgit/transformer_maskgit/")
    files = [
        item["path"]
        for item in tree
        if item.get("type") == "blob"
        and (item["path"].startswith(prefixes) or item["path"] == "README.md")
    ]
    for relative_path in files:
        output_path = destination / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{SOURCE_RAW_ROOT}/{relative_path}"
        print(f"Downloading {relative_path}")
        urllib.request.urlretrieve(url, output_path)
    print(f"Source ready: {destination}")


def download_checkpoint(destination: Path) -> None:
    token = read_env_token()
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Put it in .env before downloading gated weights.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = destination.parent / "hf_staging"
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=CHECKPOINT_FILE,
                token=token,
                local_dir=staging_dir,
            )
        )
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            raise RuntimeError(
                "Hugging Face denied the checkpoint. Enable access to public gated "
                "repositories for this token, then retry."
            ) from exc
        raise
    shutil.move(str(downloaded), destination)
    print(f"Checkpoint ready: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    args = parser.parse_args()
    source_dir = Path("external/CT-CLIP-main")
    checkpoint = Path("models/ctclip/CT-CLIP_v2.pt")
    if not args.weights_only and not source_dir.exists():
        download_source(source_dir)
    if not args.source_only and not checkpoint.exists():
        download_checkpoint(checkpoint)


if __name__ == "__main__":
    main()
