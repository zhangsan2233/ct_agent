import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path

import httpx

# hf_xet can stall on some Windows networks before writing any bytes. The
# regular HTTPS path supports resume and is more predictable for this dataset.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download an inclusive CT-RATE valid_fixed patient range."
    )
    parser.add_argument("--repo-id", default="ibrahimhamamci/CT-RATE")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--one-reconstruction",
        action="store_true",
        help="Keep only files ending in _1.nii.gz for each scan folder.",
    )
    parser.add_argument(
        "--max-download-gb",
        type=float,
        default=0.0,
        help="Abort before downloading when remaining bytes exceed this GiB limit.",
    )
    parser.add_argument("--token-env-name", default="HF_TOKEN")
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
    return os.environ.get(name) or read_dotenv(Path(".env")).get(name)


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def list_patient_files(
    repo_id: str,
    revision: str,
    patient: int,
    token: str | None,
) -> list[RemoteFile]:
    prefix = f"dataset/valid_fixed/valid_{patient}"
    entries = HfApi().list_repo_tree(
        repo_id=repo_id,
        path_in_repo=prefix,
        recursive=True,
        expand=True,
        revision=revision,
        repo_type="dataset",
        token=token,
    )
    return [
        RemoteFile(path=entry.path, size=int(entry.size))
        for entry in entries
        if getattr(entry, "size", None) is not None and entry.path.endswith(".nii.gz")
    ]


def discover_files(args: argparse.Namespace, token: str | None) -> list[RemoteFile]:
    patients = list(range(args.start, args.end + 1))
    discovered: list[RemoteFile] = []
    list_workers = min(8, len(patients))
    with ThreadPoolExecutor(max_workers=list_workers) as executor:
        futures = {
            executor.submit(
                list_patient_files,
                args.repo_id,
                args.revision,
                patient,
                token,
            ): patient
            for patient in patients
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            patient = futures[future]
            files = future.result()
            discovered.extend(files)
            print(
                f"Indexed valid_{patient}: {len(files)} volume(s) "
                f"[{completed_count}/{len(patients)}]",
                flush=True,
            )
    if args.one_reconstruction:
        discovered = [item for item in discovered if item.path.endswith("_1.nii.gz")]
    return sorted(discovered, key=lambda item: item.path)


def is_complete(local_root: Path, remote_file: RemoteFile) -> bool:
    local_path = local_root / Path(remote_file.path)
    return local_path.exists() and local_path.stat().st_size == remote_file.size


def check_download_access(
    repo_id: str,
    revision: str,
    remote_file: RemoteFile,
    token: str | None,
) -> None:
    if not token:
        raise SystemExit(
            "未找到 HF_TOKEN。请先在 .env 中配置具有 gated dataset 读取权限的 token。"
        )
    url = hf_hub_url(
        repo_id=repo_id,
        filename=remote_file.path,
        repo_type="dataset",
        revision=revision,
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Range": "bytes=0-0",
    }
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code < 400:
                return
            body = response.read().decode("utf-8", errors="replace")
    if response.status_code == 403 and "public gated repositories" in body:
        raise SystemExit(
            "HF_TOKEN 没有公开 gated 仓库下载权限。请打开 "
            "https://huggingface.co/settings/tokens ，编辑或新建 token，并启用 "
            "'Read access to contents of all public gated repos you can access'，"
            "然后更新 .env 中的 HF_TOKEN。"
        )
    raise SystemExit(
        f"Hugging Face 下载权限检查失败：HTTP {response.status_code}，{body[:300]}"
    )


def download_file(
    args: argparse.Namespace,
    token: str | None,
    remote_file: RemoteFile,
) -> str:
    return hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        filename=remote_file.path,
        local_dir=args.data_dir,
        token=token,
    )


def main() -> None:
    args = parse_args()
    if args.start < 1 or args.end < args.start:
        raise SystemExit("Require 1 <= --start <= --end.")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1.")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    token = get_token(args.token_env_name)

    print(
        f"CT-RATE valid_fixed range: valid_{args.start} through valid_{args.end}",
        flush=True,
    )
    print(f"Destination: {data_dir.resolve()}", flush=True)
    files = discover_files(args, token)
    if files and not args.dry_run:
        check_download_access(args.repo_id, args.revision, files[0], token)
    total_size = sum(item.size for item in files)
    complete_files = [item for item in files if is_complete(data_dir, item)]
    remaining_files = [item for item in files if not is_complete(data_dir, item)]
    download_size = sum(item.size for item in remaining_files)

    if args.max_download_gb > 0:
        max_download_bytes = int(args.max_download_gb * 1024**3)
        if download_size > max_download_bytes:
            raise SystemExit(
                f"Refusing download: {human_bytes(download_size)} exceeds "
                f"--max-download-gb={args.max_download_gb:.2f} GiB."
            )

    if args.dry_run:
        print(f"Matched files: {len(files)}")
        print(f"Total size: {human_bytes(total_size)}")
        print(f"Already complete locally: {len(complete_files)}")
        print(f"Remaining download: {human_bytes(download_size)}")
        if files:
            print("First matched files:")
            for item in files[:5]:
                print(f"  {item.path} ({human_bytes(item.size)})")
        return

    print(f"Matched files: {len(files)} ({human_bytes(total_size)})")
    print(
        f"Already complete: {len(complete_files)}; "
        f"to download: {len(remaining_files)} ({human_bytes(download_size)})",
        flush=True,
    )
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(download_file, args, token, remote_file): remote_file
            for remote_file in remaining_files
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            remote_file = futures[future]
            try:
                future.result()
                print(
                    f"Downloaded [{completed_count}/{len(remaining_files)}]: "
                    f"{remote_file.path}",
                    flush=True,
                )
            except Exception as exc:
                failures.append((remote_file.path, f"{type(exc).__name__}: {exc}"))
                print(f"Failed: {remote_file.path} ({type(exc).__name__})", flush=True)

    valid_root = data_dir / "dataset" / "valid_fixed"
    downloaded = [
        path
        for patient in range(args.start, args.end + 1)
        for path in (valid_root / f"valid_{patient}").rglob("*.nii.gz")
    ]
    total_size = sum(path.stat().st_size for path in downloaded)
    print(f"Local files in requested range: {len(downloaded)}")
    print(f"Local size in requested range: {human_bytes(total_size)}")
    print(f"Completed: {valid_root.resolve()}")
    if failures:
        print(f"Failed files: {len(failures)}")
        for path, error in failures:
            print(f"  {path}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
