"""Upload manifest-selected CT volumes and validation tools to the inference server."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import posixpath
import shlex
import sys
import time

import paramiko


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--volume-dir", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", default="root")
    parser.add_argument("--password-env", default="REMOTE_SSH_PASSWORD")
    parser.add_argument("--remote-root", default="/root/summer_zhl")
    parser.add_argument("--start-index", type=int, default=1,
                        help="1-based manifest position to resume from")
    return parser.parse_args()


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ[args.password_env]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.username,
        password=password,
        timeout=45,
        banner_timeout=60,
        auth_timeout=45,
    )
    return client


def upload_file_with_retries(
    args: argparse.Namespace, local: Path, remote: str, attempts: int = 4,
    persistent_client: list[paramiko.SSHClient | None] | None = None,
) -> str:
    """Upload through SCP over SSH, with size checks and reconnect retries.

    The inference host intermittently accepts SSH commands but leaves its SFTP
    subsystem unresponsive.  SCP uses the ordinary SSH exec channel instead.
    """
    local_size = local.stat().st_size
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        client = persistent_client[0] if persistent_client else None
        owns_client = persistent_client is None
        try:
            if client is None:
                client = connect(args)
                if persistent_client is not None:
                    persistent_client[0] = client
            quoted_remote = shlex.quote(remote)
            _, stdout, _ = client.exec_command(
                f"if test -f {quoted_remote}; then stat -c %s {quoted_remote}; fi"
            )
            stdout.channel.settimeout(120)
            size_text = stdout.read().decode("utf-8", errors="replace").strip()
            if size_text.isdigit() and int(size_text) == local_size:
                return "already present"

            stdin, stdout, _ = client.exec_command(f"scp -t {quoted_remote}")
            channel = stdin.channel
            channel.settimeout(180)

            def expect_ack() -> None:
                code = channel.recv(1)
                if code == b"\x00":
                    return
                detail = channel.recv(4096).decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"remote scp error {code!r}: {detail}")

            expect_ack()
            channel.sendall(f"C0644 {local_size} {local.name}\n".encode("utf-8"))
            expect_ack()
            with local.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    channel.sendall(chunk)
            channel.sendall(b"\x00")
            expect_ack()
            return "uploaded"
        except Exception as exc:  # Network drops are common on multi-GB uploads.
            last_error = exc
            print(f"  attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}", flush=True)
            if client is not None:
                client.close()
            if persistent_client is not None:
                persistent_client[0] = None
            time.sleep(min(10 * attempt, 30))
        finally:
            if owns_client and client is not None:
                client.close()
    raise RuntimeError(f"Failed to upload {local.name}: {last_error}")


def main() -> None:
    args = parse_args()
    password = os.environ.get(args.password_env)
    if not password:
        raise SystemExit(f"Set {args.password_env}; do not place the SSH password in a command argument.")
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8", newline="")))
    selected = [(row, args.volume_dir / row["volume_name"]) for row in rows]
    found = [(row, path) for row, path in selected if path.is_file()]
    missing = [row["volume_name"] for row, path in selected if not path.is_file()]
    if args.start_index < 1 or args.start_index > len(rows) + 1:
        raise SystemExit("--start-index must be within the manifest range.")
    found = found[args.start_index - 1:]
    print(f"Manifest cases: {len(rows)}; local files found: {len(found)}; missing: {len(missing)}", flush=True)

    # The target directories are created during the initial upload.  Avoid a
    # separate SSH connection on resume: the server intermittently stalls on
    # connection setup, while per-file uploads below have retry handling.
    connection: list[paramiko.SSHClient | None] = [None]
    try:
        for index, (row, local_path) in enumerate(found, start=1):
            remote_path = row["ct_volume_path"]
            started = time.monotonic()
            status = upload_file_with_retries(args, local_path, remote_path, persistent_client=connection)
            print(f"[{index}/{len(found)}] {row['volume_name']}: {status} ({time.monotonic() - started:.1f}s)", flush=True)
        tool_files = {
        PROJECT_ROOT / "scripts" / "batch_ctclip_infer.py": f"{args.remote_root}/scripts/batch_ctclip_infer.py",
        PROJECT_ROOT / "scripts" / "evaluate_ctclip_validation.py": f"{args.remote_root}/scripts/evaluate_ctclip_validation.py",
        args.manifest: f"{args.remote_root}/artifacts/ctclip_validation/test_manifest.csv",
        PROJECT_ROOT / "chestct_agent" / "__init__.py": f"{args.remote_root}/chestct_agent/__init__.py",
        PROJECT_ROOT / "chestct_agent" / "ctclip" / "__init__.py": f"{args.remote_root}/chestct_agent/ctclip/__init__.py",
        PROJECT_ROOT / "chestct_agent" / "ctclip" / "runtime.py": f"{args.remote_root}/chestct_agent/ctclip/runtime.py",
        }
        for local_path, remote_path in tool_files.items():
            status = upload_file_with_retries(args, local_path, remote_path, persistent_client=connection)
            print(f"tool {local_path.name}: {status}", flush=True)
    finally:
        if connection[0] is not None:
            connection[0].close()
    print("Upload complete.")
    if missing:
        print("Missing local volumes: " + ", ".join(missing), flush=True)


if __name__ == "__main__":
    main()
