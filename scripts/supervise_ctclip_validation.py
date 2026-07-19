"""Continue the remote CT-CLIP experiment automatically after an upload finishes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import paramiko


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-log", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", default="root")
    parser.add_argument("--password-env", default="REMOTE_SSH_PASSWORD")
    parser.add_argument("--remote-root", default="/root/summer_zhl")
    parser.add_argument("--local-results", type=Path, required=True)
    return parser.parse_args()


def connect(cfg: argparse.Namespace) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        cfg.host, port=cfg.port, username=cfg.username,
        password=os.environ[cfg.password_env], timeout=45, banner_timeout=60, auth_timeout=45,
    )
    return client


def run(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command)
    code = stdout.channel.recv_exit_status()
    return code, stdout.read().decode(), stderr.read().decode()


def wait_upload(log: Path) -> None:
    while True:
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if "Upload complete." in text:
            return
        if "Traceback" in text and "Upload complete." not in text:
            raise RuntimeError("Uploader stopped with an error; see " + str(log))
        print("Waiting for volume upload ...", flush=True)
        time.sleep(60)


def main() -> None:
    cfg = args()
    if not os.environ.get(cfg.password_env):
        raise SystemExit(f"Set {cfg.password_env} before running.")
    wait_upload(cfg.upload_log)
    root = cfg.remote_root
    out = f"{root}/artifacts/ctclip_validation"
    client = connect(cfg)
    try:
        command = (
            f"cd {root} && mkdir -p {out}/batch && "
            f"nohup env PYTHONPATH={root} CTCLIP_TEXT_MODEL_DIR={root}/models/cxrbert "
            f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
            f"{root}/conda_env/bin/python scripts/batch_ctclip_infer.py "
            f"--manifest artifacts/ctclip_validation/test_manifest.csv "
            f"--out artifacts/ctclip_validation/batch/predictions.jsonl "
            f"--checkpoint {root}/models/ctclip/CT-CLIP_v2.pt "
            f"--source-dir {root}/external/CT-CLIP-main "
            f"--text-model-dir {root}/models/cxrbert --device cuda:0 "
            f"</dev/null > {out}/batch/inference.log 2>&1 & echo $!"
        )
        code, pid, error = run(client, command)
        if code != 0 or not pid.strip():
            raise RuntimeError(error or "Unable to start remote inference")
        pid = pid.strip().splitlines()[-1]
        print(f"Remote batch PID: {pid}", flush=True)
        while True:
            code, _, _ = run(client, f"kill -0 {pid} 2>/dev/null")
            if code != 0:
                break
            print("Remote inference running ...", flush=True)
            time.sleep(60)
        evaluation = (
            f"cd {root} && {root}/conda_env/bin/python scripts/evaluate_ctclip_validation.py "
            f"--predictions artifacts/ctclip_validation/batch/predictions.jsonl "
            f"--out-dir artifacts/ctclip_validation/evaluation"
        )
        code, output, error = run(client, evaluation)
        if code != 0:
            raise RuntimeError(error or output)
        print(output, flush=True)
        cfg.local_results.mkdir(parents=True, exist_ok=True)
        sftp = client.open_sftp()
        try:
            for name in ("metrics.json", "error_cases.csv", "failed_cases.json"):
                sftp.get(f"{out}/evaluation/{name}", str(cfg.local_results / name))
            for name in ("predictions.jsonl", "predictions.summary.json", "inference.log"):
                sftp.get(f"{out}/batch/{name}", str(cfg.local_results / name))
        finally:
            sftp.close()
    finally:
        client.close()
    print("Validation experiment complete.", flush=True)


if __name__ == "__main__":
    main()
