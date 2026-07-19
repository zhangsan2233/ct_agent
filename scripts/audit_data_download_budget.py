import argparse
import hashlib
import json
from pathlib import Path


LIMIT_BYTES = 25 * 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the additional dataset download budget.")
    parser.add_argument("--root", default="data/radgenome/archives")
    parser.add_argument("--ct-root", default="data/dataset/valid_fixed")
    parser.add_argument("--ct-start", type=int, default=51)
    parser.add_argument("--out", default="artifacts/data_download_budget.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    files = sorted(path for path in root.rglob("*") if path.is_file()) if root.exists() else []
    ct_root = Path(args.ct_root)
    if ct_root.exists():
        for patient_dir in ct_root.glob("valid_*"):
            try:
                patient_number = int(patient_dir.name.removeprefix("valid_"))
            except ValueError:
                continue
            if patient_number >= args.ct_start:
                files.extend(sorted(patient_dir.rglob("*.nii.gz")))
    files = sorted(set(files))
    records = []
    total = 0
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        total += path.stat().st_size
        records.append(
            {"path": str(path.as_posix()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        )
    payload = {
        "limit_bytes": LIMIT_BYTES,
        "downloaded_bytes": total,
        "remaining_bytes": LIMIT_BYTES - total,
        "within_budget": total <= LIMIT_BYTES,
        "files": records,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if total > LIMIT_BYTES:
        raise SystemExit("Dataset download budget exceeded")


if __name__ == "__main__":
    main()
