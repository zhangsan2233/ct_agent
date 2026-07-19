import argparse
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely extract RadGenome masks for local CT cases.")
    parser.add_argument("--archive", default="data/radgenome/archives/valid_region_mask.tar.gz")
    parser.add_argument("--manifest", default="artifacts/evaluation/multimodal_manifest.csv")
    parser.add_argument("--out-dir", default="data/radgenome")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = Path(args.archive).resolve()
    output_root = Path(args.out_dir).resolve()
    cases = set(pd.read_csv(args.manifest)["case_id"].astype(str))
    selected_files = 0
    selected_bytes = 0
    existing_files = 0
    matched_cases: set[str] = set()
    with tarfile.open(archive, mode="r:gz") as handle:
        for member in handle:
            path = PurePosixPath(member.name)
            if not member.isfile() or len(path.parts) < 3:
                continue
            case_folder = path.parts[1]
            case_id = case_folder.removeprefix("seg_")
            if case_id not in cases:
                continue
            relative = Path(*path.parts)
            destination = (output_root / relative).resolve()
            if output_root not in destination.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if destination.exists() and destination.stat().st_size == member.size:
                existing_files += 1
                matched_cases.add(case_id)
                continue
            source = handle.extractfile(member)
            if source is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            selected_files += 1
            selected_bytes += member.size
            matched_cases.add(case_id)
    print(
        f"Extracted {selected_files} masks ({selected_bytes / 1024**3:.2f} GiB) "
        f"for {len(matched_cases)}/{len(cases)} local cases; "
        f"skipped {existing_files} existing files"
    )


if __name__ == "__main__":
    main()
