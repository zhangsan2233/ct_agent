"""Initialize the parallel CXR Stage-2 adapter by copying the frozen CT Stage-2 adapter.

This never overwrites an existing CXR production adapter unless --force is passed.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.modality_paths import CT_ADAPTER_REL, CXR_ADAPTER_REL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source = args.root / CT_ADAPTER_REL
    target = args.root / CXR_ADAPTER_REL
    if not source.is_dir():
        raise SystemExit(f"CT Stage-2 adapter not found: {source}")
    if target.exists() and any(target.iterdir()) and not args.force:
        raise SystemExit(f"CXR adapter already exists: {target} (use --force to replace)")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    print(f"Initialized CXR adapter at {target} from {source}")


if __name__ == "__main__":
    main()
