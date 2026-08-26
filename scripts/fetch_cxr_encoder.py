"""Prefetch TorchXRayVision weights for offline CXR encoding."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="densenet121-res224-all")
    args = parser.parse_args()
    try:
        import torchxrayvision as xrv
    except ImportError as exc:
        raise SystemExit("Install torchxrayvision first: pip install torchxrayvision") from exc
    model = xrv.models.DenseNet(weights=args.model)
    print(f"Loaded {args.model} with {len(model.pathologies)} pathologies.")
    print("Weights are cached by torchxrayvision/torch on first forward pass.")


if __name__ == "__main__":
    main()
