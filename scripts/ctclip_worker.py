import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.ctclip import CtClipRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    runtime = CtClipRuntime(
        checkpoint=Path(args.checkpoint),
        source_dir=Path(args.source_dir),
        device=args.device,
        use_fp16=args.fp16,
    )
    print(json.dumps(runtime.predict(args.volume), sort_keys=True))


if __name__ == "__main__":
    main()
