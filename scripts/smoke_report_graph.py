import argparse
import asyncio
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.config import get_settings
from chestct_agent.tools.report_graph import ReportGraphTool


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run one RadGraph-XL report extraction.")
    parser.add_argument(
        "--report",
        default=(
            "Small bilateral pleural effusions are present. No pulmonary nodule. "
            "Right lower lobe opacity may represent atelectasis."
        ),
    )
    args = parser.parse_args()
    tool = ReportGraphTool(get_settings())
    try:
        graph = await tool.extract(args.report)
        print(json.dumps(graph.model_dump(mode="json"), ensure_ascii=False, indent=2))
    finally:
        tool.close()


if __name__ == "__main__":
    asyncio.run(main())
