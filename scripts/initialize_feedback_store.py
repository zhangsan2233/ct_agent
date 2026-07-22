"""Initialize the SQLite store used by the reviewable feedback workflow."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.config import Settings
from chestct_agent.memory import AgentMemory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=Path("artifacts/memory/agent_memory.sqlite3")
    )
    args = parser.parse_args()
    memory = AgentMemory(Settings(memory_db_path=args.db))
    print(memory.path.resolve())


if __name__ == "__main__":
    main()
