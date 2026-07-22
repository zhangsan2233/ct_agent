"""Verify that the API exposes the three reviewable-feedback endpoints."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.api.main import app


REQUIRED_PATHS = {
    "/api/cases/{case_id}/feedback",
    "/api/feedback",
    "/api/feedback/{event_id}/review",
}


def main() -> None:
    paths = {route.path for route in app.routes}
    missing = sorted(REQUIRED_PATHS - paths)
    if missing:
        raise SystemExit("Feedback API routes missing: " + ", ".join(missing))
    print("feedback API routes verified")


if __name__ == "__main__":
    main()
