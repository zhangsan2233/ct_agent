"""Create a reviewable calibration candidate from approved SQLite feedback.

This safe local phase never changes an active model or threshold.  It only writes a
candidate JSON summary; server-side evaluation must approve promotion separately.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-approved", type=int, default=50)
    parser.add_argument("--modality", choices=["ct_chest", "cxr_chest", "all"], default="all")
    args = parser.parse_args()
    query = "SELECT label, before_status, corrected_status, model_version FROM feedback_events WHERE status='approved'"
    with sqlite3.connect(args.db) as connection:
        rows = connection.execute(query).fetchall()
    if args.modality != "all":
        rows = [row for row in rows if row[3].startswith(f"{args.modality}:") or (args.modality == "ct_chest" and ":" not in row[3])]
    changes = Counter(label for label, before, after, _version in rows if before != after)
    candidate = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "modality": args.modality,
        "approved_feedback_count": len(rows),
        "minimum_approved_feedback": args.minimum_approved,
        "eligible_for_server_calibration": len(rows) >= args.minimum_approved,
        "label_change_counts": dict(sorted(changes.items())),
        "notice": "Dry-run candidate only. It does not modify active thresholds or model weights.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(candidate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
