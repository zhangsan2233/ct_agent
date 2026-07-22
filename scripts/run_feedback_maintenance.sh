#!/usr/bin/env bash
# Daily, non-promoting feedback maintenance for the controlled server.
set -euo pipefail

ROOT="${CHESTCT_ROOT:-/root/summer_zhl}"
PYTHON_BIN="${CHESTCT_PYTHON:-$ROOT/conda_env/bin/python}"
DB="$ROOT/artifacts/memory/agent_memory.sqlite3"
OUT="$ROOT/artifacts/feedback/candidate_calibration.json"

cd "$ROOT"
"$PYTHON_BIN" scripts/initialize_feedback_store.py --db "$DB"
"$PYTHON_BIN" scripts/run_feedback_calibration.py \
  --db "$DB" --out "$OUT" --minimum-approved "${FEEDBACK_MINIMUM_APPROVED:-50}"
