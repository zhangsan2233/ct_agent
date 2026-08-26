#!/usr/bin/env bash
# Daily, non-promoting feedback maintenance for the controlled server.
set -euo pipefail

ROOT="${CHESTCT_ROOT:-/root/summer_zhl}"
PYTHON_BIN="${CHESTCT_PYTHON:-$ROOT/conda_env/bin/python}"
DB="$ROOT/artifacts/memory/agent_memory.sqlite3"
OUT_DIR="$ROOT/artifacts/feedback"

cd "$ROOT"
"$PYTHON_BIN" scripts/initialize_feedback_store.py --db "$DB"
for MODALITY in ct_chest cxr_chest; do
  "$PYTHON_BIN" scripts/run_feedback_calibration.py \
    --db "$DB" \
    --out "$OUT_DIR/candidate_calibration_${MODALITY}.json" \
    --modality "$MODALITY" \
    --minimum-approved "${FEEDBACK_MINIMUM_APPROVED:-50}"
done
