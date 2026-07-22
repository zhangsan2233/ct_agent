#!/usr/bin/env bash
# Fallback scheduler for inference containers that do not run cron/systemd.
set -euo pipefail

ROOT="${CHESTCT_ROOT:-/root/summer_zhl}"
INTERVAL_SECONDS="${FEEDBACK_MAINTENANCE_INTERVAL_SECONDS:-86400}"

while true; do
  flock -n /tmp/chestct-feedback-maintenance.lock \
    "$ROOT/scripts/run_feedback_maintenance.sh" || true
  sleep "$INTERVAL_SECONDS"
done
