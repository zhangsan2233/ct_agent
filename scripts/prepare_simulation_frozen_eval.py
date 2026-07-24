"""Extract the frozen patient-disjoint Stage-2 rows from a simulation manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--simulation-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.simulation_manifest.read_text(encoding="utf-8"))
    if manifest.get("simulation_only") is not True:
        raise SystemExit("Simulation manifest marker is missing.")
    frozen = set(manifest.get("frozen_case_ids", []))
    if not frozen:
        raise SystemExit("No frozen case ids found.")
    rows = []
    for line in args.inputs.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row.get("metadata", {})
        case_id = row.get("case_id", metadata.get("case_id")) if isinstance(metadata, dict) else row.get("case_id")
        if case_id in frozen:
            rows.append(row)
    selected = {row.get("case_id", row.get("metadata", {}).get("case_id")) for row in rows}
    if selected != frozen:
        raise SystemExit(f"Frozen case mismatch; missing={sorted(frozen - selected)[:3]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"frozen_cases": len(rows), "simulation_only": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
