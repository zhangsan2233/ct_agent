"""Run the multi-modality interface: chest CT (Stage-2) or chest X-ray full pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chestct_agent.modalities import analyze_study, build_stage2_agent, list_modalities
from chestct_agent.stage2_pipeline import Stage2Paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print registered modalities and exit")
    parser.add_argument("--modality", default="cxr_chest", help="ct_chest, cxr_chest, or mr_chest")
    parser.add_argument("--image", type=Path, help="CT NIfTI or CXR PNG/JPEG")
    parser.add_argument("--case-id", default="demo_case")
    parser.add_argument("--report", default="")
    parser.add_argument("--report-file", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "artifacts" / "modality_runs")
    defaults = Stage2Paths.defaults(ROOT)
    parser.add_argument("--model-dir", type=Path, default=defaults.model_dir)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--ctclip-checkpoint", type=Path, default=defaults.ctclip_checkpoint)
    parser.add_argument("--ctclip-source", type=Path, default=defaults.ctclip_source)
    parser.add_argument("--text-model-dir", type=Path, default=defaults.text_model_dir)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.list:
        print(json.dumps({"modalities": list_modalities()}, ensure_ascii=False, indent=2))
        return
    if args.image is None:
        raise SystemExit("Supply --image, or use --list.")
    report = args.report
    if args.report_file is not None:
        report = args.report_file.read_text(encoding="utf-8")
    stage2_agent = None
    if args.modality in {"ct_chest", "cxr_chest"}:
        if args.adapter_dir is not None:
            from chestct_agent.stage2_pipeline import Stage2Agent

            paths = Stage2Paths(
                model_dir=args.model_dir,
                adapter_dir=args.adapter_dir,
                ctclip_checkpoint=args.ctclip_checkpoint,
                ctclip_source=args.ctclip_source,
                text_model_dir=args.text_model_dir,
                modality=args.modality,  # type: ignore[arg-type]
            )
            stage2_agent = Stage2Agent(paths, args.device)
        else:
            stage2_agent = build_stage2_agent(ROOT, args.modality, device=args.device)
    run_dir = args.runs_dir / args.case_id
    result = analyze_study(
        modality=args.modality,
        case_id=args.case_id,
        image_path=args.image,
        report_text=report,
        run_dir=run_dir,
        stage2_agent=stage2_agent,
        root=ROOT,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
