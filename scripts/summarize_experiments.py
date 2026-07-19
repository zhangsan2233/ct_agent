import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def package_versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "transformers", "sentence-transformers", "langgraph", "mcp", "qdrant-client"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def main() -> None:
    manifest_path = Path("artifacts/evaluation/multimodal_manifest.csv")
    manifest = pd.read_csv(manifest_path).fillna("") if manifest_path.exists() else pd.DataFrame()
    mask_index_path = Path("artifacts/radgenome/mask_index.csv")
    mask_index = pd.read_csv(mask_index_path).fillna("") if mask_index_path.exists() else pd.DataFrame()
    model_files = [
        Path("models/ctclip/CT-CLIP_v2.pt"),
        Path("models/qwen/Qwen3-Embedding-0.6B/model.safetensors"),
        Path("models/qwen/Qwen3-Reranker-0.6B/model.safetensors"),
    ]
    metrics = {}
    for name, relative in {
        "text_classifier": "artifacts/evaluation/text_classifier_metrics_18.json",
        "ctclip": "artifacts/evaluation/ctclip_metrics.json",
        "patient_ablation": "artifacts/evaluation/ablation_patient_metrics.json",
        "retrieval": "artifacts/evaluation/rag_metrics.json",
        "report_evidence": "artifacts/evaluation/report_evidence_metrics.json",
        "agent_capabilities": "artifacts/evaluation/agent_capabilities.json",
        "qwen_prompt": "artifacts/evaluation/qwen_prompt_metrics.json",
        "end_to_end_agent": "artifacts/evaluation/end_to_end_valid1_metrics.json",
        "split_leakage_audit": "artifacts/evaluation/split_leakage_audit.json",
    }.items():
        value = load_json(Path(relative))
        if value is not None:
            metrics[name] = value
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "local_ct_volumes": int(len(manifest)),
            "local_patients": int(
                manifest["case_id"].astype(str).str.extract(r"^(valid_\d+)")[0].nunique()
            ) if not manifest.empty else 0,
            "radgenome_masks": int(len(mask_index)),
            "radgenome_cases": int(mask_index["case_id"].nunique()) if not mask_index.empty else 0,
            "download_budget": load_json(Path("artifacts/data_download_budget.json")),
        },
        "models": {
            path.as_posix(): {
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
            for path in model_files
        },
        "packages": package_versions(),
        "metrics": metrics,
        "safety_notes": [
            "CT-RATE validation labels used here are dataset-provided weak labels.",
            "RadGenome normalized-index overlays are marked alignment_verified=false unless affine checks pass.",
            "This system is a coursework/research prototype and is not a clinical diagnostic device.",
        ],
    }
    output = Path("artifacts/evaluation/experiment_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
