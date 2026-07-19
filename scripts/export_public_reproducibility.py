"""Export safe, public reproducibility metadata from a private result package.

The export intentionally excludes CT images, report text, raw generation output,
and CT-CLIP predictions. It retains only case identifiers, source-relative CT
paths, weak labels, split IDs, hyperparameters, and integrity hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


LABELS = [
    "arterial_wall_calcification", "atelectasis", "coronary_artery_wall_calcification",
    "emphysema", "lung_opacity", "lymphadenopathy", "pulmonary_fibrotic_sequela", "pulmonary_nodule",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=Path("reproducibility"))
    args = parser.parse_args()
    package, root, out = args.package_dir, args.project_root, args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    raw_manifest = package / "data" / "train_manifest_500.csv"
    raw_eval = package / "evaluation" / "inputs" / "stage2_with_ctclip.jsonl"
    reproducibility = json.loads((package / "reproducibility.json").read_text(encoding="utf-8"))
    sft_manifest = json.loads((package / "data" / "sft_compact_500_manifest.json").read_text(encoding="utf-8"))

    # This public manifest deliberately excludes local server paths, direct
    # download URLs and report_impression. Relative paths require authorized
    # CT-RATE access and are enough to locate the same source file.
    fields = ["case_id", "study_id", "volume_name", "hf_relative_path", *LABELS]
    rows: list[dict[str, str]] = []
    with raw_manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({field: row[field] for field in fields})
    if len(rows) != 500 or len({row["case_id"] for row in rows}) != 500:
        raise SystemExit("Expected exactly 500 unique cases in the private manifest.")
    if any("report" in key.lower() for row in rows for key in row):
        raise SystemExit("Refusing to export a manifest with report fields.")
    with (out / "ctrate_500_weak_label_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    valid_ids: list[str] = []
    for line in raw_eval.read_text(encoding="utf-8").splitlines():
        if line.strip():
            valid_ids.append(str(json.loads(line)["metadata"]["case_id"]))
    all_ids = {row["case_id"] for row in rows}
    valid_set = set(valid_ids)
    if len(valid_ids) != 50 or len(valid_set) != 50 or not valid_set <= all_ids:
        raise SystemExit("Held-out split does not match the 500-case manifest.")
    split_rows = [{"case_id": row["case_id"], "split": "valid" if row["case_id"] in valid_set else "train"} for row in rows]
    with (out / "patient_disjoint_split.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "split"])
        writer.writeheader(); writer.writerows(split_rows)

    config = {
        "dataset": {
            "source": "ibrahimhamamci/CT-RATE", "selected_cases": 500,
            "weak_supervision": True, "report_text_exported": False,
            "ct_images_exported": False, "labels": LABELS,
        },
        "split": {"seed": reproducibility["split_seed"], "train_cases": 450, "valid_cases": 50,
                  "held_out_case_set_sha256": reproducibility["held_out_case_set_sha256"]},
        "ctclip": {"frozen": True, "checkpoint": "CT-CLIP_v2.pt", "prediction_rows_exported": False},
        "stage2_qlora": {
            "base_model": "Qwen/Qwen3.5-9B", "continued_from": "stage1_report_only_adapter",
            "epochs": 2, "learning_rate": 5e-5, "batch_size": 1, "gradient_accumulation": 16,
            "max_length": 2048, "quantization": "4-bit NF4", "lora_rank": 16,
            "lora_alpha": 32, "lora_dropout": 0.05, "training_seed": reproducibility["training_seed"],
        },
        "evaluation": reproducibility["evaluation_generation"],
        "sft_manifest": sft_manifest,
    }
    (out / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    files = [
        out / "ctrate_500_weak_label_manifest.csv", out / "patient_disjoint_split.csv", out / "experiment_config.json",
        root / "artifacts" / "text_classifier.joblib",
        root / "artifacts" / "llm_qlora" / "qwen3_5_9b_evidence_json_1ep_single" / "adapter" / "adapter_model.safetensors",
        root / "artifacts" / "llm_qlora" / "qwen3_5_9b_ctclip_stage2_500_2ep" / "adapter" / "adapter_model.safetensors",
        root / "scripts" / "batch_ctclip_infer.py", root / "scripts" / "build_ctclip_stage2_sft.py", root / "scripts" / "train_llm_qlora.py",
    ]
    if not all(path.is_file() for path in files):
        missing = [str(path) for path in files if not path.is_file()]
        raise SystemExit("Cannot write checksums; missing: " + "; ".join(missing))
    (out / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name}\n" for path in files), encoding="utf-8")
    print(f"Wrote safe reproducibility export to {out}")


if __name__ == "__main__":
    main()
