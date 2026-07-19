"""Freeze the Stage-2 ablation results into a compact reproducibility package."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


LABELS = [
    "arterial_wall_calcification", "atelectasis", "coronary_artery_wall_calcification",
    "emphysema", "lung_opacity", "lymphadenopathy", "pulmonary_fibrotic_sequela",
    "pulmonary_nodule",
]
ARMS = ("stage1_report_only", "stage2_report_only", "stage2_with_ctclip")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_summary(record: dict) -> dict:
    if not record.get("json_valid"):
        return {"json_valid": False, "error": record.get("error"),
                "raw_completion_excerpt": record.get("raw_completion", "")[:500]}
    output = record.get("prediction", {})
    labels = output.get("labels", []) if isinstance(output, dict) else []
    return {
        "json_valid": True,
        "positive_labels": record.get("positive_labels_predicted", []),
        "label_names": [item.get("name") for item in labels if isinstance(item, dict)],
        "ctclip_scores_emitted": {
            item.get("name"): item.get("ctclip_score")
            for item in labels if isinstance(item, dict) and "ctclip_score" in item
        },
    }


def choose_cases(inputs: dict[str, dict], outputs: dict[str, dict]) -> list[str]:
    """Prioritize no-CT failures, then high nodule and opacity evidence."""
    scored: list[tuple[tuple, str]] = []
    for case_id, item in inputs.items():
        payload = json.loads(item["messages"][1]["content"])
        scores = payload.get("ctclip_scores", {})
        without_ct = outputs["stage2_report_only"][case_id]
        failed_without_ct = int(not without_ct.get("json_valid") or not set(without_ct.get("positive_labels_predicted", [])) & set(LABELS))
        score = max(float(scores.get("pulmonary_nodule", 0)), float(scores.get("lung_opacity", 0)))
        scored.append(((failed_without_ct, score, float(scores.get("pulmonary_nodule", 0)),
                        float(scores.get("lung_opacity", 0))), case_id))
    scored.sort(reverse=True)
    selected: list[str] = []
    # One high-nodule and one high-opacity case are explicitly required.
    for label in ("pulmonary_nodule", "lung_opacity"):
        candidate = max(inputs, key=lambda case: float(json.loads(inputs[case]["messages"][1]["content"])["ctclip_scores"].get(label, 0)))
        if candidate not in selected:
            selected.append(candidate)
    for _, case_id in scored:
        if case_id not in selected:
            selected.append(case_id)
        if len(selected) == 5:
            break
    return selected[:5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.out_dir.resolve()
    if out.exists():
        raise SystemExit(f"Output already exists: {out}")
    out.mkdir(parents=True)

    evaluation = root / "artifacts/llm_eval/stage2_ctclip_500_holdout"
    stage2 = root / "artifacts/ctclip_stage2"
    train = root / "artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep"
    files = [
        (evaluation / "inputs/stage1_report_only.jsonl", "evaluation/inputs/stage1_report_only.jsonl"),
        (evaluation / "inputs/stage2_report_only.jsonl", "evaluation/inputs/stage2_report_only.jsonl"),
        (evaluation / "inputs/stage2_with_ctclip.jsonl", "evaluation/inputs/stage2_with_ctclip.jsonl"),
        (evaluation / "stage1_report_only/metrics.json", "evaluation/stage1_report_only/metrics.json"),
        (evaluation / "stage1_report_only/predictions.jsonl", "evaluation/stage1_report_only/predictions.jsonl"),
        (evaluation / "stage1_report_only.log", "evaluation/stage1_report_only.log"),
        (evaluation / "stage2_report_only/metrics.json", "evaluation/stage2_report_only/metrics.json"),
        (evaluation / "stage2_report_only/predictions.jsonl", "evaluation/stage2_report_only/predictions.jsonl"),
        (evaluation / "stage2_report_only.log", "evaluation/stage2_report_only.log"),
        (evaluation / "stage2_with_ctclip/metrics.json", "evaluation/stage2_with_ctclip/metrics.json"),
        (evaluation / "stage2_with_ctclip/predictions.jsonl", "evaluation/stage2_with_ctclip/predictions.jsonl"),
        (evaluation / "stage2_with_ctclip.log", "evaluation/stage2_with_ctclip.log"),
        (stage2 / "train_manifest_500.csv", "data/train_manifest_500.csv"),
        (stage2 / "ctclip_predictions_500.summary.json", "data/ctclip_predictions_500.summary.json"),
        (stage2 / "sft_compact_500_v1/manifest.json", "data/sft_compact_500_manifest.json"),
        (root / "artifacts/ctclip_stage2/ctclip_infer_500.log", "logs/ctclip_infer_500.log"),
        (root / "artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep.log", "logs/stage2_qlora_train.log"),
        (train / "adapter/adapter_config.json", "model/adapter_config.json"),
        (train / "adapter/training_args.bin", "model/training_args.bin"),
    ]
    for source, relative in files:
        if not source.is_file():
            raise SystemExit(f"Missing expected asset: {source}")
        copy(source, out / relative)

    inputs = {json.loads(line)["metadata"]["case_id"]: json.loads(line)
              for line in (evaluation / "inputs/stage2_with_ctclip.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    outputs = {
        arm: {row["case_id"]: row for row in read_jsonl(evaluation / arm / "predictions.jsonl")}
        for arm in ARMS
    }
    selected = choose_cases(inputs, outputs)
    cases = []
    for case_id in selected:
        payload = json.loads(inputs[case_id]["messages"][1]["content"])
        reference = outputs["stage2_with_ctclip"][case_id].get("positive_labels_reference", [])
        cases.append({
            "case_id": case_id,
            "selection_reason": {
                "stage2_without_ct_error": outputs["stage2_report_only"][case_id].get("error"),
                "pulmonary_nodule_score": payload["ctclip_scores"].get("pulmonary_nodule"),
                "lung_opacity_score": payload["ctclip_scores"].get("lung_opacity"),
            },
            "report_impression": payload.get("report_impression", ""),
            "ctclip_scores": payload["ctclip_scores"],
            "reference_positive_labels": reference,
            "stage1_report_only": label_summary(outputs["stage1_report_only"][case_id]),
            "stage2_without_ct": label_summary(outputs["stage2_report_only"][case_id]),
            "stage2_with_ctclip": label_summary(outputs["stage2_with_ctclip"][case_id]),
        })
    (out / "case_analysis.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = {arm: json.loads((evaluation / arm / "metrics.json").read_text(encoding="utf-8")) for arm in ARMS}
    adapter = train / "adapter/adapter_model.safetensors"
    reproducibility = {
        "split_seed": 20260718,
        "training_seed": 42,
        "training_seed_note": "Transformers TrainingArguments default; no explicit --seed override was supplied.",
        "evaluation_generation": {"do_sample": False, "max_new_tokens": 1024, "generation_batch_size": 4},
        "held_out_cases": 50,
        "held_out_case_set_sha256": "0299334088bbfc2b93dbe1b1b14b4f19f57b287c2be801b3ce788abaab02f622",
        "stage2_adapter_source": str(adapter),
        "stage2_adapter_sha256": sha256(adapter),
        "stage2_adapter_size_bytes": adapter.stat().st_size,
    }
    (out / "reproducibility.json").write_text(json.dumps(reproducibility, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = []
    for arm, name in (("stage1_report_only", "Stage-1 报告-only"),
                      ("stage2_report_only", "Stage-2 去 CT-CLIP"),
                      ("stage2_with_ctclip", "Stage-2 加 CT-CLIP")):
        metric = metrics[arm]
        ct = metric["ct_evidence"]
        rows.append(f"| {name} | {metric['valid_json_cases']}/50 | {metric['micro']['f1']:.3f} | "
                    f"{ct.get('score_field_coverage')} / {ct.get('score_value_fidelity')} |")
    report = [
        "# Stage-2 CT-CLIP 融合：留出集消融评测", "",
        "## 实验设计", "",
        "三臂使用完全相同的 50 个患者级留出病例；病例集合 SHA-256 见 `reproducibility.json`。"
        "生成采用贪婪解码（`do_sample=false`）和统一 1024 新 token 上限。", "",
        "| 评测臂 | 有效 JSON | 8 标签弱监督 micro-F1 | CT 分数覆盖 / 数值一致性 |",
        "|---|---:|---:|---:|", *rows, "",
        "## 结论与边界", "",
        "- Stage-2 在提供 CT-CLIP 分数时恢复了紧凑 8 标签 JSON 的稳定输出，并逐项保留输入的 CT 分数。",
        "- 移除 CT-CLIP 后，Stage-2 常输出无关标签模板，说明该融合 adapter 对 CT 证据输入存在实际依赖。",
        "- Stage-1 的 F1=1.0 不可解释为影像诊断优势：其报告输入包含由同一报告派生的模型分数，存在目标信息泄漏。",
        "- 所有 F1 均只衡量与 CT-RATE 报告派生弱标签的一致性，不是人工金标准下的临床诊断准确率。", "",
        "## 典型病例", "",
    ]
    for case in cases:
        reason = case["selection_reason"]
        report.append(f"### {case['case_id']}")
        report.append(f"- 选择原因：无 CT 输出错误=`{reason['stage2_without_ct_error']}`；"
                      f"肺结节分数={reason['pulmonary_nodule_score']}；肺实变/高密度分数={reason['lung_opacity_score']}。")
        report.append(f"- 报告印象：{case['report_impression']}")
        report.append(f"- CT-CLIP（8 项）：`{json.dumps(case['ctclip_scores'], ensure_ascii=False)}`")
        report.append(f"- 参考弱标签阳性：`{case['reference_positive_labels']}`")
        for key, title in (("stage1_report_only", "Stage-1 报告-only"),
                           ("stage2_without_ct", "Stage-2 去 CT"),
                           ("stage2_with_ctclip", "Stage-2 加 CT")):
            summary = case[key]
            report.append(f"- {title}：JSON 有效=`{summary['json_valid']}`；阳性标签=`{summary.get('positive_labels')}`；"
                          f"标签名=`{summary.get('label_names')}`；错误=`{summary.get('error')}`。")
        report.append("")
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    checksum_lines = []
    for path in sorted(item for item in out.rglob("*") if item.is_file() and item.name != "checksums.sha256"):
        checksum_lines.append(f"{sha256(path)}  {path.relative_to(out).as_posix()}")
    (out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"package": str(out), "cases": selected, "files": len(checksum_lines)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
