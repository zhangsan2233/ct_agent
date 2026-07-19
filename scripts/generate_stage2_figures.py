"""Generate publication-ready SVG/PDF-independent figures from a Stage-2 result package.

The script deliberately reads only the packaged metrics/predictions.  It does
not recompute labels, so figures are traceable to the frozen evaluation bundle.
No third-party plotting package is required.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


ARMS = [
    ("stage1_report_only", "Stage-1\nreport only"),
    ("stage2_report_only", "Stage-2\nwithout CT"),
    ("stage2_with_ctclip", "Stage-2\nwith CT-CLIP"),
]
LABELS = [
    "arterial_wall_calcification", "atelectasis", "coronary_artery_wall_calcification",
    "emphysema", "lung_opacity", "lymphadenopathy", "pulmonary_fibrotic_sequela", "pulmonary_nodule",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_value(value, names: set[str]):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in names and isinstance(item, (int, float)):
                return float(item)
            found = find_value(item, names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_value(item, names)
            if found is not None:
                return found
    return None


def metric(metrics: dict, *names: str, default: float = 0.0) -> float:
    found = find_value(metrics, set(names))
    return default if found is None else found


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#172033}.title{font-size:20px;font-weight:700}.axis{font-size:12px}.label{font-size:13px}.small{font-size:11px}</style>',
        f'<title>{html.escape(title)}</title><rect width="100%" height="100%" fill="white"/>',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines + ["</svg>"]) + "\n", encoding="utf-8")


def bar_chart(path: Path, title: str, labels: list[str], values: list[float], *, colors: list[str] | None = None, percent: bool = True) -> None:
    w, h, left, bottom, top = 900, 510, 90, 110, 70
    plot_w, plot_h = w - left - 40, h - bottom - top
    colors = colors or ["#2563eb"] * len(values)
    out = svg_start(w, h, title)
    out += [f'<text class="title" x="{left}" y="36">{html.escape(title)}</text>', f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#64748b"/>']
    for tick in range(0, 101, 20):
        y = top + plot_h - plot_h * tick / 100
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>', f'<text class="axis" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{tick}%</text>']
    gap = plot_w / len(values)
    width = min(125, gap * 0.58)
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + gap * (i + .5) - width / 2
        height = plot_h * value
        y = top + plot_h - height
        out += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="5" fill="{colors[i]}"/>', f'<text class="label" x="{x+width/2:.1f}" y="{y-10:.1f}" text-anchor="middle">{value*100:.1f}%</text>']
        for j, part in enumerate(label.split("\n")):
            out.append(f'<text class="axis" x="{x+width/2:.1f}" y="{top+plot_h+27+j*15}" text-anchor="middle">{html.escape(part)}</text>')
    write_svg(path, out)


def per_label_chart(path: Path, values: dict[str, float]) -> None:
    labels = [label.replace("_", " ") for label in LABELS]
    bar_chart(path, "Stage-2 + CT-CLIP: per-label weak-label F1", labels, [values.get(label, 0.0) for label in LABELS], colors=["#0f766e"] * 8)


def write_case_table(path: Path, analysis: dict) -> None:
    rows = analysis.get("cases", []) if isinstance(analysis, dict) else analysis
    fields = ["case_id", "pulmonary_nodule_score", "lung_opacity_score", "stage2_without_ct", "stage2_with_ctclip"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows[:5]:
            key = item.get("key_ctclip_scores", item.get("ctclip_scores", {}))
            arms = item.get("arms", item)
            writer.writerow({
                "case_id": item.get("case_id", ""),
                "pulmonary_nodule_score": key.get("pulmonary_nodule_score", key.get("pulmonary_nodule", "")),
                "lung_opacity_score": key.get("lung_opacity_score", key.get("lung_opacity", "")),
                "stage2_without_ct": json.dumps(arms.get("stage2_report_only", arms.get("stage2_without_ct", "")), ensure_ascii=False)[:220],
                "stage2_with_ctclip": json.dumps(arms.get("stage2_with_ctclip", ""), ensure_ascii=False)[:220],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    package = args.package_dir
    out = args.out_dir or package / "formal_figures"
    out.mkdir(parents=True, exist_ok=True)
    metrics = {name: load(package / "evaluation" / name / "metrics.json") for name, _ in ARMS}
    validity = [
        metric(metrics[name], "valid_json_rate", default=metric(metrics[name], "valid_json_cases", "valid_json", default=0) /
               max(metric(metrics[name], "evaluated_cases", "total", "cases", default=50), 1))
        for name, _ in ARMS
    ]
    f1 = [metric(metrics[name], "f1", "micro_f1", default=0.0) for name, _ in ARMS]
    # Stage-1 is intentionally retained: it is a format/reference-leakage control, not diagnostic accuracy.
    bar_chart(out / "json_validity_three_arm.svg", "Valid JSON rate on the paired 50-case holdout", [label for _, label in ARMS], validity, colors=["#2563eb", "#dc2626", "#0f766e"])
    bar_chart(out / "weak_label_f1_three_arm.svg", "Weak-label agreement F1 on the paired 50-case holdout", [label for _, label in ARMS], f1, colors=["#2563eb", "#dc2626", "#0f766e"])
    evidence = metrics["stage2_with_ctclip"]
    coverage = metric(evidence, "score_field_coverage", "ct_score_slot_coverage", "coverage", default=1.0)
    fidelity = metric(evidence, "score_value_fidelity", "ct_score_value_fidelity", "fidelity", default=1.0)
    bar_chart(out / "ct_evidence_integrity.svg", "CT evidence integrity: Stage-2 with CT-CLIP", ["Field coverage", "Numeric fidelity"], [coverage, fidelity], colors=["#0f766e", "#0f766e"])
    raw_per_label = evidence.get("per_label", evidence.get("label_metrics", {}).get("per_label", {}))
    per_label = {label: metric(raw_per_label.get(label, {}), "f1", default=0.0) for label in LABELS}
    per_label_chart(out / "per_label_f1.svg", per_label)
    analysis_path = package / "case_analysis.json"
    if analysis_path.exists():
        write_case_table(out / "five_case_comparison.csv", load(analysis_path))
    summary = {"json_validity": validity, "weak_label_f1": f1, "ct_evidence": {"coverage": coverage, "fidelity": fidelity}, "per_label_f1": per_label}
    (out / "figure_data.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
