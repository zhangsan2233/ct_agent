"""Create a local, radiologist-ready CT-CLIP error-review package.

The script selects the most confident false positive and false negative for
each target label.  It renders representative axial CT slices in lung and
mediastinal windows, and writes a CSV that is deliberately left for human
adjudication.  It does not make a diagnostic conclusion from the images.
"""
from __future__ import annotations

import argparse
import array
import csv
import gzip
import json
import struct
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LABELS = [
    "arterial_wall_calcification", "atelectasis",
    "coronary_artery_wall_calcification", "emphysema", "lung_opacity",
    "lymphadenopathy", "pulmonary_fibrotic_sequela", "pulmonary_nodule",
]
DTYPES = {
    2: "B", 4: "h", 8: "i", 16: "f", 64: "d", 256: "b",
    512: "H", 768: "I", 1024: "q", 1280: "Q",
}
REPORT_PATTERNS = {
    "arterial_wall_calcification": ("calcif", "atherom", "atheroscler"),
    "atelectasis": ("atelect"),
    "coronary_artery_wall_calcification": ("coronary", "calcif", "atherom", "atheroscler"),
    "emphysema": ("emphysem"),
    "lung_opacity": ("opacity", "ground-glass", "consolidat", "infiltrat", "pneumonia"),
    "lymphadenopathy": ("lymph node", "lymphaden"),
    "pulmonary_fibrotic_sequela": ("fibrot", "sequela", "reticul", "interlobular sept"),
    "pulmonary_nodule": ("nodule", "nodul"),
}


def read_nifti_header(path: Path) -> tuple[str, tuple[int, int, int], int, int, float, float, float]:
    with gzip.open(path, "rb") as handle:
        header = handle.read(348)
    endian = "<" if struct.unpack("<I", header[:4])[0] == 348 else ">"
    if struct.unpack(endian + "I", header[:4])[0] != 348:
        raise ValueError("Not a NIfTI-1 file")
    dims = struct.unpack(endian + "8h", header[40:56])
    if dims[0] < 3:
        raise ValueError(f"Expected 3D volume, got dimensions {dims}")
    datatype, bitpix = struct.unpack(endian + "2h", header[70:74])
    offset, slope, intercept = struct.unpack(endian + "3f", header[108:120])
    if datatype not in DTYPES:
        raise ValueError(f"Unsupported NIfTI datatype {datatype}")
    return endian, tuple(int(x) for x in dims[1:4]), datatype, bitpix, offset, slope, intercept


def axial_slices(path: Path, fractions: list[float]) -> tuple[list[array.array], int, int]:
    """Read selected z slices without materialising an entire compressed volume."""
    endian, (nx, ny, nz), datatype, bitpix, offset, slope, intercept = read_nifti_header(path)
    indices = sorted({max(0, min(nz - 1, round((nz - 1) * fraction))) for fraction in fractions})
    bytes_per_slice = nx * ny * (bitpix // 8)
    output: list[array.array] = []
    with gzip.open(path, "rb") as handle:
        for index in indices:
            handle.seek(int(offset) + index * bytes_per_slice)
            raw = handle.read(bytes_per_slice)
            if len(raw) != bytes_per_slice:
                raise ValueError(f"Unexpected end of volume at axial slice {index}")
            image = array.array(DTYPES[datatype])
            image.frombytes(raw)
            if (endian == ">") == (struct.pack("=H", 1) == b"\\x01\\x00"):
                image.byteswap()
            if slope not in (0.0, 1.0) or intercept:
                image = array.array("f", ((value * (slope if slope else 1.0) + intercept) for value in image))
            output.append(image)
    return output, nx, ny


def window(image: array.array, center: float, width: float) -> bytes:
    low, high = center - width / 2, center + width / 2
    return bytes(max(0, min(255, round((value - low) * 255 / (high - low)))) for value in image)


def select_cases(errors: list[dict[str, str]]) -> list[dict[str, str]]:
    chosen = []
    for label in LABELS:
        candidates = [row for row in errors if row["label"] == label]
        for error_type, truth in (("false_positive", "0"), ("false_negative", "1")):
            matches = [row for row in candidates if row["ground_truth"] == truth]
            if not matches:
                continue
            def confidence(row: dict[str, str]) -> float:
                probability = float(row["probability"])
                return probability if error_type == "false_positive" else 1 - probability
            row = max(matches, key=confidence).copy()
            row["error_type"] = error_type
            row["confidence"] = f"{confidence(row):.6f}"
            row["confidence_tier"] = "high (>=0.80)" if confidence(row) >= 0.80 else "ranked disagreement (<0.80)"
            chosen.append(row)
    return chosen


def report_triage(label: str, error_type: str, report: str) -> tuple[str, str]:
    normalized = report.lower()
    patterns = REPORT_PATTERNS[label]
    # Coronary calcification requires both a coronary context and a calcification
    # context; the remaining labels use any one of their terms.
    mentioned = ("coronary" in normalized and any(term in normalized for term in patterns[1:])) if label == "coronary_artery_wall_calcification" else any(term in normalized for term in patterns)
    if not report.strip() or report.strip().lower() == "not given.":
        return "insufficient_report_evidence", "No usable report impression is available; CT expert review is required."
    if error_type == "false_positive" and mentioned:
        return "suspected_weak_label_or_ontology_mismatch", "The report uses label-related language while the report-derived ground truth is negative."
    if error_type == "false_negative" and mentioned:
        return "suspected_model_miss", "The report uses label-related language and the model is below threshold."
    if error_type == "false_negative":
        return "suspected_weak_report_label", "The ground truth is positive but this report impression lacks label-related language."
    return "suspected_model_false_positive", "The report impression lacks label-related language while the model is above threshold."


def render_case(row: dict[str, str], volume: Path, output: Path) -> None:
    slices, nx, ny = axial_slices(volume, [0.25, 0.38, 0.50, 0.62, 0.75])
    font = ImageFont.load_default()
    tile_size, margin, header, footer = 260, 10, 50, 100
    canvas = Image.new("RGB", (margin + len(slices) * (tile_size + margin), header + 2 * (tile_size + margin) + footer), "white")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(slices):
        x = margin + index * (tile_size + margin)
        for row_index, (center, width) in enumerate(((-600, 1500), (40, 400))):
            rendered = Image.frombytes("L", (nx, ny), window(image, center, width)).transpose(Image.Transpose.ROTATE_90).convert("RGB")
            rendered.thumbnail((tile_size, tile_size))
            y = header + row_index * (tile_size + margin)
            canvas.paste(rendered, (x + (tile_size - rendered.width) // 2, y + (tile_size - rendered.height) // 2))
        draw.text((x, header - 14), f"Axial {index + 1}", fill="black", font=font)
    draw.text((2, header + tile_size // 2), "Lung", fill="black", font=font)
    draw.text((2, header + tile_size + margin + tile_size // 2), "Mediastinal", fill="black", font=font)
    probability = float(row["probability"])
    title = (f"{row['case_id']} | {row['label']} | {row['error_type'].replace('_', ' ')} | "
             f"ground truth={row['ground_truth']}, CT-CLIP probability={probability:.3f}, confidence={float(row['confidence']):.3f}")
    draw.text((margin, 10), title, fill="black", font=font)
    report = "Report impression: " + " ".join(row.get("report_impression", "").split())
    draw.multiline_text((margin, header + 2 * (tile_size + margin) + 4), "\n".join(textwrap.wrap(report, width=200)), fill="black", font=font, spacing=2)
    canvas.save(output)


def write_review_html(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a self-contained local reviewer page; no web server is needed."""
    data = []
    for row in rows:
        item = dict(row)
        item["montage_relative"] = "montages/" + Path(row["ct_montage"]).name
        data.append(item)
    serialized = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>CT-CLIP 本地错误病例复核</title>
<style>body{{font:15px system-ui;margin:24px;background:#f5f6f8}} .card{{background:#fff;padding:18px;margin:18px 0;border-radius:8px;box-shadow:0 1px 4px #bbb}} img{{width:100%;max-width:1300px;border:1px solid #ddd}} textarea{{width:100%;min-height:60px}} select{{padding:5px}} .meta{{color:#333}} code{{background:#eee;padding:2px 4px}} button{{padding:10px 14px;font-size:15px}}</style>
<h1>CT-CLIP 错误病例人工复核</h1>
<p>这些是与 <code>valid_predicted_labels.csv</code>（报告自动生成伪标签）不一致的病例，而非 CT 金标准错误。请查看图像和报告后选择裁定；完成后点击下载。</p>
<button onclick="downloadReview()">下载填写后的 adjudicated_review.csv</button><div id="cases"></div>
<script>const rows={serialized}; const choices=['pending_expert_review','model_false_positive','model_false_negative','weak_report_label_or_ontology_mismatch','indeterminate'];
const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
document.querySelector('#cases').innerHTML=rows.map((r,i)=>`<section class="card"><h2>${{i+1}}. ${{esc(r.label)}} — ${{esc(r.error_type)}}</h2><p class="meta">病例：${{esc(r.case_id)}}；伪标签=${{esc(r.ground_truth)}}；模型概率=${{Number(r.model_probability).toFixed(3)}}；置信度=${{Number(r.confidence).toFixed(3)}}（${{esc(r.confidence_tier)}}）</p><img src="${{esc(r.montage_relative)}}" alt="CT montage"><p><b>报告：</b>${{esc(r.report_impression)}}</p><p><b>自动报告分流：</b>${{esc(r.preliminary_report_triage)}} — ${{esc(r.triage_evidence)}}</p><label>人工 CT 所见<br><textarea id="finding-${{i}}"></textarea></label><p><label>裁定 <select id="decision-${{i}}">${{choices.map(x=>`<option value="${{x}}">${{x}}</option>`).join('')}}</select></label></p><label>依据/关键层面<br><textarea id="reason-${{i}}"></textarea></label></section>`).join('');
function csvCell(v){{return '"'+String(v??'').replaceAll('"','""')+'"'}} function downloadReview(){{const headers=[...Object.keys(rows[0]),'human_ct_finding','adjudication','rationale'];const out=[headers.map(csvCell).join(',')];rows.forEach((r,i)=>{{const x={{...r,human_ct_finding:document.querySelector('#finding-'+i).value,adjudication:document.querySelector('#decision-'+i).value,rationale:document.querySelector('#reason-'+i).value}};out.push(headers.map(k=>csvCell(x[k])).join(','))}});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out.join('\\n')],{{type:'text/csv;charset=utf-8'}}));a.download='adjudicated_review.csv';a.click();}}</script></html>"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    montage_dir = args.out_dir / "montages"
    montage_dir.mkdir(exist_ok=True)
    with args.errors.open(encoding="utf-8", newline="") as handle:
        selected = select_cases(list(csv.DictReader(handle)))
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        manifest = {row["case_id"]: row for row in csv.DictReader(handle)}
    review_rows = []
    for row in selected:
        source = manifest.get(row["case_id"])
        if source is None:
            raise ValueError(f"Case absent from manifest: {row['case_id']}")
        volume = args.dataset_dir / source["volume_name"]
        if not volume.exists():
            raise FileNotFoundError(volume)
        filename = f"{row['label']}__{row['error_type']}__{row['case_id']}.png"
        montage_path = montage_dir / filename
        if not montage_path.exists():
            render_case(row, volume, montage_path)
        preliminary, evidence = report_triage(row["label"], row["error_type"], row["report_impression"])
        review_rows.append({
            "label": row["label"], "case_id": row["case_id"], "study_id": row["study_id"],
            "error_type": row["error_type"], "ground_truth": row["ground_truth"],
            "label_provenance": "CT-RATE valid_predicted_labels.csv (report-derived pseudo-label; not radiologist-adjudicated CT ground truth)",
            "model_probability": row["probability"], "confidence": row["confidence"],
            "confidence_tier": row["confidence_tier"], "report_impression": row["report_impression"],
            "ct_montage": str(montage_path.resolve()),
            "preliminary_report_triage": preliminary,
            "triage_evidence": evidence,
            "human_ct_finding": "",
            "adjudication": "pending_expert_review",
            "rationale": "",
        })
    fields = list(review_rows[0]) if review_rows else []
    with (args.out_dir / "review_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)
    summary = {
        "selected_cases": len(review_rows),
        "selection": "Top-confidence false positive and false negative per label; rows below 0.80 are retained as ranked disagreements because no high-confidence example exists for that label/type.",
        "label_provenance": "The manifest was built from data/dataset/multi_abnormality_labels/valid_predicted_labels.csv. These are report-derived predicted labels, so this review assesses CT-CLIP disagreement with weak labels rather than definitive CT diagnostic error.",
        "adjudication_values": ["pending_expert_review", "model_false_positive", "model_false_negative", "weak_report_label_or_ontology_mismatch", "indeterminate"],
    }
    (args.out_dir / "README.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review_html(args.out_dir / "review.html", review_rows)
    guide = """# CT-CLIP 错误病例人工复核包

每行对应一个标签的一例最高置信假阳性或假阴性；`ct_montage` 给出同一 CT 的肺窗和纵隔窗代表性轴位层面，`report_impression` 为原始报告印象。

**标签来源限制：** 本验证集清单使用 `valid_predicted_labels.csv`，即从报告自动生成的伪标签，而非放射科医师逐例阅片后的 CT 金标准。因此“假阳性/假阴性”仅表示与伪标签不一致；在人工阅片前，不能把它们称为模型诊断错误。

请由具备胸部 CT 阅片能力的人员填写：

1. `human_ct_finding`：CT 是否支持该标签及关键层面/依据；
2. `adjudication`：`model_false_positive`、`model_false_negative`、`weak_report_label_or_ontology_mismatch` 或 `indeterminate`；
3. `rationale`：简短说明。`preliminary_report_triage` 仅基于报告关键词自动生成，不能替代影像判读。

注意：并非每个标签/方向都有概率 >=0.80 的样本。此包为保证每标签都有可审查样本，保留了该方向置信度最高但低于 0.80 的不一致病例。
"""
    (args.out_dir / "REVIEW_GUIDE.md").write_text(guide, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
