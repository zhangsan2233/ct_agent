"""Optional 2D slice review branch for Stage-2: base Qwen vision votes vs frozen CT-CLIP."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from chestct_agent.stage2_pipeline import LABELS, extract_json


VOTE_CHOICES = frozenset({"visible", "not_visible_on_slices", "insufficient_coverage"})
AUDIT_CHOICES = frozenset({"confirm", "reject", "insufficient_coverage"})
AUDIT_SYSTEM_PROMPT = (
    "You are a research-only auditor of frozen CT-CLIP chest CT scores. "
    "You are given axial paired lung/mediastinal window images AND the eight CT-CLIP probabilities. "
    "Do not make an independent diagnosis. For each label, judge only whether the CT-CLIP binary call "
    "(score >= 0.5 means positive) is supported by the supplied slices. Return compact JSON only. "
    "insufficient_coverage if the slices cannot support a confirm or reject. "
    "reject only when the slices clearly contradict the CT-CLIP binary call."
)
AGREEMENT_DECISIONS = frozenset({"agree", "partial_agree", "disagree", "unable_to_judge"})
REVIEW_SECTION_KEYS = (
    "match_summary_zh",
    "prediction_standard_zh",
    "reason_zh",
    "agreement_decision",
    "agreement_zh",
    "reader_advice_zh",
)
LABEL_ZH = {
    "arterial_wall_calcification": "动脉壁钙化",
    "atelectasis": "肺不张",
    "coronary_artery_wall_calcification": "冠状动脉壁钙化",
    "emphysema": "肺气肿",
    "lung_opacity": "肺实变/密度增高",
    "lymphadenopathy": "淋巴结肿大",
    "pulmonary_fibrotic_sequela": "肺纤维化后遗改变",
    "pulmonary_nodule": "肺结节",
}
LIMITATIONS_ZH = (
    "本 2D 评审仅基于少量轴位肺窗/纵隔窗切片，不能代表全肺三维覆盖，"
    "不能否定 CT-CLIP 主结果，不构成临床诊断。"
)
VOTE_SYSTEM_PROMPT = (
    "You are an independent chest CT slice reviewer for research only. "
    "Inspect only the supplied axial paired lung/mediastinal window images. "
    "Do not use any report text or external model scores. "
    "Return compact JSON only."
)
REVIEW_SYSTEM_PROMPT = (
    "You are a research-only auditor comparing 2D slice votes with frozen CT-CLIP scores. "
    "Write concise Chinese. Return compact JSON only with the required review fields."
)


def _window_ct(array: np.ndarray, center: float = -600.0, width: float = 1500.0) -> np.ndarray:
    low = center - width / 2
    high = center + width / 2
    clipped = np.clip(array, low, high)
    normalized = (clipped - low) / (high - low)
    return (normalized * 255).astype(np.uint8)


def render_stage2_review_slices(
    case_id: str,
    ct_path: Path,
    out_dir: Path,
    *,
    max_images: int = 6,
) -> list[Path]:
    """Render paired lung/mediastinal axial slices into ``out_dir``."""
    import nibabel as nib

    out_dir.mkdir(parents=True, exist_ok=True)
    image = nib.load(str(ct_path))
    volume = image.dataobj
    axis = int(np.argmin(volume.shape))
    count = min(max_images, volume.shape[axis])
    indices = np.linspace(
        int(volume.shape[axis] * 0.12),
        int(volume.shape[axis] * 0.88),
        num=count,
        dtype=int,
    )
    rendered: list[Path] = []
    for idx in indices:
        if axis == 0:
            slice_arr = np.asarray(volume[idx, :, :], dtype=np.float32)
        elif axis == 1:
            slice_arr = np.asarray(volume[:, idx, :], dtype=np.float32)
        else:
            slice_arr = np.asarray(volume[:, :, idx], dtype=np.float32)
        rotated = np.rot90(slice_arr)
        lung = Image.fromarray(_window_ct(rotated, center=-600, width=1500)).convert("RGB")
        mediastinal = Image.fromarray(_window_ct(rotated, center=40, width=400)).convert("RGB")
        max_pane_width = 384
        if lung.width > max_pane_width:
            ratio = max_pane_width / lung.width
            resized = (max_pane_width, max(1, int(lung.height * ratio)))
            lung = lung.resize(resized, Image.Resampling.LANCZOS)
            mediastinal = mediastinal.resize(resized, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (lung.width * 2, lung.height + 28), "black")
        canvas.paste(lung, (0, 28))
        canvas.paste(mediastinal, (lung.width, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), f"AXIAL SLICE {idx} | LUNG WINDOW", fill="white")
        draw.text((lung.width + 8, 7), "MEDIASTINAL WINDOW", fill="white")
        out_path = out_dir / f"slice_{idx:04d}_paired.jpg"
        canvas.save(out_path, quality=88, optimize=True)
        rendered.append(out_path)
    return rendered


def normalize_votes(raw_votes: list[dict[str, Any]] | None, labels: list[str] | None = None) -> list[dict[str, Any]]:
    """Ensure every label has a valid vote entry."""
    label_list = labels or LABELS
    by_name: dict[str, dict[str, Any]] = {}
    if isinstance(raw_votes, list):
        for item in raw_votes:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("label") or "").strip()
            if name not in label_list:
                continue
            vote = str(item.get("vote") or item.get("status") or "").strip()
            if vote not in VOTE_CHOICES:
                vote = "insufficient_coverage"
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            evidence_zh = str(item.get("evidence_zh") or item.get("evidence") or "").strip()
            by_name[name] = {
                "name": name,
                "vote": vote,
                "confidence": confidence,
                "evidence_zh": evidence_zh,
            }
    normalized: list[dict[str, Any]] = []
    for name in label_list:
        normalized.append(
            by_name.get(
                name,
                {
                    "name": name,
                    "vote": "insufficient_coverage",
                    "confidence": 0.0,
                    "evidence_zh": "模型未返回该项，按切片覆盖不足处理。",
                },
            )
        )
    return normalized


def normalize_audits(raw_audits: list[dict[str, Any]] | None, labels: list[str] | None = None) -> list[dict[str, Any]]:
    """Ensure every label has a valid CT-CLIP audit entry."""
    label_list = labels or LABELS
    by_name: dict[str, dict[str, Any]] = {}
    if isinstance(raw_audits, list):
        for item in raw_audits:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("label") or "").strip()
            if name not in label_list:
                continue
            verdict = str(item.get("verdict") or item.get("vote") or item.get("decision") or "").strip()
            if verdict in {"agree", "correct", "confirm", "supported"}:
                verdict = "confirm"
            elif verdict in {"disagree", "incorrect", "reject", "contradicted", "flip"}:
                verdict = "reject"
            if verdict not in AUDIT_CHOICES:
                verdict = "insufficient_coverage"
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            evidence_zh = str(item.get("evidence_zh") or item.get("evidence") or "").strip()
            by_name[name] = {
                "name": name,
                "verdict": verdict,
                "confidence": confidence,
                "evidence_zh": evidence_zh,
            }
    normalized: list[dict[str, Any]] = []
    for name in label_list:
        normalized.append(
            by_name.get(
                name,
                {
                    "name": name,
                    "verdict": "insufficient_coverage",
                    "confidence": 0.0,
                    "evidence_zh": "模型未返回该项，按切片覆盖不足处理。",
                },
            )
        )
    return normalized


def apply_audit_to_clip(
    audits: list[dict[str, Any]],
    ctclip_scores: dict[str, float],
    *,
    positive_threshold: float = 0.5,
) -> dict[str, bool]:
    """Keep the CT-CLIP binary call unless the auditor rejects it."""
    corrected: dict[str, bool] = {}
    by_name = {item["name"]: item for item in audits}
    for name in LABELS:
        clip_positive = float(ctclip_scores.get(name, 0.0)) >= positive_threshold
        verdict = by_name.get(name, {}).get("verdict", "insufficient_coverage")
        corrected[name] = (not clip_positive) if verdict == "reject" else clip_positive
    return corrected


def build_agreement(
    votes: list[dict[str, Any]],
    ctclip_scores: dict[str, float],
    *,
    positive_threshold: float = 0.5,
) -> dict[str, Any]:
    aligned: list[str] = []
    llm_visible_ctclip_low: list[str] = []
    llm_not_visible_ctclip_high: list[str] = []
    insufficient: list[str] = []
    for item in votes:
        name = item["name"]
        vote = item["vote"]
        score = float(ctclip_scores.get(name, 0.0))
        ct_positive = score >= positive_threshold
        if vote == "insufficient_coverage":
            insufficient.append(name)
        elif vote == "visible" and ct_positive:
            aligned.append(name)
        elif vote == "not_visible_on_slices" and not ct_positive:
            aligned.append(name)
        elif vote == "visible" and not ct_positive:
            llm_visible_ctclip_low.append(name)
        elif vote == "not_visible_on_slices" and ct_positive:
            llm_not_visible_ctclip_high.append(name)
    comparable = len(votes) - len(insufficient)
    if comparable == 0:
        overall = "unable_to_judge"
    elif len(aligned) == comparable:
        overall = "full"
    elif aligned:
        overall = "partial"
    else:
        overall = "poor"
    return {
        "aligned_labels": aligned,
        "llm_visible_ctclip_low": llm_visible_ctclip_low,
        "llm_not_visible_ctclip_high": llm_not_visible_ctclip_high,
        "insufficient_coverage_labels": insufficient,
        "overall_match": overall,
    }


def _default_review_sections(agreement: dict[str, Any]) -> dict[str, str]:
    overall = agreement.get("overall_match", "unable_to_judge")
    match_map = {
        "full": "相符",
        "partial": "部分相符",
        "poor": "不相符",
        "unable_to_judge": "无法判断",
    }
    decision_map = {
        "full": "agree",
        "partial": "partial_agree",
        "poor": "disagree",
        "unable_to_judge": "unable_to_judge",
    }
    return {
        "match_summary_zh": f"2D 投票与 CT-CLIP 总体{match_map.get(overall, '无法判断')}。",
        "prediction_standard_zh": (
            "仅依据抽到的轴位肺窗/纵隔窗切片上的可见征象判断；"
            "某一切片未见异常不等于全肺没有；覆盖不足时标记为 insufficient_coverage。"
        ),
        "reason_zh": "自动模板：模型未完整返回五段评审，已根据协定表生成占位说明。",
        "agreement_decision": decision_map.get(overall, "unable_to_judge"),
        "agreement_zh": "2D 评审不能覆盖或改写 CT-CLIP 主结果；请结合完整三维 CT 与报告人工复核。",
        "reader_advice_zh": "本输出仅供课程演示；必须人工复核，不能作为临床诊断依据。",
    }


def normalize_review_sections(raw: dict[str, Any] | None, agreement: dict[str, Any]) -> tuple[dict[str, str], bool]:
    defaults = _default_review_sections(agreement)
    review: dict[str, str] = {}
    incomplete = False
    source = raw if isinstance(raw, dict) else {}
    for key in REVIEW_SECTION_KEYS:
        value = source.get(key)
        if key == "agreement_decision":
            decision = str(value or defaults[key]).strip()
            if decision not in AGREEMENT_DECISIONS:
                decision = defaults[key]
                incomplete = True
            review[key] = decision
            continue
        text = str(value or "").strip()
        if not text:
            text = defaults[key]
            incomplete = True
        review[key] = text
    return review, incomplete


def format_review_zh(review: dict[str, str]) -> str:
    sections = [
        ("一、符合度", review["match_summary_zh"]),
        ("二、自己的预测标准", review["prediction_standard_zh"]),
        ("三、原因", review["reason_zh"]),
        ("四、是否同意主结果", review["agreement_zh"]),
        ("五、对阅读者的建议", review["reader_advice_zh"]),
    ]
    return "\n\n".join(f"{title}\n{body}" for title, body in sections)


def write_markdown_report(
    run_dir: Path,
    *,
    case_id: str,
    votes: list[dict[str, Any]],
    agreement: dict[str, Any],
    review: dict[str, str],
    review_zh: str,
    slice_paths: list[Path],
    ctclip_scores: dict[str, float],
) -> Path:
    lines = [
        f"# LLM 2D 对照评审 — {case_id}",
        "",
        "> 实验支路：2D 基座视觉投票对照冻结 CT-CLIP。不构成临床诊断，不能改写主结果。",
        "",
        review_zh,
        "",
        "## 2D 投票明细",
        "",
        "| 标签 | 投票 | 置信度 | 证据 |",
        "| --- | --- | ---: | --- |",
    ]
    for item in votes:
        name = item["name"]
        zh = LABEL_ZH.get(name, name)
        lines.append(
            f"| {zh} (`{name}`) | {item['vote']} | {item['confidence']:.2f} | {item['evidence_zh']} |"
        )
    lines.extend(
        [
            "",
            "## CT-CLIP 主结果分数",
            "",
            "| 标签 | CT-CLIP |",
            "| --- | ---: |",
        ]
    )
    for name in LABELS:
        lines.append(f"| {LABEL_ZH.get(name, name)} | {ctclip_scores.get(name, 0.0):.4f} |")
    lines.extend(["", "## 协定摘要", "", "```json", json.dumps(agreement, ensure_ascii=False, indent=2), "```", ""])
    lines.append("## LLM 看过的 2D 切片")
    lines.append("")
    for path in slice_paths:
        rel = path.relative_to(run_dir).as_posix()
        caption = path.stem.replace("_", " ")
        lines.append(f"### {caption}")
        lines.append("")
        lines.append(f"![{caption}]({rel})")
        lines.append("")
    report_path = run_dir / "llm_2d_review.md"
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return report_path


@dataclass
class BaseQwenVisionRuntime:
    model_dir: Path
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        self.processor = None
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.processor = AutoProcessor.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_dir,
            quantization_config=qconfig,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()

    def release(self) -> None:
        self.model = None
        self.processor = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _generate_from_messages(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        import torch

        self.load()
        assert self.processor is not None and self.model is not None and self.tokenizer is not None
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except TypeError:
                prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(prompt, return_tensors="pt")
        if isinstance(inputs, dict):
            inputs = {
                key: value.to(self.model.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        else:
            inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1] if isinstance(inputs, dict) else inputs.input_ids.shape[1]
        return self.tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)

    def generate_with_images(
        self,
        *,
        system: str,
        user_text: str,
        image_paths: list[Path],
        max_new_tokens: int = 2048,
    ) -> str:
        content: list[dict[str, Any]] = []
        for path in image_paths:
            content.append({"type": "image", "image": Image.open(path).convert("RGB")})
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        return self._generate_from_messages(messages, max_new_tokens)

    def generate_text(self, *, system: str, user_text: str, max_new_tokens: int = 1536) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]
        return self._generate_from_messages(messages, max_new_tokens)


def _vote_user_payload(labels: list[str]) -> str:
    contract = [
        {
            "name": name,
            "name_zh": LABEL_ZH.get(name, name),
            "vote": "visible|not_visible_on_slices|insufficient_coverage",
            "confidence": "0..1",
            "evidence_zh": "short Chinese rationale",
        }
        for name in labels
    ]
    return json.dumps(
        {
            "task": (
                "For each label, inspect only the supplied paired axial slices and choose one vote. "
                "visible = finding appears on these slices; not_visible_on_slices = adequately covered "
                "slices show no finding; insufficient_coverage = cannot judge from these slices. "
                "Do not claim whole-lung absence."
            ),
            "labels": contract,
            "output_schema": {"votes": contract},
        },
        ensure_ascii=False,
    )


def _review_user_payload(
    votes: list[dict[str, Any]],
    ctclip_scores: dict[str, float],
    agreement: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "task": (
                "Write the required five Chinese review fields comparing independent 2D votes with "
                "frozen CT-CLIP scores. 2D cannot overturn CT-CLIP."
            ),
            "ctclip_scores": ctclip_scores,
            "votes": votes,
            "agreement": agreement,
            "required_fields": {
                "match_summary_zh": "overall match between 2D votes and CT-CLIP",
                "prediction_standard_zh": "how you judged from limited 2D slices",
                "reason_zh": "why aligned or diverged per key labels",
                "agreement_decision": "agree|partial_agree|disagree|unable_to_judge",
                "agreement_zh": "stance on CT-CLIP primary result; cannot override it",
                "reader_advice_zh": "human review and 3D CT advice; not clinical diagnosis",
            },
        },
        ensure_ascii=False,
    )


def _audit_user_payload(ctclip_scores: dict[str, float], *, positive_threshold: float = 0.5) -> str:
    items = []
    for name in LABELS:
        score = float(ctclip_scores.get(name, 0.0))
        items.append(
            {
                "name": name,
                "name_zh": LABEL_ZH.get(name, name),
                "ctclip_score": round(score, 4),
                "ctclip_binary": "positive" if score >= positive_threshold else "negative",
                "verdict": "confirm|reject|insufficient_coverage",
                "confidence": "0..1",
                "evidence_zh": "cite visible 2D findings or say why coverage is insufficient",
            }
        )
    return json.dumps(
        {
            "task": (
                "For each label, inspect the supplied slices together with the CT-CLIP score. "
                "confirm = the 2D slices are compatible with the CT-CLIP binary call. "
                "reject = the 2D slices clearly contradict that binary call. "
                "insufficient_coverage = these slices cannot fairly confirm or reject. "
                "Do not invent whole-volume absence. Do not output an independent 8-label diagnosis."
            ),
            "positive_threshold": positive_threshold,
            "audits": items,
            "output_schema": {"audits": items},
        },
        ensure_ascii=False,
    )


def run_clip_audits(
    runtime: BaseQwenVisionRuntime,
    slice_paths: list[Path],
    ctclip_scores: dict[str, float],
    *,
    positive_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    raw = runtime.generate_with_images(
        system=AUDIT_SYSTEM_PROMPT,
        user_text=_audit_user_payload(ctclip_scores, positive_threshold=positive_threshold),
        image_paths=slice_paths,
    )
    parsed, _ = extract_json(raw)
    audits_raw = []
    if isinstance(parsed, dict):
        audits_raw = parsed.get("audits") or parsed.get("labels") or parsed.get("votes") or []
    return normalize_audits(audits_raw if isinstance(audits_raw, list) else [])


def run_independent_votes(runtime: BaseQwenVisionRuntime, slice_paths: list[Path]) -> list[dict[str, Any]]:
    raw = runtime.generate_with_images(
        system=VOTE_SYSTEM_PROMPT,
        user_text=_vote_user_payload(LABELS),
        image_paths=slice_paths,
    )
    parsed, _ = extract_json(raw)
    votes_raw = []
    if isinstance(parsed, dict):
        votes_raw = parsed.get("votes") or parsed.get("labels") or parsed.get("assessments") or []
    return normalize_votes(votes_raw if isinstance(votes_raw, list) else [])


def run_review_zh(
    runtime: BaseQwenVisionRuntime,
    votes: list[dict[str, Any]],
    ctclip_scores: dict[str, float],
    agreement: dict[str, Any],
) -> tuple[dict[str, str], bool]:
    raw = runtime.generate_text(
        system=REVIEW_SYSTEM_PROMPT,
        user_text=_review_user_payload(votes, ctclip_scores, agreement),
    )
    parsed, _ = extract_json(raw)
    return normalize_review_sections(parsed if isinstance(parsed, dict) else None, agreement)


def _relative_slice_paths(run_dir: Path, slice_paths: list[Path]) -> list[str]:
    rel_paths: list[str] = []
    for path in slice_paths:
        try:
            rel_paths.append(path.relative_to(run_dir).as_posix())
        except ValueError:
            rel_paths.append(path.as_posix())
    return rel_paths


def run_llm_2d_review(
    *,
    model_dir: Path,
    device: str,
    case_id: str,
    ct_path: Path,
    ctclip_scores: dict[str, float],
    run_dir: Path,
) -> dict[str, Any]:
    """Run optional 2D base-model review; never raises to the Stage-2 caller."""
    slice_dir = run_dir / "llm_2d_slices"
    runtime = BaseQwenVisionRuntime(model_dir=model_dir, device=device)
    try:
        slice_paths = render_stage2_review_slices(case_id, ct_path, slice_dir, max_images=6)
        if not slice_paths:
            return {
                "enabled": True,
                "backend": "local_base_qwen_vision",
                "adapter_used": False,
                "degraded": True,
                "degraded_reason": "slice_render_failed",
                "limitations_zh": LIMITATIONS_ZH,
            }
        votes = run_independent_votes(runtime, slice_paths)
        agreement = build_agreement(votes, ctclip_scores)
        review, review_incomplete = run_review_zh(runtime, votes, ctclip_scores, agreement)
        review_zh = format_review_zh(review)
        report_path = write_markdown_report(
            run_dir,
            case_id=case_id,
            votes=votes,
            agreement=agreement,
            review=review,
            review_zh=review_zh,
            slice_paths=slice_paths,
            ctclip_scores=ctclip_scores,
        )
        return {
            "enabled": True,
            "backend": "local_base_qwen_vision",
            "adapter_used": False,
            "slice_count": len(slice_paths),
            "slice_paths": _relative_slice_paths(run_dir, slice_paths),
            "votes": votes,
            "agreement": agreement,
            "review": review,
            "review_zh": review_zh,
            "report_markdown_path": report_path.relative_to(run_dir).as_posix(),
            "limitations_zh": LIMITATIONS_ZH,
            "review_incomplete": review_incomplete,
            "degraded": False,
            "degraded_reason": None,
        }
    except Exception as exc:
        return {
            "enabled": True,
            "backend": "local_base_qwen_vision",
            "adapter_used": False,
            "degraded": True,
            "degraded_reason": f"{type(exc).__name__}: {exc}",
            "limitations_zh": LIMITATIONS_ZH,
        }
    finally:
        runtime.release()
