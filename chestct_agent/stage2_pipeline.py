"""Offline, auditable CT-CLIP / CXR + Stage-2 inference pipeline for demonstrations.

This module deliberately keeps frozen image encoders and the QLoRA adapter separate:
one study is converted to eight evidence scores, then the language model produces a
strict, research-only JSON response.  It never downloads models at run time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Literal

from chestct_agent.modality_paths import (
    adapter_dir_for_modality,
    cxr_adapter_dir,
    ct_adapter_dir,
    resolve_qwen_model_dir,
)
from chestct_agent.stage2_report import build_report_zh

from chestct_agent.stage2_contract import DISCLAIMER, LABELS, SYSTEM_PROMPT, CXR_DISCLAIMER
ModalityId = Literal["ct_chest", "cxr_chest"]


@dataclass(frozen=True)
class Stage2Paths:
    model_dir: Path
    adapter_dir: Path
    ctclip_checkpoint: Path
    ctclip_source: Path
    text_model_dir: Path
    modality: ModalityId = "ct_chest"

    @classmethod
    def defaults(cls, root: Path) -> "Stage2Paths":
        return cls.for_modality(root, "ct_chest")

    @classmethod
    def for_modality(cls, root: Path, modality: ModalityId) -> "Stage2Paths":
        return cls(
            model_dir=resolve_qwen_model_dir(root),
            adapter_dir=adapter_dir_for_modality(root, modality),
            ctclip_checkpoint=root / "models" / "ctclip" / "CT-CLIP_v2.pt",
            ctclip_source=root / "external" / "CT-CLIP-main",
            text_model_dir=root / "models" / "cxrbert",
            modality=modality,
        )


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Extract the first JSON object without silently repairing model output."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = candidate.find("{")
    if start < 0:
        return None, "No JSON object was generated."
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc.msg}."
    return (value, None) if isinstance(value, dict) else (None, "Generated JSON root is not an object.")


def validate_stage2_json(value: dict[str, Any] | None, case_id: str, scores: dict[str, float]) -> list[str]:
    """Validate the compact Stage-2 contract and evidence fidelity."""
    if value is None:
        return ["No parseable JSON output."]
    errors: list[str] = []
    if value.get("case_id") != case_id:
        errors.append("case_id is missing or does not match the input.")
    labels = value.get("labels")
    if not isinstance(labels, list) or len(labels) != len(LABELS):
        return errors + [f"labels must contain exactly {len(LABELS)} items."]
    names = [item.get("name") for item in labels if isinstance(item, dict)]
    if set(names) != set(LABELS) or len(names) != len(LABELS):
        errors.append("labels do not match the required eight-label schema.")
    for item in labels:
        if not isinstance(item, dict):
            errors.append("A label item is not an object.")
            continue
        name = item.get("name")
        if item.get("status") not in {"positive", "negative", "uncertain"}:
            errors.append(f"{name}: invalid status.")
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            errors.append(f"{name}: confidence must be in [0, 1].")
        ct_score = item.get("ctclip_score")
        if not isinstance(ct_score, (int, float)):
            errors.append(f"{name}: CT-CLIP score is missing.")
        elif name in scores and abs(float(ct_score) - scores[name]) > 1e-4:
            errors.append(f"{name}: CT-CLIP score differs from the supplied evidence.")
    if value.get("need_human_review") is not True:
        errors.append("need_human_review must be true for this research workflow.")
    return errors


class Stage2Agent:
    """Lazy model holder for repeated single-case or batch inference."""

    def __init__(self, paths: Stage2Paths, device: str = "cuda:0", max_new_tokens: int = 1024):
        self.paths = paths
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.modality = paths.modality
        self.ctclip: Any = None
        self.cxr_encoder: Any = None
        self.tokenizer = None
        self.model = None

    def readiness_errors(self) -> list[str]:
        expected = {"Qwen model": self.paths.model_dir, "Stage-2 adapter": self.paths.adapter_dir}
        errors = [f"{name} not found: {path}" for name, path in expected.items() if not path.exists()]
        if self.modality == "ct_chest":
            for name, path in {
                "CT-CLIP checkpoint": self.paths.ctclip_checkpoint,
                "CT-CLIP source": self.paths.ctclip_source,
                "CXR-BERT": self.paths.text_model_dir,
            }.items():
                if not path.exists():
                    errors.append(f"{name} not found: {path}")
        if self.modality == "cxr_chest":
            from chestct_agent.cxr_encoder import CxrEncoderRuntime

            txv_error = CxrEncoderRuntime(device=self.device).readiness_error()
            if txv_error:
                errors.append(txv_error)
        return errors

    def _ctclip_scores(self, ct_path: Path) -> dict[str, float]:
        os.environ.update(
            {
                "CTCLIP_TEXT_MODEL_DIR": str(self.paths.text_model_dir),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        from chestct_agent.ctclip import CtClipRuntime

        self.ctclip = CtClipRuntime(
            self.paths.ctclip_checkpoint,
            self.paths.ctclip_source,
            self.device,
            use_fp16=True,
            variant="zeroshot",
        )
        raw = self.ctclip.predict(str(ct_path))
        return {label: round(float(raw[label]), 4) for label in LABELS}

    def _cxr_scores(self, image_path: Path) -> tuple[dict[str, float], dict[str, Any]]:
        from chestct_agent.cxr_encoder import CxrEncoderRuntime

        self.cxr_encoder = CxrEncoderRuntime(device=self.device)
        payload = self.cxr_encoder.predict(image_path)
        return payload["mapped_scores"], payload

    def _release_ctclip(self) -> None:
        self.ctclip = None
        self._release_gpu()

    def _release_cxr_encoder(self) -> None:
        self.cxr_encoder = None
        self._release_gpu()

    def _release_gpu(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _release_llm(self) -> None:
        """Avoid holding both encoders and Qwen in a 24 GB demo GPU."""
        self.model = None
        self.tokenizer = None
        self._release_gpu()

    def _load_llm(self) -> None:
        if self.model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        processor = AutoProcessor.from_pretrained(
            self.paths.model_dir, local_files_only=True, trust_remote_code=True
        )
        self.tokenizer = processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForImageTextToText.from_pretrained(
            self.paths.model_dir,
            quantization_config=qconfig,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(
            base, self.paths.adapter_dir, local_files_only=True
        ).eval()

    def _generate(self, payload: dict[str, Any]) -> str:
        self._load_llm()
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(generated[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

    def fuse_scores_and_report(
        self,
        *,
        case_id: str,
        report_text: str,
        scores: dict[str, float],
        evidence_encoder: str,
        modality: ModalityId,
        encoder_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "case_id": case_id,
            "task": "Integrate report and CT-CLIP evidence into the required compact JSON.",
            "report_impression": report_text.strip()[:3500],
            "ctclip_scores": scores,
            "ctclip_score_definition": "probability of the finding being present",
            "label_provenance": "research-only; report-derived weak labels, not radiologist-adjudicated ground truth",
            "evidence_encoder": evidence_encoder,
            "modality": modality,
        }
        raw = self._generate(payload)
        stage2_json, parse_error = extract_json(raw)
        validation_errors = ([parse_error] if parse_error else []) + validate_stage2_json(
            stage2_json, case_id, scores
        )
        report_zh = build_report_zh(
            stage2_json, modality=modality, report_text=report_text, scores=scores
        )
        positive = (
            []
            if not stage2_json
            else [
                item["name"]
                for item in stage2_json.get("labels", [])
                if isinstance(item, dict) and item.get("status") == "positive"
            ]
        )
        return {
            "stage2_json": stage2_json,
            "raw_stage2_output": raw,
            "validation": {
                "parseable_json": parse_error is None,
                "schema_valid": not validation_errors,
                "errors": validation_errors,
            },
            "report_zh": report_zh,
            "summary_zh": "阳性标签：" + ("、".join(positive) if positive else "未识别或需人工复核"),
            "encoder_details": encoder_details or {},
        }

    def analyze(
        self,
        *,
        case_id: str,
        ct_path: Path,
        report_text: str,
        run_dir: Path | None = None,
        enable_llm_2d_review: bool = False,
    ) -> dict[str, Any]:
        if not ct_path.is_file():
            raise FileNotFoundError(f"CT volume not found: {ct_path}")
        if not report_text.strip():
            raise ValueError("Report text is empty.")
        errors = self.readiness_errors()
        if errors:
            raise FileNotFoundError("; ".join(errors))
        started = time.perf_counter()
        self._release_llm()
        scores = self._ctclip_scores(ct_path)
        self._release_ctclip()
        llm_2d_review: dict[str, Any] = {"enabled": False}
        if enable_llm_2d_review:
            if run_dir is None:
                llm_2d_review = {
                    "enabled": True,
                    "degraded": True,
                    "degraded_reason": "run_dir_required_for_llm_2d_review",
                }
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                from chestct_agent.stage2_llm_2d_review import run_llm_2d_review

                llm_2d_review = run_llm_2d_review(
                    model_dir=self.paths.model_dir,
                    device=self.device,
                    case_id=case_id,
                    ct_path=ct_path,
                    ctclip_scores=scores,
                    run_dir=run_dir,
                )
            self._release_llm()
        fused = self.fuse_scores_and_report(
            case_id=case_id,
            report_text=report_text,
            scores=scores,
            evidence_encoder="frozen_ctclip_v2",
            modality="ct_chest",
        )
        result = {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + re.sub(r"[^A-Za-z0-9_-]+", "_", case_id)[:48],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "modality": {"id": "ct_chest", "encoder": "frozen_ctclip_v2 + stage2_qlora"},
            "input": {"case_id": case_id, "ct_path": str(ct_path.resolve()), "report_text": report_text.strip()},
            "ctclip_scores": scores,
            "image_scores": None,
            **fused,
            "provenance": {
                "paths": {key: str(value) for key, value in asdict(self.paths).items()},
                "adapter_dir": str(self.paths.adapter_dir),
                "device": self.device,
                "max_new_tokens": self.max_new_tokens,
                "ctclip_frozen": True,
                "evidence_encoder": "frozen_ctclip_v2",
            },
            "warning": DISCLAIMER,
            "llm_2d_review": llm_2d_review,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return result

    def analyze_cxr(
        self,
        *,
        case_id: str,
        image_path: Path,
        report_text: str,
        run_dir: Path | None = None,
    ) -> dict[str, Any]:
        if not image_path.is_file():
            raise FileNotFoundError(f"CXR image not found: {image_path}")
        if not report_text.strip():
            raise ValueError("Report text is empty.")
        errors = self.readiness_errors()
        if errors:
            raise FileNotFoundError("; ".join(errors))
        started = time.perf_counter()
        from chestct_agent.modalities import inspect_cxr_image

        image_qc = inspect_cxr_image(image_path)
        self._release_llm()
        scores, encoder_details = self._cxr_scores(image_path)
        self._release_cxr_encoder()
        fused = self.fuse_scores_and_report(
            case_id=case_id,
            report_text=report_text,
            scores=scores,
            evidence_encoder=str(encoder_details.get("encoder", "torchxrayvision")),
            modality="cxr_chest",
            encoder_details=encoder_details,
        )
        result = {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + re.sub(r"[^A-Za-z0-9_-]+", "_", case_id)[:48],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "modality": {
                "id": "cxr_chest",
                "encoder": "torchxrayvision_mapped + cxr_stage2_qlora",
                "status": "schematic",
            },
            "input": {
                "case_id": case_id,
                "image_path": str(image_path.resolve()),
                "report_text": report_text.strip(),
            },
            "image_qc": image_qc,
            "ctclip_scores": scores,
            "image_scores": scores,
            **fused,
            "provenance": {
                "paths": {key: str(value) for key, value in asdict(self.paths).items()},
                "adapter_dir": str(self.paths.adapter_dir),
                "device": self.device,
                "max_new_tokens": self.max_new_tokens,
                "schematic": True,
                "evidence_encoder": encoder_details.get("encoder"),
                "limited_labels": encoder_details.get("limited_labels", []),
            },
            "warning": CXR_DISCLAIMER,
            "need_human_review": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return result
