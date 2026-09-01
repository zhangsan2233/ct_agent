"""Frozen PatchChestCT-style Agent + tools + audited-memory inference path."""

from __future__ import annotations

import asyncio
import json
import hashlib
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Awaitable, Callable

import numpy as np
import httpx
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from chestct_agent.config import Settings
from chestct_agent.llm import QwenClient
from chestct_agent.schemas import (
    CtAttributionArtifact,
    LabelPrediction,
    ModelAttributionEvidence,
)
from chestct_agent.tools.ct_attribution import CtAttributionTool


LABELS = (
    "arterial_wall_calcification",
    "atelectasis",
    "bronchiectasis",
    "consolidation",
    "coronary_wall_calcification",
    "hiatal_hernia",
    "lung_opacity",
    "lymphadenopathy",
    "pericardial_effusion",
)
LABEL_ZH = {
    "arterial_wall_calcification": "动脉壁钙化",
    "atelectasis": "肺不张",
    "bronchiectasis": "支气管扩张",
    "consolidation": "肺实变",
    "coronary_wall_calcification": "冠状动脉壁钙化",
    "hiatal_hernia": "食管裂孔疝",
    "lung_opacity": "肺部密度增高影",
    "lymphadenopathy": "淋巴结肿大",
    "pericardial_effusion": "心包积液",
}
CTCLIP_LABEL = {"coronary_wall_calcification": "coronary_artery_wall_calcification"}
LABEL_ALIASES = {
    **{label: label for label in LABELS},
    **{name_zh: label for label, name_zh in LABEL_ZH.items()},
    "coronary_artery_wall_calcification": "coronary_wall_calcification",
}

FIRST_PASS_SYSTEM = (
    "You are performing a research-only blinded chest CT image review. Inspect only the supplied "
    "uniformly sampled axial CT images. Each tile shows the same slice in lung and mediastinal "
    "windows. For every requested finding make one binary positive or negative decision; never "
    "use uncertain. Write visible_evidence in concise Chinese. Return valid JSON with public "
    "image evidence, not private chain-of-thought."
)
TOOL_SYSTEM = (
    "You are the adjudication stage of a research chest CT agent. Reinspect the supplied lung and "
    "mediastinal sheets. You receive a prior visual answer and scores from a fallible independent "
    "CT-CLIP tool. Decide every finding as positive or negative. Cite visible slice indices when "
    "changing an answer. Write visible_evidence in concise Chinese. Return valid JSON with public evidence only."
)
MEMORY_SYSTEM = (
    "You are doing a second-pass blinded CT review. Compact memories are fallible inspection "
    "instructions, never patient answers. Reinspect the current images. Return a proposed positive "
    "or negative status, confidence, memory IDs used, at least two visible slice indices for any "
    "change, and concise Chinese current-image evidence. Return JSON only."
)


class ValidatedLabelResult(BaseModel):
    name: str
    name_zh: str
    status: str
    confidence: float
    evidence: str = ""
    initial_status: str
    ctclip_score: float
    memory_proposed_status: str | None = None
    memory_change_accepted: bool = False
    gate_reasons: list[str] = Field(default_factory=list)


class ValidatedStage(BaseModel):
    name: str
    status: str
    summary: str
    latency_ms: float


class ValidatedMemoryResponse(BaseModel):
    case_id: str
    mode: str = "patchchestct_9label_agent_tools_memory"
    labels: list[ValidatedLabelResult]
    sheet_paths: list[str]
    stages: list[ValidatedStage]
    retrieved_memories: list[dict[str, Any]]
    accepted_memory_changes: int
    total_latency_ms: float
    benchmark: dict[str, Any]
    model_attributions: dict[str, ModelAttributionEvidence] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _window(array: np.ndarray, center: float, width: float) -> Image.Image:
    low = center - width / 2.0
    scaled = np.clip((array - low) / width, 0.0, 1.0)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L").convert("RGB")


def render_blind_sheets(ct_path: Path, output_dir: Path, sampled_slices: int = 24) -> list[Path]:
    import nibabel as nib

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("sheet_*.jpg"))
    if len(existing) == 6:
        return existing
    image = nib.load(str(ct_path))
    volume = image.dataobj
    axis = int(np.argmin(volume.shape))
    indices = np.linspace(
        max(0, int(volume.shape[axis] * 0.04)),
        min(volume.shape[axis] - 1, int(volume.shape[axis] * 0.96)),
        num=sampled_slices,
        dtype=int,
    )
    sheets: list[Path] = []
    for sheet_number, start in enumerate(range(0, len(indices), 4), start=1):
        sheet = Image.new("RGB", (1024, 568), "black")
        for offset, index in enumerate(indices[start : start + 4]):
            if axis == 0:
                array = np.asarray(volume[index, :, :], dtype=np.float32)
            elif axis == 1:
                array = np.asarray(volume[:, index, :], dtype=np.float32)
            else:
                array = np.asarray(volume[:, :, index], dtype=np.float32)
            array = np.rot90(array)
            lung = _window(array, -600.0, 1500.0)
            mediastinal = _window(array, 40.0, 400.0)
            ratio = 256 / max(1, lung.width)
            pane_size = (256, max(1, min(260, int(lung.height * ratio))))
            lung = lung.resize(pane_size, Image.Resampling.LANCZOS)
            mediastinal = mediastinal.resize(pane_size, Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (512, 284), "black")
            tile.paste(lung, (0, 24))
            tile.paste(mediastinal, (256, 24))
            draw = ImageDraw.Draw(tile)
            draw.text((6, 6), f"AXIAL {int(index)} | LUNG", fill="white")
            draw.text((262, 6), "MEDIASTINAL", fill="white")
            sheet.paste(tile, ((offset % 2) * 512, (offset // 2) * 284))
        path = output_dir / f"sheet_{sheet_number:02d}.jpg"
        sheet.save(path, quality=90, optimize=True)
        sheets.append(path)
    return sheets


def _normalize(raw: Any, *, proposed: bool = False) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}

    items: Any = [raw] if raw.get("label") or raw.get("name") else []
    if not items:
        for key in (
            "assessment",
            "assessments",
            "labels",
            "results",
            "findings",
            "predictions",
            "output",
        ):
            candidate = raw.get(key)
            if isinstance(candidate, (list, dict)) and candidate:
                items = (
                    [candidate]
                    if isinstance(candidate, dict)
                    and (candidate.get("label") or candidate.get("name"))
                    else candidate
                )
                break
    if not items and any(str(key) in LABEL_ALIASES for key in raw):
        items = raw
    if isinstance(items, dict):
        items = [{"label": key, **value} for key, value in items.items() if isinstance(value, dict)]
    result: dict[str, dict[str, Any]] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        raw_label = str(
            item.get("label")
            or item.get("name")
            or item.get("finding")
            or item.get("disease")
            or ""
        ).strip()
        normalized_label = raw_label.lower().replace("-", "_").replace(" ", "_")
        label = LABEL_ALIASES.get(raw_label) or LABEL_ALIASES.get(normalized_label)
        status_key = "proposed_status" if proposed else "status"
        raw_status = (
            item.get(status_key)
            if item.get(status_key) is not None
            else item.get("status", item.get("decision", item.get("prediction")))
        )
        if isinstance(raw_status, bool):
            status = "positive" if raw_status else "negative"
        else:
            status = str(raw_status or "").lower().strip()
            status = {
                "阳性": "positive",
                "阴性": "negative",
                "present": "positive",
                "absent": "negative",
                "yes": "positive",
                "no": "negative",
                "true": "positive",
                "false": "negative",
            }.get(status, status)
        if label not in LABELS or status not in {"positive", "negative"}:
            continue
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        indices: list[int] = []
        for value in item.get("supporting_slice_indices", []) or []:
            try:
                indices.append(int(value))
            except (TypeError, ValueError):
                pass
        result[label] = {
            "label": label,
            "status": status,
            "confidence": confidence,
            "visible_evidence": str(item.get("visible_evidence") or item.get("evidence") or "")[:500],
            "memory_ids_used": [str(value) for value in item.get("memory_ids_used", []) or []],
            "supporting_slice_indices": sorted(set(indices)),
        }
    return result


class ValidatedMemoryPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.qwen = QwenClient(settings)
        project_root = Path(__file__).resolve().parents[1]
        artifact_memory_path = (
            Path(settings.artifact_dir)
            / "patchchestct_qwen_tools_100_50/tool_memory/audited_candidates.json"
        )
        artifact_metrics_path = (
            Path(settings.artifact_dir)
            / "patchchestct_qwen_tools_100_50/rerun_valid/agent_tools_memory_calibrated/metrics.json"
        )
        packaged_root = project_root / "reproducibility/validated_memory"
        self.memory_path = (
            artifact_memory_path
            if artifact_memory_path.is_file()
            else packaged_root / "audited_candidates.json"
        )
        self.metrics_path = (
            artifact_metrics_path
            if artifact_metrics_path.is_file()
            else packaged_root / "metrics.json"
        )
        self.vision_model = os.getenv(
            "VALIDATED_MEMORY_VISION_MODEL",
            "qwen/qwen3-vl-30b-a3b-instruct",
        )
        self.attribution_tool = CtAttributionTool(settings)

    async def _vision_review(
        self,
        system_prompt: str,
        user_prompt: str,
        sheets: list[Path],
        *,
        proposed: bool = False,
    ) -> tuple[Any, dict[str, dict[str, Any]]]:
        failures: list[str] = []
        last_call = None
        for attempt in range(3):
            last_call = await self.qwen.chat_json_with_images(
                system_prompt,
                user_prompt,
                [str(path) for path in sheets],
                {},
                max_tokens=3072,
                model=self.vision_model,
            )
            normalized = _normalize(last_call.value, proposed=proposed)
            if last_call.used_remote and len(normalized) == len(LABELS):
                return last_call, normalized
            reason = last_call.fallback_reason or "empty_or_incomplete_json"
            failures.append(f"attempt_{attempt + 1}:{reason}:labels={len(normalized)}")
            if attempt < 2:
                await asyncio.sleep(1.0 + attempt)
        raise RuntimeError("; ".join(failures))

    async def _single_label_review(
        self,
        system_prompt: str,
        finding: dict[str, Any],
        sheets: list[Path],
    ) -> dict[str, Any]:
        label = str(finding["label"])
        failures: list[str] = []
        user_prompt = json.dumps(
            {
                "task": "Recheck exactly one finding after reading the CT classifier score",
                "finding": finding,
                "required_label": label,
                "output_schema": {
                    "label": label,
                    "status": "positive|negative",
                    "confidence": "0..1",
                    "visible_evidence": "concise Chinese evidence from current slices",
                    "supporting_slice_indices": [1, 2],
                },
                "constraint": "Return exactly this one label as one JSON object. Do not omit it.",
            },
            ensure_ascii=False,
        )
        for attempt in range(2):
            call = await self.qwen.chat_json_with_images(
                system_prompt,
                user_prompt,
                [str(path) for path in sheets],
                {},
                max_tokens=1024,
                model=self.vision_model,
            )
            normalized = _normalize(call.value)
            if call.used_remote and label in normalized:
                return normalized[label]
            reason = call.fallback_reason or "missing_required_label"
            failures.append(f"attempt_{attempt + 1}:{reason}")
            if attempt == 0:
                await asyncio.sleep(1.0)
        raise RuntimeError(f"{label}:" + ";".join(failures))

    def _memories(self) -> list[dict[str, Any]]:
        payload = json.loads(self.memory_path.read_text(encoding="utf-8"))
        return [
            item for item in payload.get("memory_group_audits", [])
            if item.get("candidate_usable_for_retrieval")
            and int(item.get("support_count", 0)) >= 3
            and str(item.get("memory_summary_text", "")).strip()
        ]

    @staticmethod
    def _retrieve(initial: dict[str, dict[str, Any]], memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        retrieved = []
        for memory in memories:
            label = str(memory["label"])
            target_error = "FP" if initial[label]["status"] == "positive" else "FN"
            if f":{target_error}:" not in str(memory["group_id"]):
                continue
            retrieved.append({
                "memory_id": memory["group_id"],
                "label": label,
                "initial_status_trigger": initial[label]["status"],
                "support_count": int(memory["support_count"]),
                "recheck_instruction": memory["memory_summary_text"],
            })
        return retrieved

    def _ctclip(
        self, ct_path: Path
    ) -> tuple[dict[str, float], CtAttributionArtifact | None]:
        project_root = Path(__file__).resolve().parents[1]
        checkpoint = project_root / "models/ctclip/CT-CLIP_v2.pt"
        volume_stat = ct_path.stat()
        checkpoint_stat = checkpoint.stat()
        fingerprint = hashlib.sha256(json.dumps({
            "version": 1,
            "variant": "zeroshot",
            "volume": str(ct_path.resolve()),
            "volume_size": volume_stat.st_size,
            "volume_mtime_ns": volume_stat.st_mtime_ns,
            "checkpoint_size": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
            "fp16": True,
        }, sort_keys=True).encode("utf-8")).hexdigest()
        cache_path = Path(self.settings.artifact_dir) / "validated_memory_ctclip_cache" / f"{fingerprint}.json"
        attribution_path = ct_path.parent / ".ctclip_attribution.npz"
        if cache_path.is_file() and attribution_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return (
                {key: float(value) for key, value in payload.items()},
                CtAttributionArtifact(artifact_path=str(attribution_path.resolve())),
            )
        if self.settings.ctclip_service_enabled:
            with httpx.Client(
                timeout=self.settings.ctclip_timeout_seconds,
                trust_env=False,
            ) as client:
                response = client.post(
                    self.settings.ctclip_service_url.rstrip("/") + "/predict",
                    json={
                        "volume": str(ct_path),
                        "include_attribution": True,
                        "include_cflt": False,
                    },
                    headers={
                        "Authorization": f"Bearer {self.settings.ctclip_service_api_key}"
                    },
                )
            response.raise_for_status()
            response_payload = response.json()
            probabilities = {
                key: float(value)
                for key, value in response_payload["probabilities"].items()
            }
            artifact = (
                CtAttributionArtifact.model_validate(response_payload["attribution"])
                if response_payload.get("attribution")
                else None
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(probabilities, sort_keys=True), encoding="utf-8")
            return probabilities, artifact
        command = [
            str(self.settings.ctclip_python), str(project_root / "scripts/ctclip_worker.py"),
            "--volume", str(ct_path), "--checkpoint", str(checkpoint),
            "--source-dir", str(self.settings.ctclip_source_dir), "--device", self.settings.ctclip_device,
            "--variant", "zeroshot", "--fp16",
            "--attribution-output", str(attribution_path),
        ]
        completed = subprocess.run(
            command, cwd=project_root, capture_output=True, text=True,
            timeout=self.settings.ctclip_timeout_seconds, check=False,
        )
        if completed.returncode != 0:
            lines = [line for line in completed.stderr.splitlines() if line.strip()]
            raise RuntimeError(lines[-1] if lines else "zero-shot CT-CLIP failed")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("zero-shot CT-CLIP returned no output")
        worker_payload = json.loads(lines[-1])
        probabilities_payload = worker_payload.get("probabilities", worker_payload)
        probabilities = {
            key: float(value) for key, value in probabilities_payload.items()
        }
        artifact = (
            CtAttributionArtifact.model_validate(worker_payload["attribution"])
            if worker_payload.get("attribution")
            else None
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(probabilities, sort_keys=True), encoding="utf-8")
        return probabilities, artifact

    async def run(
        self,
        case_id: str,
        ct_path: Path,
        publish: Callable[[ValidatedStage], Awaitable[None]] | None = None,
    ) -> ValidatedMemoryResponse:
        started = time.perf_counter()
        stages: list[ValidatedStage] = []

        async def stage(name: str, status: str, summary: str, stage_started: float) -> None:
            item = ValidatedStage(
                name=name, status=status, summary=summary,
                latency_ms=round((time.perf_counter() - stage_started) * 1000, 2),
            )
            stages.append(item)
            if publish:
                await publish(item)

        current = time.perf_counter()
        sheets = render_blind_sheets(
            ct_path, Path(self.settings.static_dir) / "cases" / case_id / "validated_sheets"
        )
        await stage("读取CT并生成24层双窗切片", "success", "6张盲测图，每张含4个轴位层面的肺窗与纵隔窗", current)

        schema = {"assessments": [{"label": label, "status": "positive|negative", "confidence": "0..1", "visible_evidence": "visible evidence"} for label in LABELS]}
        current = time.perf_counter()
        try:
            first, initial = await self._vision_review(
            FIRST_PASS_SYSTEM,
            json.dumps({"task": "Screen all nine findings", "findings": LABELS, "output_schema": schema}),
            sheets,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Qwen blind review unavailable or incomplete: {exc}") from exc
        initial_positive = [
            LABEL_ZH[label]
            for label, item in initial.items()
            if item["status"] == "positive"
        ]
        await stage(
            "Qwen全胸部盲筛",
            "success",
            f"模型={self.vision_model}；输入=24层肺窗+纵隔窗；初筛阳性="
            + ("、".join(initial_positive) if initial_positive else "无"),
            current,
        )

        current = time.perf_counter()
        scores, attribution_artifact = self._ctclip(ct_path)
        await stage(
            "调用CT-CLIP zero-shot工具",
            "success",
            "模型=CT-CLIP_v2.pt zero-shot；输出9类检查级概率；最高3项="
            + "、".join(
                f"{LABEL_ZH.get(name, name)} {score:.2f}"
                for name, score in sorted(
                    ((name, scores[CTCLIP_LABEL.get(name, name)]) for name in LABELS),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
            )
            + "；同时返回Gradient × Token归因体",
            current,
        )

        findings = [{
            "label": label,
            "qwen_initial_status": initial[label]["status"],
            "qwen_initial_confidence": initial[label]["confidence"],
            "qwen_visible_evidence": initial[label]["visible_evidence"],
            "ctclip_score": round(scores[CTCLIP_LABEL.get(label, label)], 6),
        } for label in LABELS]
        current = time.perf_counter()
        warnings: list[str] = []
        ranked_labels = sorted(
            LABELS,
            key=lambda label: scores[CTCLIP_LABEL.get(label, label)],
            reverse=True,
        )
        candidate_labels = list(dict.fromkeys(
            [label for label in LABELS if initial[label]["status"] == "positive"]
            + ranked_labels[:3]
        ))
        finding_by_label = {item["label"]: item for item in findings}
        semaphore = asyncio.Semaphore(3)

        async def review_candidate(label: str) -> dict[str, Any]:
            async with semaphore:
                return await self._single_label_review(
                    TOOL_SYSTEM,
                    finding_by_label[label],
                    sheets,
                )

        candidate_outputs = await asyncio.gather(
            *(review_candidate(label) for label in candidate_labels),
            return_exceptions=True,
        )
        adjudicated = {
            label: {
                **item,
                "memory_ids_used": [],
                "supporting_slice_indices": [],
            }
            for label, item in initial.items()
        }
        adjudication_failures: list[str] = []
        for label, output in zip(candidate_labels, candidate_outputs, strict=True):
            if isinstance(output, BaseException):
                adjudication_failures.append(str(output))
                continue
            adjudicated[label] = output
        adjudication_status = "degraded" if adjudication_failures else "success"
        if adjudication_failures:
            warnings.append(
                "部分候选疾病的Qwen逐项工具复核失败；失败项保留上一阶段盲筛结论："
                + "；".join(adjudication_failures)
            )
        changed_after_tool = [
            f"{LABEL_ZH[label]}:{initial[label]['status']}→{adjudicated[label]['status']}"
            for label in LABELS
            if initial[label]["status"] != adjudicated[label]["status"]
        ]
        await stage(
            "Agent融合视觉与工具",
            adjudication_status,
            (
                f"裁决模型={self.vision_model}；按候选疾病逐项复核="
                + "、".join(LABEL_ZH[label] for label in candidate_labels)
                + "；状态修改="
                + ("、".join(changed_after_tool) if changed_after_tool else "无")
                + (
                    "；失败项保留盲筛结果=" + "；".join(adjudication_failures)
                    if adjudication_failures
                    else ""
                )
            ),
            current,
        )

        memories = self._retrieve(adjudicated, self._memories())
        memory_schema = {"assessments": [{
            "label": label, "proposed_status": "positive|negative", "confidence": "0..1",
            "memory_ids_used": ["memory_id"], "supporting_slice_indices": [1, 2],
            "visible_evidence": "current image evidence",
        } for label in LABELS]}
        current = time.perf_counter()
        memory_status = "success"
        try:
            memory_call, proposed = await self._vision_review(
            MEMORY_SYSTEM,
            json.dumps({
                "task": "Reinspect all labels after reading relevant memories",
                "initial_assessments": list(adjudicated.values()),
                "retrieved_memories": memories,
                "change_policy": "Change only with discriminating current evidence on at least two slices",
                "output_schema": memory_schema,
            }, ensure_ascii=False),
            sheets,
            proposed=True,
            )
        except RuntimeError as exc:
            memory_status = "degraded"
            warnings.append(
                "Memory复核连续三次未返回完整9类JSON；最终结果保留Agent与CT-CLIP融合结论。"
            )
            proposed = {
                label: {
                    **item,
                    "memory_ids_used": [],
                    "supporting_slice_indices": [],
                }
                for label, item in adjudicated.items()
            }

        available = {item["memory_id"]: item for item in memories}
        results: list[ValidatedLabelResult] = []
        accepted_count = 0
        for label in LABELS:
            original = adjudicated[label]
            item = proposed[label]
            final_status = original["status"]
            reasons: list[str] = []
            accepted = False
            if item["status"] == original["status"]:
                reasons.append("unchanged")
            else:
                cited = [mid for mid in item["memory_ids_used"] if mid in available and available[mid]["label"] == label]
                if not cited:
                    reasons.append("no_matching_audited_memory")
                if len(item["supporting_slice_indices"]) < 2:
                    reasons.append("fewer_than_two_current_slices")
                if item["confidence"] < 0.75:
                    reasons.append("confidence_below_0.75")
                if not item["visible_evidence"].strip():
                    reasons.append("missing_current_image_evidence")
                # This threshold was selected on the disjoint 100-case feedback split.
                if label == "consolidation" and item["status"] == "negative" and scores[label] >= 0.475:
                    reasons.append("ctclip_consolidation_gate")
                accepted = not reasons
                if accepted:
                    final_status = item["status"]
                    accepted_count += 1
            results.append(ValidatedLabelResult(
                name=label, name_zh=LABEL_ZH[label], status=final_status,
                confidence=item["confidence"] if accepted else original["confidence"],
                evidence=item["visible_evidence"] if accepted else original["visible_evidence"],
                initial_status=initial[label]["status"],
                ctclip_score=round(scores[CTCLIP_LABEL.get(label, label)], 4),
                memory_proposed_status=item["status"], memory_change_accepted=accepted,
                gate_reasons=reasons,
            ))
        memory_summary = (
            f"从audited_candidates.json检索{len(memories)}条Memory；"
            + (
                "命中="
                + "；".join(
                    f"{LABEL_ZH.get(item['label'], item['label'])}({item['memory_id']}，"
                    f"支持病例{item['support_count']}例)：{item['recheck_instruction']}"
                    for item in memories
                )
                if memories
                else "本病例无匹配经验"
            )
            + f"；门控接受{accepted_count}项修改"
            if memory_status == "success"
            else "Memory复核失败；保留Agent与CT-CLIP融合结果"
        )
        await stage("审核Memory复查与门控", memory_status, memory_summary, current)

        current = time.perf_counter()
        attribution_predictions = [
            LabelPrediction(
                name=CTCLIP_LABEL.get(item.name, item.name),
                status=item.status,
                confidence=item.ctclip_score,
                source="ct",
            )
            for item in results
        ]
        model_attributions, attribution_warnings, cache_hit, _ = (
            self.attribution_tool.render(
                case_id,
                str(ct_path),
                attribution_predictions,
                attribution_artifact,
            )
        )
        warnings.extend(attribution_warnings)
        await stage(
            "生成CT-CLIP模型归因图",
            "success" if model_attributions else "degraded",
            f"生成{sum(len(item.overlay_images) for item in model_attributions.values())}张归因图；"
            f"缓存={'命中' if cache_hit else '未命中'}",
            current,
        )

        benchmark = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        return ValidatedMemoryResponse(
            case_id=case_id, labels=results, sheet_paths=[str(path.resolve()) for path in sheets],
            stages=stages, retrieved_memories=memories, accepted_memory_changes=accepted_count,
            total_latency_ms=round((time.perf_counter() - started) * 1000, 2),
            model_attributions=model_attributions,
            warnings=warnings,
            benchmark={
                "dataset": "PatchChestCT frozen split", "cases": benchmark["cases"],
                "micro_f1": benchmark["with_memory"]["micro_f1"],
                "macro_f1": benchmark["with_memory"]["macro_f1"],
                "baseline_micro_f1": benchmark["baseline"]["micro_f1"],
                "baseline_macro_f1": benchmark["baseline"]["macro_f1"],
                "accepted_changes": benchmark["accepted_changes"],
                "beneficial_changes": benchmark["beneficial_changes"],
                "harmful_changes": benchmark["harmful_changes"],
                "scope": "9 labels; blinded CT only; no report or reference labels in prompts",
            },
        )
