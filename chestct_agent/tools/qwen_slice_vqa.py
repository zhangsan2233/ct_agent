import json
from pathlib import Path
import re
import time

from chestct_agent.config import Settings
from chestct_agent.knowledge import LABEL_ZH
from chestct_agent.labels import LABEL_IDS
from chestct_agent.llm import QwenClient
from chestct_agent.schemas import (
    DiagnosticToolEvidence,
    LabelPrediction,
    QwenVisualLabelReview,
    QwenVisualRegion,
)
from chestct_agent.tools.ct_preprocess import CtPreprocessTool


class QwenSliceVqaTool:
    """Independent slice VLM review; the legacy name is retained for API compatibility."""

    def __init__(self, settings: Settings, qwen: QwenClient, preprocess: CtPreprocessTool):
        self.settings = settings
        self.qwen = qwen
        self.preprocess = preprocess

    @staticmethod
    def _slice_index(path: str) -> int | None:
        match = re.search(r"slice_(\d+)", Path(path).name)
        return int(match.group(1)) if match else None

    async def review(
        self,
        case_id: str,
        ct_volume_path: str | None,
        ct_predictions: list[LabelPrediction],
        diagnostic_evidence: list[DiagnosticToolEvidence] | None = None,
    ) -> tuple[list[QwenVisualLabelReview], list[str], bool, str | None, float]:
        started = time.perf_counter()
        if not self.settings.qwen_vision_enabled or not self.settings.slice_vlm_enabled:
            return [], [], False, "disabled", 0.0
        representative = self.preprocess.render_qwen_visual_slices(case_id, ct_volume_path)
        targeted = [
            item.preview_images[0]
            for item in (diagnostic_evidence or [])
            if item.preview_images and Path(item.preview_images[0]).is_file()
        ]
        images = list(dict.fromkeys(targeted or representative))[
            : self.settings.slice_vlm_max_images
        ]
        if not images:
            return [], [], False, "preview_generation_failed", 0.0

        slice_indices = [index for path in images if (index := self._slice_index(path)) is not None]
        candidate_names = [
            item.name for item in ct_predictions if item.status in {"positive", "uncertain"}
        ]
        targeted_labels = list(
            dict.fromkeys(item.label for item in (diagnostic_evidence or []))
        )
        labels_to_review = targeted_labels or LABEL_IDS
        label_contract = [
            {"label": label, "name_zh": LABEL_ZH.get(label, label)}
            for label in labels_to_review
        ]
        fallback = {"assessments": [], "coverage_warning": "vision_call_unavailable"}
        call = await self.qwen.chat_json_with_images(
            system=(
                "You are an independent chest CT slice reviewer. Inspect only the supplied axial "
                "images. Some images pair lung and mediastinal windows; candidate images may carry "
                "a colored segmentation overlay. Judge the underlying CT appearance, not merely "
                "the overlay. Do not use the initial classifier as visual evidence. Representative "
                "slices do not prove whole-volume absence, so use uncertain whenever coverage is "
                "insufficient. Return valid JSON only."
            ),
            user=json.dumps(
                {
                    "task": (
                        "Review every requested label using visible image evidence. Return one assessment "
                        "per label with label, status (positive|negative|uncertain), confidence "
                        "from 0 to 1, supporting slice_indices, concise evidence_zh, and regions. "
                        "A positive "
                        "requires a visible finding. Negative is allowed only when these images "
                        "adequately cover the relevant structure; otherwise use uncertain. For "
                        "each visibly localizable positive or uncertain finding, return up to three "
                        "regions. Each region must use one supplied slice_index, window must be "
                        "lung or mediastinal, and bbox_2d must be [x1,y1,x2,y2] normalized to "
                        "0..1000 within that window pane's CT content (excluding the title strip). "
                        "Do not invent a region for a negative or non-localizable finding."
                    ),
                    "image_order_slice_indices": slice_indices,
                    "classifier_candidates_for_priority_only": candidate_names,
                    "independent_tool_candidates": [
                        {
                            "label": item.label,
                            "verdict": item.verdict,
                            "metrics": item.metrics,
                        }
                        for item in (diagnostic_evidence or [])
                    ],
                    "labels": label_contract,
                    "output_schema": {
                        "assessments": [
                            {
                                "label": "pulmonary_nodule",
                                "status": "uncertain",
                                "confidence": 0.6,
                                "slice_indices": [],
                                "evidence_zh": "代表性切片不足以排除小结节",
                                "regions": [
                                    {
                                        "slice_index": 105,
                                        "window": "lung",
                                        "bbox_2d": [250, 300, 520, 620],
                                        "confidence": 0.7,
                                        "description_zh": "可疑局部区域",
                                    }
                                ],
                            }
                        ],
                        "coverage_warning": "string",
                    },
                },
                ensure_ascii=False,
            ),
            image_paths=images,
            fallback=fallback,
            max_tokens=4096,
            model=self.settings.slice_vlm_model,
        )
        reviews: list[QwenVisualLabelReview] = []
        seen: set[str] = set()
        raw_assessments = (
            call.value.get("assessments")
            or call.value.get("results")
            or call.value.get("labels")
            or []
        )
        if isinstance(raw_assessments, dict):
            raw_assessments = [
                {"label": label, **value}
                for label, value in raw_assessments.items()
                if isinstance(value, dict)
            ]
        zh_to_label = {name_zh: label for label, name_zh in LABEL_ZH.items()}
        if isinstance(raw_assessments, list):
            for raw in raw_assessments:
                if not isinstance(raw, dict):
                    continue
                name = str(raw.get("label") or raw.get("name") or "").strip()
                name = zh_to_label.get(name, name)
                if name not in LABEL_IDS or name in seen:
                    continue
                status = str(raw.get("status", "uncertain")).strip().lower()
                if status not in {"positive", "negative", "uncertain"}:
                    status = "uncertain"
                try:
                    confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.0))))
                except (TypeError, ValueError):
                    confidence = 0.0
                raw_slices = raw.get("slice_indices", [])
                valid_slices = []
                if isinstance(raw_slices, list):
                    valid_slices = [
                        int(value)
                        for value in raw_slices
                        if isinstance(value, (int, float)) and int(value) in slice_indices
                    ]
                regions: list[QwenVisualRegion] = []
                if status != "negative" and isinstance(raw.get("regions"), list):
                    for raw_region in raw["regions"][:3]:
                        if not isinstance(raw_region, dict):
                            continue
                        try:
                            region_slice = int(raw_region.get("slice_index", -1))
                            if region_slice not in slice_indices:
                                continue
                            window = str(raw_region.get("window", "")).strip().lower()
                            if window not in {"lung", "mediastinal"}:
                                continue
                            bbox = raw_region.get("bbox_2d", [])
                            if not isinstance(bbox, list) or len(bbox) != 4:
                                continue
                            regions.append(
                                QwenVisualRegion(
                                    slice_index=region_slice,
                                    window=window,
                                    bbox_2d=[int(value) for value in bbox],
                                    confidence=min(
                                        1.0,
                                        max(0.0, float(raw_region.get("confidence", confidence))),
                                    ),
                                    description_zh=str(
                                        raw_region.get("description_zh", "")
                                    ).strip()[:300],
                                )
                            )
                        except (TypeError, ValueError):
                            continue
                reviews.append(
                    QwenVisualLabelReview(
                        name=name,
                        status=status,
                        confidence=confidence,
                        slice_indices=valid_slices,
                        evidence_zh=str(raw.get("evidence_zh", "")).strip()[:500],
                        regions=regions,
                        backend="independent_slice_vlm",
                        model=self.settings.slice_vlm_model,
                    )
                )
                seen.add(name)
        rendered = (
            {}
            if targeted
            else self.preprocess.render_qwen_grounding_heatmaps(case_id, images, reviews)
        )
        for review in reviews:
            review.grounding_heatmap_images = rendered.get(review.name, [])
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        used_remote = call.used_remote and bool(reviews)
        fallback_reason = call.fallback_reason
        if call.used_remote and not reviews:
            fallback_reason = "empty_valid_assessments"
        return reviews, images, used_remote, fallback_reason, latency_ms
