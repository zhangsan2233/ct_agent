from __future__ import annotations

import json
from pathlib import Path
import re
import time

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from chestct_agent.config import Settings
from chestct_agent.schemas import (
    CtAttributionArtifact,
    LabelPrediction,
    ModelAttributionEvidence,
)
from chestct_agent.tools.ct_preprocess import _window_ct


RENDER_VERSION = 7
MAX_RENDER_SIZE = 768


class CtAttributionTool:
    """Maps CT-CLIP token attribution back to native axial CT slices."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"

    @staticmethod
    def _top_regions(
        attribution: np.ndarray,
        count: int,
        eligible_indices: np.ndarray | None = None,
    ) -> list[int]:
        positive = np.maximum(attribution, 0.0)
        flat = positive.reshape(positive.shape[0], -1)
        cutoff = max(1, flat.shape[1] // 10)
        scores = np.partition(flat, -cutoff, axis=1)[:, -cutoff:].mean(axis=1)
        ranked = np.argsort(scores, kind="stable")[::-1]
        eligible = (
            {int(index) for index in eligible_indices}
            if eligible_indices is not None
            else set(range(positive.shape[0]))
        )
        selected: list[int] = []
        for index in ranked:
            index = int(index)
            if index not in eligible:
                continue
            if any(abs(index - existing) <= 1 for existing in selected):
                continue
            selected.append(index)
            if len(selected) == count:
                break
        return selected

    @staticmethod
    def _native_slice_index(token_index: int, grid_depth: int, metadata: dict) -> int:
        target_depth = int(metadata["target_shape"][0])
        pad_before = int(metadata["pad_before"][0])
        crop_start = int(metadata["crop_start"][0])
        resampled_depth = int(metadata["resampled_shape"][0])
        original_depth = int(metadata["original_shape"][2])

        target_index = (token_index + 0.5) * target_depth / grid_depth - 0.5
        resampled_index = target_index - pad_before + crop_start
        resampled_index = float(np.clip(resampled_index, 0, resampled_depth - 1))
        native_index = (resampled_index + 0.5) * original_depth / resampled_depth - 0.5
        return int(np.clip(round(native_index), 0, original_depth - 1))

    @staticmethod
    def _native_heatmap(token_map: np.ndarray, metadata: dict) -> np.ndarray:
        target_shape = [int(value) for value in metadata["target_shape"]]
        crop_start = [int(value) for value in metadata["crop_start"]]
        crop_shape = [int(value) for value in metadata["crop_shape"]]
        pad_before = [int(value) for value in metadata["pad_before"]]
        resampled_shape = [int(value) for value in metadata["resampled_shape"]]
        original_shape = [int(value) for value in metadata["original_shape"]]

        target = np.asarray(
            Image.fromarray(token_map.astype(np.float32), mode="F").resize(
                (target_shape[2], target_shape[1]),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
        x0, y0 = pad_before[1], pad_before[2]
        cropped = target[x0 : x0 + crop_shape[1], y0 : y0 + crop_shape[2]]
        resampled = np.zeros((resampled_shape[1], resampled_shape[2]), dtype=np.float32)
        rx0, ry0 = crop_start[1], crop_start[2]
        resampled[rx0 : rx0 + crop_shape[1], ry0 : ry0 + crop_shape[2]] = cropped
        return np.asarray(
            Image.fromarray(resampled, mode="F").resize(
                (original_shape[1], original_shape[0]),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _valid_grid_weights(metadata: dict, grid_shape: tuple[int, ...]) -> np.ndarray:
        target_shape = np.asarray(metadata["target_shape"], dtype=np.float32)
        valid_start = np.asarray(metadata["pad_before"], dtype=np.float32)
        valid_end = valid_start + np.asarray(metadata["crop_shape"], dtype=np.float32)
        axis_weights: list[np.ndarray] = []
        for target, start, end, count in zip(
            target_shape, valid_start, valid_end, grid_shape, strict=True
        ):
            edges = np.linspace(0.0, target, num=count + 1, dtype=np.float32)
            overlap = np.maximum(
                0.0,
                np.minimum(edges[1:], end) - np.maximum(edges[:-1], start),
            )
            axis_weights.append(overlap / np.maximum(edges[1:] - edges[:-1], 1e-8))
        return (
            axis_weights[0][:, None, None]
            * axis_weights[1][None, :, None]
            * axis_weights[2][None, None, :]
        )

    def _render_overlay(
        self,
        ct_slice: np.ndarray,
        heatmap: np.ndarray,
        original_path: Path,
        output_path: Path,
        label: str,
        status: str,
        score: float,
    ) -> None:
        base = _window_ct(np.rot90(ct_slice))
        heat = np.clip(np.rot90(heatmap), 0.0, 1.0)
        if max(base.shape) > MAX_RENDER_SIZE:
            scale = MAX_RENDER_SIZE / max(base.shape)
            output_size = (
                max(1, round(base.shape[1] * scale)),
                max(1, round(base.shape[0] * scale)),
            )
            base = np.asarray(
                Image.fromarray(base).resize(output_size, resample=Image.Resampling.BILINEAR)
            )
            heat = np.asarray(
                Image.fromarray(heat.astype(np.float32), mode="F").resize(
                    output_size, resample=Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
        blur_radius = max(1.0, max(heat.shape) / 420)
        heat = np.asarray(
            Image.fromarray(np.clip(heat * 255.0, 0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            ),
            dtype=np.float32,
        ) / 255.0
        original_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(base).save(original_path, compress_level=3)
        rgb = np.repeat(base[:, :, None], 3, axis=2).astype(np.float32)
        strength = np.clip((heat - 0.20) / 0.55, 0.0, 1.0)
        strength = strength * strength * (3.0 - 2.0 * strength)
        color = np.empty_like(rgb)
        color[:, :, 0] = np.interp(strength, [0.0, 0.35, 0.65, 0.85, 1.0], [80, 210, 255, 255, 255])
        color[:, :, 1] = np.interp(strength, [0.0, 0.35, 0.65, 0.85, 1.0], [0, 20, 85, 200, 255])
        color[:, :, 2] = np.interp(strength, [0.0, 0.35, 0.65, 0.85, 1.0], [18, 25, 5, 0, 210])
        alpha = self.settings.ct_attribution_alpha * strength
        darkened = rgb * (1.0 - 0.18 * strength[:, :, None])
        blended = darkened * (1.0 - alpha[:, :, None]) + color * alpha[:, :, None]

        peak_y, peak_x = np.unravel_index(int(np.argmax(heat)), heat.shape)
        max_heat = float(heat.max())
        peak_component = np.zeros_like(heat, dtype=bool)
        if max_heat >= 0.25:
            component_image = Image.fromarray((heat >= max_heat * 0.78).astype(np.uint8) * 255)
            ImageDraw.floodfill(component_image, (int(peak_x), int(peak_y)), 128)
            peak_component = np.asarray(component_image) == 128
        strong_mask = Image.fromarray(peak_component.astype(np.uint8) * 255)
        expanded = np.asarray(strong_mask.filter(ImageFilter.MaxFilter(7)))
        contracted = np.asarray(strong_mask.filter(ImageFilter.MinFilter(5)))
        boundary = (expanded > contracted) & (expanded > 0)
        blended[boundary] = np.array([255.0, 246.0, 170.0], dtype=np.float32)
        overlay = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

        header_height = 42 if overlay.width >= 300 else 0
        legend_height = 30 if overlay.width >= 180 else 22
        canvas = Image.new(
            "RGB", (overlay.width, header_height + overlay.height + legend_height), "black"
        )
        canvas.paste(overlay, (0, header_height))
        draw = ImageDraw.Draw(canvas)
        font_path = Path("C:/Windows/Fonts/arial.ttf")
        try:
            title_font = ImageFont.truetype(str(font_path), 17)
            small_font = ImageFont.truetype(str(font_path), 12)
        except OSError:
            title_font = ImageFont.load_default()
            small_font = title_font
        if header_height:
            display_label = label.replace("_", " ").upper()
            draw.text(
                (12, 11),
                f"{display_label}  |  CT SCORE {score:.2f}",
                fill=(238, 242, 247),
                font=title_font,
            )
            status_colors = {
                "positive": (255, 91, 78),
                "uncertain": (255, 190, 64),
                "negative": (80, 205, 180),
            }
            status_text = status.upper()
            status_width = draw.textbbox((0, 0), status_text, font=small_font)[2]
            badge_left = overlay.width - status_width - 28
            draw.rounded_rectangle(
                (badge_left, 9, overlay.width - 10, 33),
                radius=7,
                fill=status_colors.get(status, (150, 160, 170)),
            )
            draw.text((badge_left + 9, 14), status_text, fill=(8, 12, 16), font=small_font)

        if max_heat >= 0.25:
            radius = max(7, min(13, overlay.width // 55))
            peak_y += header_height
            draw.ellipse(
                (peak_x - radius, peak_y - radius, peak_x + radius, peak_y + radius),
                outline=(255, 255, 255),
                width=2,
            )
            draw.line(
                (peak_x - radius - 5, peak_y, peak_x + radius + 5, peak_y),
                fill=(255, 255, 255),
                width=1,
            )
            draw.line(
                (peak_x, peak_y - radius - 5, peak_x, peak_y + radius + 5),
                fill=(255, 255, 255),
                width=1,
            )

        legend_top = header_height + overlay.height
        bar_width = min(220, max(90, overlay.width // 3))
        bar_x = overlay.width - bar_width - 44
        for offset in range(bar_width):
            value = offset / max(bar_width - 1, 1)
            red = int(np.interp(value, [0.0, 0.35, 0.65, 0.85, 1.0], [80, 210, 255, 255, 255]))
            green = int(np.interp(value, [0.0, 0.35, 0.65, 0.85, 1.0], [0, 20, 85, 200, 255]))
            blue = int(np.interp(value, [0.0, 0.35, 0.65, 0.85, 1.0], [18, 25, 5, 0, 210]))
            draw.line(
                (bar_x + offset, legend_top + 8, bar_x + offset, legend_top + 18),
                fill=(red, green, blue),
            )
        draw.text(
            (10, legend_top + 7),
            "MODEL ATTRIBUTION",
            fill=(220, 225, 230),
            font=small_font,
        )
        draw.text((bar_x - 30, legend_top + 7), "LOW", fill=(150, 160, 170), font=small_font)
        draw.text(
            (bar_x + bar_width + 6, legend_top + 7),
            "HIGH",
            fill=(235, 238, 242),
            font=small_font,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, compress_level=3)

    def _manifest_key(
        self,
        volume_path: Path,
        artifact_path: Path,
        predictions: list[LabelPrediction],
    ) -> dict:
        volume_stat = volume_path.stat()
        artifact_stat = artifact_path.stat()
        return {
            "version": RENDER_VERSION,
            "volume_size": volume_stat.st_size,
            "volume_mtime_ns": volume_stat.st_mtime_ns,
            "artifact_size": artifact_stat.st_size,
            "artifact_mtime_ns": artifact_stat.st_mtime_ns,
            "targets": [
                {
                    "label": prediction.name,
                    "status": prediction.status,
                    "score": round(prediction.confidence, 8),
                }
                for prediction in sorted(predictions, key=lambda item: item.name)
            ],
            "slices_per_label": self.settings.ct_attribution_slices_per_label,
            "alpha": self.settings.ct_attribution_alpha,
        }

    @staticmethod
    def _load_manifest(path: Path, expected: dict) -> dict[str, ModelAttributionEvidence] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != expected:
                return None
            evidence = {
                label: ModelAttributionEvidence.model_validate(item)
                for label, item in payload["evidence"].items()
            }
            if not all(
                Path(image).exists()
                for item in evidence.values()
                for image in [*item.original_images, *item.overlay_images]
            ):
                return None
            return evidence
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def render(
        self,
        case_id: str,
        ct_volume_path: str | None,
        predictions: list[LabelPrediction],
        artifact: CtAttributionArtifact | None,
    ) -> tuple[dict[str, ModelAttributionEvidence], list[str], bool | None, float]:
        started = time.perf_counter()
        target_predictions = sorted(predictions, key=lambda item: item.name)
        if not self.settings.ct_attribution_enabled or not target_predictions:
            return {}, [], None, round((time.perf_counter() - started) * 1000, 2)
        if not ct_volume_path or artifact is None:
            return (
                {},
                ["CT标签存在，但模型归因体不可用；已降级为普通CT预览。"],
                None,
                round((time.perf_counter() - started) * 1000, 2),
            )

        volume_path = Path(ct_volume_path)
        artifact_path = Path(artifact.artifact_path)
        if not volume_path.exists() or not artifact_path.exists():
            return (
                {},
                ["CT模型归因所需的原始体积或归因缓存不存在；已降级为普通CT预览。"],
                None,
                round((time.perf_counter() - started) * 1000, 2),
            )

        safe_case_id = self._safe_name(case_id)
        output_root = Path(self.settings.static_dir) / "cases" / safe_case_id / "attribution"
        manifest_path = output_root / "manifest.json"
        fingerprint = self._manifest_key(volume_path, artifact_path, target_predictions)
        cached = self._load_manifest(manifest_path, fingerprint)
        if cached is not None:
            cached = {
                label: item.model_copy(update={"cache_hit": True}) for label, item in cached.items()
            }
            return (
                cached,
                [],
                True,
                round((time.perf_counter() - started) * 1000, 2),
            )

        try:
            with np.load(artifact_path, allow_pickle=False) as payload:
                attributions = np.asarray(payload["attributions"], dtype=np.float32)
                labels = [str(value) for value in payload["labels"].tolist()]
                preprocess = json.loads(str(payload["preprocess_json"].item()))
            if attributions.ndim != 4 or not np.isfinite(attributions).all():
                raise ValueError("Attribution tensor is invalid.")
            label_indices = {label: index for index, label in enumerate(labels)}
            predictions_by_label = {
                prediction.name: prediction for prediction in target_predictions
            }
            image = nib.load(str(volume_path))
            volume = image.get_fdata(dtype=np.float32, caching="unchanged")
            evidence: dict[str, ModelAttributionEvidence] = {}
            for prediction in target_predictions:
                label = prediction.name
                if label not in label_indices:
                    continue
                label_map = attributions[label_indices[label]]
                valid_grid_weights = self._valid_grid_weights(preprocess, label_map.shape)
                label_map = label_map * valid_grid_weights
                eligible_indices = np.flatnonzero(valid_grid_weights.sum(axis=(1, 2)) > 0)
                token_indices = self._top_regions(
                    label_map,
                    max(self.settings.ct_attribution_slices_per_label * 3, 3),
                    eligible_indices,
                )
                native_indices: list[int] = []
                original_images: list[str] = []
                overlay_images: list[str] = []
                for token_index in token_indices:
                    native_index = self._native_slice_index(
                        token_index, label_map.shape[0], preprocess
                    )
                    if native_index in native_indices:
                        continue
                    native_indices.append(native_index)
                    heatmap = self._native_heatmap(label_map[token_index], preprocess)
                    ct_slice = np.asarray(volume[:, :, native_index], dtype=np.float32)
                    output_path = (
                        output_root
                        / self._safe_name(label)
                        / f"slice_{native_index:03d}_attribution.png"
                    )
                    original_path = output_path.with_name(f"slice_{native_index:03d}_lung.png")
                    self._render_overlay(
                        ct_slice,
                        heatmap,
                        original_path,
                        output_path,
                        label,
                        prediction.status,
                        prediction.confidence,
                    )
                    original_images.append(str(original_path.as_posix()))
                    overlay_images.append(str(output_path.as_posix()))
                    if len(native_indices) >= self.settings.ct_attribution_slices_per_label:
                        break
                if overlay_images:
                    evidence[label] = ModelAttributionEvidence(
                        target_label=label,
                        target_status=prediction.status,
                        target_score=prediction.confidence,
                        grid_shape=list(label_map.shape),
                        slice_indices=native_indices,
                        original_images=original_images,
                        overlay_images=overlay_images,
                        cache_hit=False,
                    )
            output_root.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "evidence": {label: item.model_dump() for label, item in evidence.items()},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            warnings = []
            missing = sorted(set(predictions_by_label) - set(evidence))
            if missing:
                warnings.append("以下CT标签未生成有效归因切片：" + ", ".join(missing))
            return (
                evidence,
                warnings,
                False,
                round((time.perf_counter() - started) * 1000, 2),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return (
                {},
                [f"CT模型归因映射失败，已降级为普通CT预览。原因：{exc}"],
                None,
                round((time.perf_counter() - started) * 1000, 2),
            )
