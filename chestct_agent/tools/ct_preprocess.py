import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from chestct_agent.config import Settings
from chestct_agent.schemas import QwenVisualLabelReview


def _window_ct(array: np.ndarray, center: float = -600.0, width: float = 1500.0) -> np.ndarray:
    low = center - width / 2
    high = center + width / 2
    clipped = np.clip(array, low, high)
    normalized = (clipped - low) / (high - low)
    return (normalized * 255).astype(np.uint8)


class CtPreprocessTool:
    """Renders representative axial slices without loading the full CT into RAM."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def render_preview_slices(self, case_id: str, ct_volume_path: str | None) -> list[str]:
        if not ct_volume_path:
            return []
        path = Path(ct_volume_path)
        if not path.exists():
            return []

        safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._") or "case"
        out_dir = Path(self.settings.static_dir) / "cases" / safe_case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        cached = self._load_cached_previews(path, out_dir)
        if cached:
            return cached
        existing = sorted(out_dir.glob("slice_*_lung.png"))
        volume_mtime_ns = path.stat().st_mtime_ns
        if existing and all(image.stat().st_mtime_ns >= volume_mtime_ns for image in existing):
            rendered = [str(image.as_posix()) for image in existing]
            self._write_preview_cache(path, out_dir, rendered)
            return rendered

        try:
            import nibabel as nib

            image = nib.load(str(path))
            volume = image.dataobj
        except Exception:
            return []

        axis = int(np.argmin(volume.shape))
        # Avoid nearly empty slices at the superior and inferior volume boundaries.
        indices = np.linspace(
            int(volume.shape[axis] * 0.2),
            int(volume.shape[axis] * 0.8),
            num=min(5, volume.shape[axis]),
            dtype=int,
        )
        rendered: list[str] = []
        for idx in indices:
            if axis == 0:
                slice_arr = np.asarray(volume[idx, :, :], dtype=np.float32)
            elif axis == 1:
                slice_arr = np.asarray(volume[:, idx, :], dtype=np.float32)
            else:
                slice_arr = np.asarray(volume[:, :, idx], dtype=np.float32)
            image = Image.fromarray(_window_ct(np.rot90(slice_arr)))
            out_path = out_dir / f"slice_{idx:03d}_lung.png"
            image.save(out_path)
            rendered.append(str(out_path.as_posix()))
        self._write_preview_cache(path, out_dir, rendered)
        return rendered

    def render_qwen_visual_slices(
        self, case_id: str, ct_volume_path: str | None
    ) -> list[str]:
        """Render paired lung/mediastinal axial views for multimodal review."""
        if not ct_volume_path:
            return []
        path = Path(ct_volume_path)
        if not path.exists():
            return []

        safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._") or "case"
        out_dir = Path(self.settings.static_dir) / "cases" / safe_case_id / "qwen_visual"
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = out_dir / "cache.json"
        try:
            if metadata_path.exists():
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                stat = path.stat()
                cached = [str(item) for item in payload.get("images", [])]
                if (
                    payload.get("version") == 2
                    and payload.get("volume_size") == stat.st_size
                    and payload.get("volume_mtime_ns") == stat.st_mtime_ns
                    and payload.get("image_count") == self.settings.qwen_vision_max_images
                    and cached
                    and all(Path(item).exists() for item in cached)
                ):
                    return cached
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

        try:
            import nibabel as nib

            nii = nib.load(str(path))
            volume = nii.dataobj
        except Exception:
            return []

        axis = int(np.argmin(volume.shape))
        count = min(self.settings.qwen_vision_max_images, volume.shape[axis])
        indices = np.linspace(
            int(volume.shape[axis] * 0.12),
            int(volume.shape[axis] * 0.88),
            num=count,
            dtype=int,
        )
        rendered: list[str] = []
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
            rendered.append(str(out_path.as_posix()))

        try:
            stat = path.stat()
            metadata_path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "volume_size": stat.st_size,
                        "volume_mtime_ns": stat.st_mtime_ns,
                        "image_count": self.settings.qwen_vision_max_images,
                        "images": rendered,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
        return rendered

    def render_qwen_grounding_heatmaps(
        self,
        case_id: str,
        source_images: list[str],
        reviews: list[QwenVisualLabelReview],
    ) -> dict[str, list[str]]:
        """Render Qwen-reported normalized boxes as transparent evidence heatmaps."""
        if not self.settings.qwen_grounding_enabled:
            return {}
        safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._") or "case"
        source_by_slice: dict[int, Path] = {}
        for image_path in source_images:
            match = re.search(r"slice_(\d+)", Path(image_path).name)
            if match:
                source_by_slice[int(match.group(1))] = Path(image_path)

        rendered_by_label: dict[str, list[str]] = {}
        for review in reviews:
            regions_by_slice: dict[int, list] = {}
            for region in review.regions:
                if region.slice_index in source_by_slice:
                    regions_by_slice.setdefault(region.slice_index, []).append(region)
            if not regions_by_slice:
                continue

            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", review.name).strip("._")
            out_dir = (
                Path(self.settings.static_dir)
                / "cases"
                / safe_case_id
                / "qwen_grounding"
                / safe_label
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            outputs: list[str] = []
            for slice_index, regions in sorted(regions_by_slice.items()):
                try:
                    source = Image.open(source_by_slice[slice_index]).convert("RGB")
                except OSError:
                    continue
                base = np.asarray(source, dtype=np.float32)
                height, width = base.shape[:2]
                pane_width = width // 2
                content_top = min(28, max(0, height - 1))
                content_height = max(1, height - content_top)
                heat = np.zeros((height, width), dtype=np.float32)

                for region in regions:
                    x1, y1, x2, y2 = region.bbox_2d
                    pane_left = 0 if region.window == "lung" else pane_width
                    px1 = pane_left + int((x1 / 1000.0) * pane_width)
                    px2 = pane_left + int((x2 / 1000.0) * pane_width)
                    py1 = content_top + int((y1 / 1000.0) * content_height)
                    py2 = content_top + int((y2 / 1000.0) * content_height)
                    px1 = max(pane_left, min(pane_left + pane_width - 1, px1))
                    px2 = max(px1 + 1, min(pane_left + pane_width, px2))
                    py1 = max(content_top, min(height - 1, py1))
                    py2 = max(py1 + 1, min(height, py2))

                    pane_right = pane_left + pane_width
                    yy, xx = np.mgrid[content_top:height, pane_left:pane_right]
                    center_x = (px1 + px2 - 1) / 2.0
                    center_y = (py1 + py2 - 1) / 2.0
                    sigma_x = max(4.0, (px2 - px1) / 2.8)
                    sigma_y = max(4.0, (py2 - py1) / 2.8)
                    bump = np.exp(
                        -0.5
                        * (
                            ((xx - center_x) / sigma_x) ** 2
                            + ((yy - center_y) / sigma_y) ** 2
                        )
                    )
                    bump *= max(0.25, region.confidence)
                    bump[bump < 0.035] = 0.0
                    heat[content_top:height, pane_left:pane_right] = np.maximum(
                        heat[content_top:height, pane_left:pane_right],
                        bump.astype(np.float32),
                    )

                if not np.any(heat > 0):
                    continue
                heat /= max(float(heat.max()), 1e-6)
                color = np.zeros_like(base)
                color[..., 0] = 255.0
                color[..., 1] = 255.0 * np.power(heat, 1.7)
                alpha = (
                    self.settings.qwen_grounding_alpha * np.power(heat, 0.72)
                )[..., None]
                blended = base * (1.0 - alpha) + color * alpha
                overlay = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
                draw = ImageDraw.Draw(overlay)
                draw.rectangle((0, 0, width - 1, 27), fill=(0, 0, 0))
                draw.text(
                    (8, 7),
                    f"QWEN VISUAL GROUNDING | {review.name} | SLICE {slice_index}",
                    fill="white",
                )
                out_path = out_dir / f"slice_{slice_index:04d}_qwen_grounding.png"
                overlay.save(out_path, optimize=True)
                outputs.append(str(out_path.as_posix()))
            if outputs:
                rendered_by_label[review.name] = outputs
        return rendered_by_label

    @staticmethod
    def _load_cached_previews(volume_path: Path, out_dir: Path) -> list[str]:
        metadata_path = out_dir / "preview_cache.json"
        if not metadata_path.exists():
            return []
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            stat = volume_path.stat()
            images = [str(item) for item in payload["images"]]
            if payload.get("version") != 1:
                return []
            if payload.get("volume_size") != stat.st_size:
                return []
            if payload.get("volume_mtime_ns") != stat.st_mtime_ns:
                return []
            if not all(Path(image).exists() for image in images):
                return []
            return images
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return []

    @staticmethod
    def _write_preview_cache(volume_path: Path, out_dir: Path, images: list[str]) -> None:
        try:
            stat = volume_path.stat()
            payload = {
                "version": 1,
                "volume_size": stat.st_size,
                "volume_mtime_ns": stat.st_mtime_ns,
                "images": images,
            }
            (out_dir / "preview_cache.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return
