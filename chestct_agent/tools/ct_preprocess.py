import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from chestct_agent.config import Settings


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
