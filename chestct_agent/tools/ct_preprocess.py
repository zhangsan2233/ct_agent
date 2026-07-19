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
        try:
            import nibabel as nib

            image = nib.load(str(path))
            volume = image.dataobj
        except Exception:
            return []

        safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._") or "case"
        out_dir = Path(self.settings.static_dir) / "cases" / safe_case_id
        out_dir.mkdir(parents=True, exist_ok=True)
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
        return rendered
