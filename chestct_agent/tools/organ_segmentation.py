from pathlib import Path
import re

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from chestct_agent.config import Settings
from chestct_agent.schemas import AnatomyMaskResult


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "region"


def _bounds(mask: np.ndarray) -> tuple[list[int], list[int]] | None:
    occupied = mask > 0
    if not occupied.any():
        return None
    ranges = []
    for axis in range(3):
        reduce_axes = tuple(index for index in range(3) if index != axis)
        indices = np.flatnonzero(occupied.any(axis=reduce_axes))
        ranges.append((int(indices[0]), int(indices[-1])))
    return [value[0] for value in ranges], [value[1] for value in ranges]


def _map_index(value: int, source_size: int, target_size: int) -> int:
    if source_size <= 1 or target_size <= 1:
        return 0
    return int(round(value / (source_size - 1) * (target_size - 1)))


class OrganSegmentationTool:
    """Loads RadGenome masks and maps them to the local CT index grid."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.index_path = Path(settings.radgenome_index_path)
        self.mask_root = Path(settings.radgenome_mask_dir).resolve()
        self.index = self._load_index()

    def _load_index(self) -> pd.DataFrame:
        if not self.index_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.index_path).fillna("")
        except Exception:
            return pd.DataFrame()

    def available(self, case_id: str) -> bool:
        return not self.index.empty and bool(self.index["case_id"].astype(str).eq(case_id).any())

    def segment(
        self,
        case_id: str,
        ct_volume_path: str | None,
        requested_regions: set[str] | None = None,
    ) -> tuple[list[AnatomyMaskResult], list[str]]:
        if not ct_volume_path or not Path(ct_volume_path).exists():
            return [], ["器官分割缺少可读取的CT体数据。"]
        if self.index.empty:
            return [], ["RadGenome mask索引不存在，请先运行索引脚本。"]
        rows = self.index[self.index["case_id"].astype(str).eq(case_id)].copy()
        if rows.empty:
            return [], [f"RadGenome中没有病例 {case_id} 的mask。"]
        if requested_regions:
            exact_regions = {
                region.lower().replace("_", " ") for region in requested_regions
            }
            normalized_regions = {
                token
                for region in requested_regions
                for token in {region.lower(), *region.lower().replace("_", " ").split()}
                if len(token) >= 4 and token not in {"left", "right", "upper", "lower"}
            }
            rows = rows[
                rows["anatomy_name"].astype(str).str.lower().map(
                    lambda name: any(region in name or name in region for region in normalized_regions)
                )
            ]
        if rows.empty:
            return [], ["RadGenome中没有匹配请求区域的mask。"]

        rows["mask_priority"] = rows["mask_type"].astype(str).map(
            {"anatomy": 0, "region": 1}
        ).fillna(2)
        if requested_regions:
            rows["request_priority"] = rows["anatomy_name"].astype(str).str.lower().map(
                lambda name: 0 if name in exact_regions else 1
            )
        else:
            rows["request_priority"] = 1
        rows = rows.sort_values(["request_priority", "mask_priority", "anatomy_name"]).head(
            self.settings.radgenome_max_masks
        )

        ct_image = nib.load(str(ct_volume_path))
        ct_shape = tuple(int(value) for value in ct_image.shape[:3])
        results: list[AnatomyMaskResult] = []
        warnings: list[str] = []
        unverified_names: list[str] = []
        for _, row in rows.iterrows():
            mask_path = Path(str(row["mask_path"])).resolve()
            if self.mask_root not in mask_path.parents or not mask_path.exists():
                continue
            mask_image = nib.load(str(mask_path))
            mask = np.asarray(mask_image.dataobj, dtype=np.uint8)
            bounds = _bounds(mask)
            if bounds is None:
                continue
            lower, upper = bounds
            mapped_lower = [
                _map_index(lower[axis], mask.shape[axis], ct_shape[axis]) for axis in range(3)
            ]
            mapped_upper = [
                _map_index(upper[axis], mask.shape[axis], ct_shape[axis]) for axis in range(3)
            ]
            directly_aligned = str(row.get("aligned", False)).lower() in {
                "1",
                "true",
                "yes",
            }
            overlay = self._render_overlay(
                case_id,
                str(row["anatomy_name"]),
                ct_image,
                mask,
                mapped_lower[2],
                mapped_upper[2],
            )
            results.append(
                AnatomyMaskResult(
                    case_id=case_id,
                    anatomy_name=str(row["anatomy_name"]),
                    mask_type=str(row.get("mask_type", "region")),
                    mask_path=str(mask_path.as_posix()),
                    native_shape=list(mask.shape),
                    ct_shape=list(ct_shape),
                    slice_range=[mapped_lower[2], mapped_upper[2]],
                    bbox_3d=[*mapped_lower, *mapped_upper],
                    overlay_images=[overlay] if overlay else [],
                    alignment_method=("affine" if directly_aligned else "normalized_index_resample"),
                    alignment_verified=directly_aligned,
                )
            )
            if not directly_aligned:
                unverified_names.append(str(row["anatomy_name"]))
        if unverified_names:
            preview = "、".join(unverified_names[:5])
            suffix = f"等{len(unverified_names)}个mask" if len(unverified_names) > 5 else ""
            warnings.append(
                f"{preview}{suffix} 使用归一化索引映射，尚未通过原始affine验证。"
            )
        return results, sorted(set(warnings))

    def _render_overlay(
        self,
        case_id: str,
        anatomy_name: str,
        ct_image,
        mask: np.ndarray,
        ct_start: int,
        ct_end: int,
    ) -> str:
        ct_z = int(round((ct_start + ct_end) / 2))
        mask_z = _map_index(ct_z, ct_image.shape[2], mask.shape[2])
        ct_slice = np.asarray(ct_image.dataobj[:, :, ct_z], dtype=np.float32)
        mask_slice = np.asarray(mask[:, :, mask_z] > 0, dtype=np.uint8)
        resized = Image.fromarray(mask_slice * 255).resize(
            (ct_slice.shape[1], ct_slice.shape[0]), Image.Resampling.NEAREST
        )
        if 0 < np.count_nonzero(np.asarray(resized)) < 64:
            resized = resized.filter(ImageFilter.MaxFilter(11))
        mapped_mask = np.asarray(resized) > 0
        gray = np.uint8(np.clip((ct_slice + 1350.0) / 1500.0, 0.0, 1.0) * 255)
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb[mapped_mask, 0] = (rgb[mapped_mask, 0] * 0.35).astype(np.uint8)
        rgb[mapped_mask, 1] = np.maximum(rgb[mapped_mask, 1], 170)
        rgb[mapped_mask, 2] = np.maximum(rgb[mapped_mask, 2], 210)
        output_dir = Path(self.settings.static_dir) / "cases" / _safe_name(case_id) / "grounding"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{_safe_name(anatomy_name)}_slice_{ct_z:03d}.png"
        Image.fromarray(np.rot90(rgb)).save(output)
        return str(output.as_posix())
