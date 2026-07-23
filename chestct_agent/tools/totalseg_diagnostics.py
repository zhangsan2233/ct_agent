from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

from chestct_agent.config import Settings
from chestct_agent.schemas import DiagnosticToolEvidence


TASK_CLASSES = {
    "lung_nodules": {"lung_nodules": 2},
    "pleural_pericard_effusion": {
        "lung_pleural": 1,
        "pleural_effusion": 2,
        "pericardial_effusion": 3,
    },
    "total": {
        "lung_upper_lobe_left": 10,
        "lung_lower_lobe_left": 11,
        "lung_upper_lobe_right": 12,
        "lung_middle_lobe_right": 13,
        "lung_lower_lobe_right": 14,
        "heart": 51,
    },
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "case"


def _window(slice_data: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    return np.uint8(np.clip((slice_data - low) / width, 0.0, 1.0) * 255.0)


def _top_slices(mask: np.ndarray, count: int = 3, minimum_gap: int = 4) -> list[int]:
    areas = np.count_nonzero(mask, axis=(0, 1))
    selected: list[int] = []
    for index in np.argsort(areas)[::-1]:
        index = int(index)
        if areas[index] <= 0:
            break
        if all(abs(index - other) >= minimum_gap for other in selected):
            selected.append(index)
        if len(selected) >= count:
            break
    return sorted(selected)


def _bbox(mask: np.ndarray) -> list[int]:
    if not np.any(mask):
        return []
    xs = np.flatnonzero(mask.any(axis=(1, 2)))
    ys = np.flatnonzero(mask.any(axis=(0, 2)))
    zs = np.flatnonzero(mask.any(axis=(0, 1)))
    return [int(xs[0]), int(ys[0]), int(zs[0]), int(xs[-1]), int(ys[-1]), int(zs[-1])]


class TotalSegmentatorDiagnosticTool:
    """Runs independent 3D segmentation and turns masks into label evidence."""

    version = "totalsegmentator-diagnostics-v1"

    def __init__(self, settings: Settings):
        self.settings = settings

    def available(self) -> bool:
        return self.settings.totalseg_enabled and self._executable() is not None

    def analyze(
        self,
        case_id: str,
        volume_path: str | None,
        requested_labels: set[str] | None = None,
    ) -> tuple[list[DiagnosticToolEvidence], list[str], float]:
        started = time.perf_counter()
        requested = requested_labels or {
            "pulmonary_nodule",
            "pleural_effusion",
            "pericardial_effusion",
            "cardiomegaly",
        }
        if not volume_path or not Path(volume_path).is_file():
            return [], ["专病影像工具缺少可读取的CT体数据。"], 0.0
        executable = self._executable()
        if not self.settings.totalseg_enabled:
            return [], ["TotalSegmentator专病工具已禁用。"], 0.0
        if executable is None:
            return [], ["未找到TotalSegmentator可执行文件，专病分割未运行。"], 0.0

        volume = Path(volume_path).resolve()
        warnings: list[str] = []
        task_outputs: dict[str, tuple[Path, bool, float]] = {}
        needed_tasks = []
        if "pulmonary_nodule" in requested:
            needed_tasks.append("lung_nodules")
        if requested & {"pleural_effusion", "pericardial_effusion"}:
            needed_tasks.append("pleural_pericard_effusion")
        if "cardiomegaly" in requested:
            needed_tasks.append("total")

        for task in needed_tasks:
            try:
                task_outputs[task] = self._run_task(executable, case_id, volume, task)
            except Exception as exc:
                detail = str(exc).strip().replace("\n", " ")[-500:]
                warnings.append(
                    f"{task}分割失败：{type(exc).__name__}：{detail or '无错误详情'}。"
                )

        ct_image = nib.load(str(volume))
        ct_volume = np.asarray(ct_image.dataobj, dtype=np.float32)
        evidence: list[DiagnosticToolEvidence] = []
        if "lung_nodules" in task_outputs:
            output, cache_hit, task_latency = task_outputs["lung_nodules"]
            segmentation = self._load_segmentation(output, "lung_nodules")
            if segmentation is not None:
                evidence.extend(
                    self._nodule_evidence(
                        case_id,
                        ct_image,
                        ct_volume,
                        segmentation,
                        output,
                        cache_hit,
                        task_latency,
                        requested,
                    )
                )
        if "pleural_pericard_effusion" in task_outputs:
            output, cache_hit, task_latency = task_outputs["pleural_pericard_effusion"]
            segmentation = self._load_segmentation(output, "pleural_pericard_effusion")
            if segmentation is not None:
                evidence.extend(
                    self._effusion_evidence(
                        case_id,
                        ct_image,
                        ct_volume,
                        segmentation,
                        output,
                        cache_hit,
                        task_latency,
                        requested,
                    )
                )
        if "total" in task_outputs:
            output, cache_hit, task_latency = task_outputs["total"]
            segmentation = self._load_segmentation(output, "total")
            if segmentation is not None:
                evidence.extend(
                    self._cardiac_evidence(
                        case_id,
                        ct_image,
                        ct_volume,
                        segmentation,
                        output,
                        cache_hit,
                        task_latency,
                        requested,
                    )
                )

        total_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return evidence, warnings, total_ms

    def _executable(self) -> Path | None:
        configured = Path(self.settings.totalseg_executable)
        if configured.is_file():
            return configured.resolve()
        discovered = shutil.which("TotalSegmentator")
        return Path(discovered).resolve() if discovered else None

    @staticmethod
    def _fingerprint(volume: Path, task: str) -> str:
        stat = volume.stat()
        raw = f"{volume.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{task}|2.16|ml|ns1"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _run_task(
        self, executable: Path, case_id: str, volume: Path, task: str
    ) -> tuple[Path, bool, float]:
        fingerprint = self._fingerprint(volume, task)
        out_dir = Path(self.settings.totalseg_cache_dir) / _safe_name(case_id) / task
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "segmentation.nii.gz"
        report = out_dir / "run_report.json"
        manifest = out_dir / "cache_manifest.json"
        if output.is_file() and report.is_file() and manifest.is_file():
            try:
                cached = json.loads(manifest.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == fingerprint:
                    return output, True, 0.0
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        legacy_outputs = {
            "lung_nodules": [out_dir / "lung.nii.gz", out_dir / "lung_nodules.nii.gz"],
            "pleural_pericard_effusion": [out_dir / "effusion_multilabel.nii.gz"],
        }
        legacy_candidates = legacy_outputs.get(task, [])
        if legacy_candidates and all(path.is_file() for path in legacy_candidates):
            return output, True, 0.0

        command = [
            str(executable),
            "-i",
            str(volume),
            "-o",
            str(output.resolve()),
            "-ta",
            task,
            "-d",
            self.settings.totalseg_device,
            "-ml",
            "-nr",
            "1",
            "-ns",
            "1",
            "-rp",
            str(report.resolve()),
            "-q",
        ]
        if task == "total":
            command.extend(
                [
                    "-f",
                    "-rs",
                    "heart",
                    "lung_upper_lobe_left",
                    "lung_lower_lobe_left",
                    "lung_upper_lobe_right",
                    "lung_middle_lobe_right",
                    "lung_lower_lobe_right",
                ]
            )
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
            timeout=self.settings.totalseg_timeout_seconds,
            check=False,
        )
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if completed.returncode != 0 or not output.is_file():
            detail = (completed.stderr or completed.stdout)[-800:]
            raise RuntimeError(f"TotalSegmentator {task} failed: {detail}")
        manifest.write_text(
            json.dumps(
                {
                    "version": self.version,
                    "fingerprint": fingerprint,
                    "task": task,
                    "input": str(volume),
                    "latency_ms": latency_ms,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return output, False, latency_ms

    @staticmethod
    def _load_segmentation(output: Path, task: str) -> np.ndarray | None:
        if output.is_file():
            return np.asarray(nib.load(str(output)).dataobj, dtype=np.uint8)
        legacy_multilabel = output.parent / "effusion_multilabel.nii.gz"
        if task == "pleural_pericard_effusion" and legacy_multilabel.is_file():
            return np.asarray(nib.load(str(legacy_multilabel)).dataobj, dtype=np.uint8)
        # Supports early experimental caches that wrote one binary file per class.
        class_map = TASK_CLASSES[task]
        arrays = []
        reference = None
        for name, value in class_map.items():
            path = output.parent / f"{name}.nii.gz"
            if not path.is_file():
                continue
            binary = np.asarray(nib.load(str(path)).dataobj, dtype=np.uint8) > 0
            if reference is None:
                reference = np.zeros(binary.shape, dtype=np.uint8)
            arrays.append((binary, value))
        if reference is None:
            return None
        for binary, value in arrays:
            reference[binary] = value
        return reference

    def _nodule_evidence(
        self,
        case_id: str,
        ct_image,
        ct_volume: np.ndarray,
        segmentation: np.ndarray,
        mask_path: Path,
        cache_hit: bool,
        latency_ms: float,
        requested: set[str],
    ) -> list[DiagnosticToolEvidence]:
        result: list[DiagnosticToolEvidence] = []
        spacing = tuple(float(value) for value in ct_image.header.get_zooms()[:3])
        voxel_mm3 = float(np.prod(spacing))
        if "pulmonary_nodule" in requested:
            nodule_mask = segmentation == TASK_CLASSES["lung_nodules"]["lung_nodules"]
            bbox = _bbox(nodule_mask)
            component_count = 0
            qualifying_count = 0
            maximum_diameter = 0.0
            if bbox:
                x1, y1, z1, x2, y2, z2 = bbox
                cropped = nodule_mask[x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1]
                components, component_count = ndimage.label(
                    cropped, structure=np.ones((3, 3, 3), dtype=np.uint8)
                )
                counts = np.bincount(components.ravel())[1:]
                volumes = counts.astype(float) * voxel_mm3
                diameters = np.cbrt(6.0 * volumes / math.pi)
                qualifying_count = int(np.count_nonzero(diameters >= 3.0))
                maximum_diameter = float(diameters.max(initial=0.0))
            total_volume_ml = float(np.count_nonzero(nodule_mask) * voxel_mm3 / 1000.0)
            if qualifying_count:
                verdict, confidence = "positive", min(0.97, 0.82 + maximum_diameter / 250.0)
            elif component_count:
                verdict, confidence = "uncertain", 0.58
            else:
                verdict, confidence = "negative", 0.90
            slices = _top_slices(nodule_mask)
            previews = self._render_overlays(
                case_id,
                "pulmonary_nodule",
                ct_volume,
                nodule_mask,
                slices,
                -600.0,
                1500.0,
            )
            result.append(
                DiagnosticToolEvidence(
                    label="pulmonary_nodule",
                    tool="nodule_segmentation_tool",
                    backend="TotalSegmentator",
                    model_version="2.16.0/lung_nodules",
                    verdict=verdict,
                    confidence=confidence,
                    coverage="complete",
                    metrics={
                        "candidate_volume_ml": round(total_volume_ml, 3),
                        "component_count": float(component_count),
                        "components_ge_3mm": float(qualifying_count),
                        "max_equivalent_diameter_mm": round(maximum_diameter, 3),
                    },
                    mask_paths=[str(mask_path.as_posix())],
                    slice_indices=slices,
                    preview_images=previews,
                    cache_hit=cache_hit,
                    latency_ms=latency_ms,
                    rationale_zh=(
                        f"3D候选分割得到{qualifying_count}个等效直径不小于3 mm的连通区域。"
                    ),
                    limitation_zh=(
                        "该模型可能把肺门肿块、实变或血管邻近区域并入候选，阳性候选仍需切片VLM复核。"
                    ),
                )
            )

        return result

    def _effusion_evidence(
        self,
        case_id: str,
        ct_image,
        ct_volume: np.ndarray,
        segmentation: np.ndarray,
        mask_path: Path,
        cache_hit: bool,
        latency_ms: float,
        requested: set[str],
    ) -> list[DiagnosticToolEvidence]:
        spacing = tuple(float(value) for value in ct_image.header.get_zooms()[:3])
        voxel_mm3 = float(np.prod(spacing))
        configurations = {
            "pleural_effusion": (
                self.settings.pleural_effusion_uncertain_ml,
                self.settings.pleural_effusion_positive_ml,
            ),
            "pericardial_effusion": (
                self.settings.pericardial_effusion_uncertain_ml,
                self.settings.pericardial_effusion_positive_ml,
            ),
        }
        result = []
        for label, (uncertain_ml, positive_ml) in configurations.items():
            if label not in requested:
                continue
            value = TASK_CLASSES["pleural_pericard_effusion"][label]
            mask = segmentation == value
            volume_ml = float(np.count_nonzero(mask) * voxel_mm3 / 1000.0)
            if volume_ml >= positive_ml:
                verdict = "positive"
                confidence = min(0.97, 0.80 + (volume_ml - positive_ml) / max(positive_ml, 1) * 0.08)
            elif volume_ml >= uncertain_ml:
                verdict, confidence = "uncertain", 0.60
            else:
                verdict, confidence = "negative", 0.90
            slices = _top_slices(mask)
            previews = self._render_overlays(
                case_id, label, ct_volume, mask, slices, 40.0, 400.0
            )
            result.append(
                DiagnosticToolEvidence(
                    label=label,
                    tool="effusion_segmentation_tool",
                    backend="TotalSegmentator",
                    model_version="2.16.0/pleural_pericard_effusion",
                    verdict=verdict,
                    confidence=confidence,
                    coverage="complete",
                    metrics={
                        "volume_ml": round(volume_ml, 3),
                        "uncertain_threshold_ml": uncertain_ml,
                        "positive_threshold_ml": positive_ml,
                    },
                    mask_paths=[str(mask_path.as_posix())],
                    slice_indices=slices,
                    preview_images=previews,
                    cache_hit=cache_hit,
                    latency_ms=latency_ms,
                    rationale_zh=(
                        f"3D分割体积为{volume_ml:.1f} mL；阳性阈值为{positive_ml:.1f} mL。"
                    ),
                    limitation_zh="小体积生理性液体和部分容积效应可能造成弱阳性mask。",
                )
            )
        return result

    def _cardiac_evidence(
        self,
        case_id: str,
        ct_image,
        ct_volume: np.ndarray,
        segmentation: np.ndarray,
        mask_path: Path,
        cache_hit: bool,
        latency_ms: float,
        requested: set[str],
    ) -> list[DiagnosticToolEvidence]:
        spacing = tuple(float(value) for value in ct_image.header.get_zooms()[:3])
        heart = segmentation == TASK_CLASSES["total"]["heart"]
        lungs = np.isin(
            segmentation,
            [
                TASK_CLASSES["total"][name]
                for name in TASK_CLASSES["total"]
                if name.startswith("lung_")
            ],
        )
        result: list[DiagnosticToolEvidence] = []

        if "cardiomegaly" in requested:
            heart_widths = np.zeros(heart.shape[2], dtype=np.float32)
            for index in np.flatnonzero(np.any(heart, axis=(0, 1))):
                xs = np.flatnonzero(np.any(heart[:, :, index], axis=1))
                if xs.size:
                    heart_widths[index] = (xs[-1] - xs[0] + 1) * spacing[0]
            slice_index = int(np.argmax(heart_widths)) if heart_widths.size else 0
            thorax = heart[:, :, slice_index] | lungs[:, :, slice_index]
            thorax_xs = np.flatnonzero(np.any(thorax, axis=1))
            thorax_width = (
                float((thorax_xs[-1] - thorax_xs[0] + 1) * spacing[0])
                if thorax_xs.size
                else 0.0
            )
            heart_width = float(heart_widths[slice_index])
            ratio = heart_width / thorax_width if thorax_width > 0 else 0.0
            if not np.any(heart) or thorax_width <= 0:
                verdict, confidence, coverage = "unavailable", 0.0, "unavailable"
            elif ratio >= self.settings.cardiothoracic_ratio_positive:
                verdict, confidence, coverage = "positive", 0.78, "complete"
            elif ratio <= self.settings.cardiothoracic_ratio_positive - 0.05:
                verdict, confidence, coverage = "negative", 0.80, "complete"
            else:
                verdict, confidence, coverage = "uncertain", 0.60, "complete"
            previews = self._render_overlays(
                case_id,
                "cardiomegaly",
                ct_volume,
                heart,
                [slice_index] if np.any(heart) else [],
                40.0,
                400.0,
            )
            result.append(
                DiagnosticToolEvidence(
                    label="cardiomegaly",
                    tool="cardiac_measurement_tool",
                    backend="TotalSegmentator + deterministic CT measurement",
                    model_version="2.16.0/total-fast/ct-ctr-v1",
                    verdict=verdict,
                    confidence=confidence,
                    coverage=coverage,
                    metrics={
                        "ct_cardiothoracic_ratio": round(ratio, 4),
                        "heart_width_mm": round(heart_width, 2),
                        "thorax_width_mm": round(thorax_width, 2),
                    },
                    mask_paths=[str(mask_path.as_posix())],
                    slice_indices=[slice_index] if np.any(heart) else [],
                    preview_images=previews,
                    cache_hit=cache_hit,
                    latency_ms=latency_ms,
                    rationale_zh=(
                        f"最大心脏横径与同层胸廓横径之比为 {ratio:.3f}；"
                        f"研究阈值为 {self.settings.cardiothoracic_ratio_positive:.2f}。"
                    ),
                    limitation_zh=(
                        "该指标是CT筛查测量，不等价于超声心动图或心腔容积；"
                        "当前需经本地队列校准后才参与标签融合。"
                    ),
                )
            )

        return result

    def _render_overlays(
        self,
        case_id: str,
        label: str,
        ct_volume: np.ndarray,
        mask: np.ndarray,
        slices: list[int],
        center: float,
        width: float,
    ) -> list[str]:
        out_dir = (
            Path(self.settings.static_dir)
            / "cases"
            / _safe_name(case_id)
            / "diagnostic_tools"
            / label
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for index in slices:
            ct_slice = ct_volume[:, :, index]
            gray = _window(ct_slice, center, width)
            rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
            current = mask[:, :, index]
            rgb[current] = rgb[current] * 0.35 + np.asarray([255.0, 68.0, 24.0]) * 0.65
            output = out_dir / f"slice_{index:04d}_{label}.png"
            Image.fromarray(np.rot90(np.uint8(np.clip(rgb, 0, 255)))).save(output)
            outputs.append(str(output.as_posix()))
        return outputs
