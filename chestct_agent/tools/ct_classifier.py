import hashlib
import json
from pathlib import Path
import subprocess
import sys

from chestct_agent.calibration import CalibrationStore
from chestct_agent.config import Settings
from chestct_agent.ctclip import CtClipRuntime, CtClipUnavailable
from chestct_agent.schemas import LabelPrediction


class CtClassifierTool:
    """Runs the official pretrained CT-CLIP model when its assets are ready."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.runtime = CtClipRuntime(
            checkpoint=settings.ctclip_checkpoint,
            source_dir=settings.ctclip_source_dir,
            device=settings.ctclip_device,
            use_fp16=settings.ctclip_use_fp16,
            variant=settings.ctclip_variant,
        )
        self.cache_dir = Path(settings.artifact_dir) / "ct_cache"
        self.calibration = CalibrationStore(settings)

    def _cache_path(self, ct_volume_path: str) -> Path | None:
        if not self.settings.ct_cache_enabled:
            return None
        volume = Path(ct_volume_path)
        checkpoint = Path(self.settings.ctclip_checkpoint)
        if not volume.exists() or not checkpoint.exists():
            return None
        volume_stat = volume.stat()
        checkpoint_stat = checkpoint.stat()
        fingerprint = json.dumps(
            {
                "version": 2,
                "variant": self.settings.ctclip_variant,
                "volume": str(volume.resolve()),
                "volume_size": volume_stat.st_size,
                "volume_mtime_ns": volume_stat.st_mtime_ns,
                "checkpoint_size": checkpoint_stat.st_size,
                "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
                "fp16": self.settings.ctclip_use_fp16,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _load_cache(self, ct_volume_path: str) -> dict[str, float] | None:
        cache_path = self._cache_path(ct_volume_path)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return {key: float(value) for key, value in payload["probabilities"].items()}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, ct_volume_path: str, probabilities: dict[str, float]) -> None:
        cache_path = self._cache_path(ct_volume_path)
        if cache_path is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"probabilities": probabilities}, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        except OSError:
            # Cache failures must not discard a successful model inference.
            return

    def readiness_error(self) -> str | None:
        if self.settings.ct_model_backend.lower() != "ct-clip":
            return f"CT model backend is disabled: {self.settings.ct_model_backend}"
        asset_error = self.runtime.asset_error()
        if asset_error:
            return asset_error
        python_path = Path(self.settings.ctclip_python)
        if python_path.exists() and python_path.resolve() != Path(sys.executable).resolve():
            return None
        return self.runtime.readiness_error()

    def _predict_external(self, ct_volume_path: str) -> dict[str, float]:
        project_root = Path(__file__).resolve().parents[2]
        command = [
            str(self.settings.ctclip_python),
            str(project_root / "scripts" / "ctclip_worker.py"),
            "--volume",
            ct_volume_path,
            "--checkpoint",
            str(self.settings.ctclip_checkpoint),
            "--source-dir",
            str(self.settings.ctclip_source_dir),
            "--device",
            self.settings.ctclip_device,
            "--variant",
            self.settings.ctclip_variant,
        ]
        if self.settings.ctclip_use_fp16:
            command.append("--fp16")
        completed = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=self.settings.ctclip_timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            error_lines = [line for line in completed.stderr.splitlines() if line.strip()]
            raise RuntimeError(error_lines[-1] if error_lines else "CT-CLIP worker failed.")
        output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not output_lines:
            raise RuntimeError("CT-CLIP worker returned no output.")
        return {key: float(value) for key, value in json.loads(output_lines[-1]).items()}

    def predict(
        self, ct_volume_path: str | None, preview_images: list[str]
    ) -> tuple[list[LabelPrediction], list[str], bool | None]:
        if not ct_volume_path and not preview_images:
            return [], [], None
        if self.settings.ct_model_backend.lower() != "ct-clip":
            return [], [f"CT model backend is disabled: {self.settings.ct_model_backend}"], None
        if not ct_volume_path:
            return [], ["CT-CLIP requires the original NIfTI volume, not preview images alone."], None
        asset_error = self.runtime.asset_error()
        if asset_error:
            return [], [asset_error], None

        probabilities = self._load_cache(ct_volume_path)
        cache_hit = probabilities is not None
        try:
            if probabilities is None:
                python_path = Path(self.settings.ctclip_python)
                if python_path.exists() and python_path.resolve() != Path(sys.executable).resolve():
                    probabilities = self._predict_external(ct_volume_path)
                else:
                    probabilities = self.runtime.predict(ct_volume_path)
                self._write_cache(ct_volume_path, probabilities)
        except CtClipUnavailable as exc:
            return [], [str(exc)], None
        except subprocess.TimeoutExpired:
            return [], ["CT-CLIP inference timed out."], None
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                message = "CT-CLIP ran out of GPU memory; use CPU mode or a larger GPU."
            return [], [message], None

        predictions: list[LabelPrediction] = []
        for label, probability in probabilities.items():
            calibrated = self.calibration.calibrate("ct", label, probability)
            probability = calibrated.probability
            if probability >= calibrated.positive_threshold:
                status = "positive"
            elif probability >= calibrated.uncertain_threshold:
                status = "uncertain"
            else:
                status = "negative"
            predictions.append(
                LabelPrediction(
                    name=label,
                    status=status,
                    confidence=round(probability, 4),
                    source="ct",
                    calibrated=calibrated.calibrated,
                    calibration_version=calibrated.version,
                )
            )
        warnings: list[str] = []
        positive_count = sum(item.status == "positive" for item in predictions)
        if positive_count > self.settings.ct_max_positive_labels:
            predictions = [
                item.model_copy(update={"status": "uncertain"})
                if item.status == "positive"
                else item
                for item in predictions
            ]
            warnings.append(
                "CT质量门控触发：CT-CLIP 将 "
                f"{positive_count}/{len(predictions)} 类标记为阳性，超过允许上限 "
                f"{self.settings.ct_max_positive_labels}；系统已取消这些阳性结论并降级为待复核候选。"
            )
        return predictions, warnings, cache_hit
