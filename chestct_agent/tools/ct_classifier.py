import json
from pathlib import Path
import subprocess
import sys

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
        )

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
    ) -> tuple[list[LabelPrediction], list[str]]:
        if not ct_volume_path and not preview_images:
            return [], []
        if self.settings.ct_model_backend.lower() != "ct-clip":
            return [], [f"CT model backend is disabled: {self.settings.ct_model_backend}"]
        if not ct_volume_path:
            return [], ["CT-CLIP requires the original NIfTI volume, not preview images alone."]
        asset_error = self.runtime.asset_error()
        if asset_error:
            return [], [asset_error]

        try:
            python_path = Path(self.settings.ctclip_python)
            if python_path.exists() and python_path.resolve() != Path(sys.executable).resolve():
                probabilities = self._predict_external(ct_volume_path)
            else:
                probabilities = self.runtime.predict(ct_volume_path)
        except CtClipUnavailable as exc:
            return [], [str(exc)]
        except subprocess.TimeoutExpired:
            return [], ["CT-CLIP inference timed out."]
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                message = "CT-CLIP ran out of GPU memory; use CPU mode or a larger GPU."
            return [], [message]

        predictions: list[LabelPrediction] = []
        for label, probability in probabilities.items():
            if probability >= 0.5:
                status = "positive"
            elif probability >= 0.35:
                status = "uncertain"
            else:
                status = "negative"
            predictions.append(
                LabelPrediction(
                    name=label,
                    status=status,
                    confidence=round(probability, 4),
                    source="ct",
                )
            )
        return predictions, []
