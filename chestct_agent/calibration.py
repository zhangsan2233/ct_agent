from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from chestct_agent.config import Settings


@dataclass(frozen=True)
class CalibratedScore:
    probability: float
    positive_threshold: float
    uncertain_threshold: float
    calibrated: bool
    version: str | None


class CalibrationStore:
    """Loads patient-held-out per-label calibrators without making them mandatory."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = Path(settings.calibration_path)
        self.artifact: dict = {}
        if self.path.exists():
            try:
                loaded = joblib.load(self.path)
                if isinstance(loaded, dict):
                    self.artifact = loaded
            except Exception:
                self.artifact = {}

    def calibrate(self, source: str, label: str, score: float) -> CalibratedScore:
        source_items = self.artifact.get("sources", {}).get(source, {})
        item = source_items.get(label, {}) if isinstance(source_items, dict) else {}
        probability = float(score)
        model = item.get("model") if isinstance(item, dict) else None
        calibrated = bool(item)
        if model is not None:
            try:
                probability = float(model.predict_proba(np.asarray([[score]]))[:, 1][0])
                calibrated = True
            except Exception:
                probability = float(score)
        return CalibratedScore(
            probability=min(max(probability, 0.0), 1.0),
            positive_threshold=float(
                item.get("positive_threshold", self.settings.positive_label_threshold)
            ),
            uncertain_threshold=float(
                item.get("uncertain_threshold", self.settings.min_label_confidence)
            ),
            calibrated=calibrated,
            version=str(self.artifact.get("version")) if calibrated else None,
        )
