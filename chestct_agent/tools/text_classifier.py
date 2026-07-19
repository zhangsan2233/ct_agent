from pathlib import Path

import joblib

from chestct_agent.calibration import CalibrationStore
from chestct_agent.config import Settings
from chestct_agent.labels import LABEL_BY_ID, LABEL_IDS
from chestct_agent.schemas import LabelPrediction, ParsedReport, ReportEvidence
from chestct_agent.tools.evidence_extractor import extract_evidence


def _status_from_probability(
    probability: float,
    positive_threshold: float,
    uncertain_threshold: float,
) -> str:
    if probability >= positive_threshold:
        return "positive"
    if probability >= uncertain_threshold:
        return "uncertain"
    return "negative"


def _apply_direct_evidence(
    probability: float,
    items: list[ReportEvidence],
    positive_threshold: float,
    uncertain_threshold: float,
) -> tuple[str, float]:
    polarities = {item.polarity for item in items}
    if "positive" in polarities and "negative" in polarities:
        return "uncertain", 0.5
    if "positive" in polarities:
        return "positive", max(probability, 0.9)
    if "uncertain" in polarities:
        return "uncertain", max(min(probability, 0.79), 0.55)
    if "negative" in polarities:
        return "negative", min(probability, 0.05)
    if "historical" in polarities:
        return "uncertain", max(min(probability, 0.49), uncertain_threshold)
    return _status_from_probability(
        probability, positive_threshold, uncertain_threshold
    ), probability


class TextClassifierTool:
    """Eighteen-label report classifier with explicit evidence overrides."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_path = Path(settings.artifact_dir) / "text_classifier.joblib"
        self.model = None
        self.labels: list[str] = []
        self.calibration = CalibrationStore(settings)
        if self.model_path.exists():
            try:
                artifact = joblib.load(self.model_path)
                if isinstance(artifact, dict) and "model" in artifact:
                    self.model = artifact["model"]
                    self.labels = [str(label) for label in artifact.get("labels", [])]
                else:
                    self.model = artifact
                    self.labels = [str(label) for label in getattr(self.model, "classes_", [])]
            except Exception:
                self.model = None

    def predict(self, parsed_report: ParsedReport) -> list[LabelPrediction]:
        text = parsed_report.full_report
        scores = {label: 0.05 for label in LABEL_IDS}
        if self.model is not None:
            probabilities = self.model.predict_proba([text])[0]
            for label, probability in zip(self.labels, probabilities, strict=False):
                if label in LABEL_BY_ID:
                    scores[label] = float(probability)

        evidence = extract_evidence(text, LABEL_IDS)
        predictions: list[LabelPrediction] = []
        for label in LABEL_IDS:
            calibrated = self.calibration.calibrate("report", label, scores[label])
            status, probability = _apply_direct_evidence(
                calibrated.probability,
                evidence[label],
                calibrated.positive_threshold,
                calibrated.uncertain_threshold,
            )
            predictions.append(
                LabelPrediction(
                    name=label,
                    status=status,
                    confidence=round(float(probability), 6),
                    source="report",
                    calibrated=calibrated.calibrated,
                    calibration_version=calibrated.version,
                )
            )
        return predictions
