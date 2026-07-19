import re
from pathlib import Path

import joblib

from chestct_agent.config import Settings
from chestct_agent.knowledge import LABEL_KNOWLEDGE, NEGATION_TERMS, UNCERTAIN_TERMS
from chestct_agent.schemas import LabelPrediction, ParsedReport


def _contains_negation(window: str) -> bool:
    lower = window.lower()
    return any(term in lower for term in NEGATION_TERMS)


def _contains_uncertainty(window: str) -> bool:
    lower = window.lower()
    return any(term in lower for term in UNCERTAIN_TERMS)


def _sentences(text: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", text) if piece.strip()]


class TextClassifierTool:
    """Report multi-label classifier with optional sklearn model and keyword fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_path = Path(settings.artifact_dir) / "text_classifier.joblib"
        self.model = None
        self.labels = None
        if self.model_path.exists():
            try:
                artifact = joblib.load(self.model_path)
                if isinstance(artifact, dict) and "model" in artifact:
                    self.model = artifact["model"]
                    self.labels = artifact.get("labels")
                else:
                    self.model = artifact
            except Exception:
                self.model = None

    def predict(self, parsed_report: ParsedReport) -> list[LabelPrediction]:
        if self.model is not None:
            return self._predict_model(parsed_report.full_report)
        return self._predict_keyword(parsed_report.full_report)

    def _predict_model(self, text: str) -> list[LabelPrediction]:
        probabilities = self.model.predict_proba([text])
        labels = self.labels or getattr(self.model, "classes_", list(LABEL_KNOWLEDGE.keys()))
        predictions: list[LabelPrediction] = []
        for label, probability in zip(labels, probabilities[0], strict=False):
            status = "positive" if probability >= self.settings.min_label_confidence else "negative"
            predictions.append(
                LabelPrediction(
                    name=str(label),
                    status=status,
                    confidence=float(probability),
                    source="report",
                )
            )
        return predictions

    def _predict_keyword(self, text: str) -> list[LabelPrediction]:
        lower_text = text.lower()
        sentences = _sentences(lower_text)
        predictions: list[LabelPrediction] = []
        for label, entry in LABEL_KNOWLEDGE.items():
            confidence = 0.05
            status = "negative"
            for term in entry["terms"]:
                pattern = rf"\b{re.escape(term.lower())}\b"
                for sentence in sentences:
                    if not re.search(pattern, sentence):
                        continue
                    if _contains_negation(sentence):
                        confidence = max(confidence, 0.15)
                        status = "negative"
                    elif _contains_uncertainty(sentence):
                        confidence = max(confidence, 0.55)
                        status = "uncertain"
                    else:
                        confidence = max(confidence, 0.82)
                        status = "positive"
            predictions.append(
                LabelPrediction(name=label, status=status, confidence=confidence, source="report")
            )
        return predictions
