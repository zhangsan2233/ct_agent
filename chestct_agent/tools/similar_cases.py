import math
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from chestct_agent.config import Settings
from chestct_agent.labels import LABEL_BY_ID, LABEL_IDS
from chestct_agent.schemas import SimilarCase


SIMILAR_CASE_INDEX_VERSION = 2
_CASE_ID_PATTERN = re.compile(r"^(train|valid)_(\d+)(?:_|$)", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z_-]+", text.lower())


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    dot = sum(a[k] * b[k] for k in set(a) & set(b))
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _patient_id(case_id: str) -> str:
    normalized = Path(str(case_id)).name
    if normalized.lower().endswith(".nii.gz"):
        normalized = normalized[:-7]
    else:
        normalized = Path(normalized).stem
    match = _CASE_ID_PATTERN.match(normalized)
    if match:
        return f"{match.group(1).lower()}_{match.group(2)}"
    return normalized


def _source_split(case_id: str) -> str:
    patient_id = _patient_id(case_id)
    return patient_id.split("_", 1)[0] if "_" in patient_id else "unknown"


def _summary(report_text: str, limit: int = 320) -> str:
    marker = "Impression:"
    index = report_text.lower().rfind(marker.lower())
    text = report_text[index + len(marker) :] if index >= 0 else report_text
    return " ".join(text.split())[:limit]


def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    if not len(scores) or count <= 0:
        return np.array([], dtype=int)
    count = min(len(scores), count)
    if count == len(scores):
        return np.argsort(scores)[::-1]
    indices = np.argpartition(scores, len(scores) - count)[-count:]
    return indices[np.argsort(scores[indices])[::-1]]


class SimilarCaseRetrieverTool:
    """Report/label similarity retriever over prepared CT-RATE cases."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.index_path = Path(settings.artifact_dir) / "prepared" / "case_index.csv"
        self.index = self._load_index()
        self.vector_index_path = (
            Path(settings.artifact_dir) / "prepared" / "similar_case_index.joblib"
        )
        self.vectorizer, self.report_matrix = self._load_vector_index()
        self.patient_ids = [
            _patient_id(str(case_id)) for case_id in self.index["case_id"].fillna("")
        ]
        self.row_labels = [
            {label for label in str(value).split(";") if label in LABEL_BY_ID}
            for value in self.index["labels"].fillna("")
        ]
        label_to_column = {label: index for index, label in enumerate(LABEL_IDS)}
        self.label_matrix = np.zeros((len(self.index), len(LABEL_IDS)), dtype=np.float32)
        for row_index, labels in enumerate(self.row_labels):
            for label in labels:
                self.label_matrix[row_index, label_to_column[label]] = 1.0

    def _load_index(self) -> pd.DataFrame:
        if self.index_path.exists():
            try:
                return pd.read_csv(self.index_path)
            except Exception:
                pass
        return pd.DataFrame(
            [
                {
                    "case_id": "demo_pleural_effusion",
                    "report_text": "Small bilateral pleural effusions with mild atelectasis.",
                    "labels": "pleural_effusion;atelectasis",
                },
                {
                    "case_id": "demo_nodule",
                    "report_text": "Right lower lobe pulmonary nodule. No pleural effusion.",
                    "labels": "pulmonary_nodule",
                },
            ]
        )

    def _load_vector_index(self):
        if not self.vector_index_path.exists() or not self.index_path.exists():
            return None, None
        try:
            artifact = joblib.load(self.vector_index_path)
            source_stat = self.index_path.stat()
            if artifact.get("version") != SIMILAR_CASE_INDEX_VERSION:
                return None, None
            if artifact.get("source_size") != source_stat.st_size:
                return None, None
            if artifact.get("source_mtime_ns") != source_stat.st_mtime_ns:
                return None, None
            if artifact["matrix"].shape[0] != len(self.index):
                return None, None
            return artifact["vectorizer"], artifact["matrix"]
        except (EOFError, KeyError, OSError, TypeError, ValueError):
            return None, None

    @staticmethod
    def _condition_query(labels: list[str]) -> str:
        parts: list[str] = []
        for label in labels[:6]:
            spec = LABEL_BY_ID.get(label)
            if spec is None:
                continue
            parts.extend([spec.title, " ".join(spec.terms[:6]), spec.imaging])
        return " ".join(parts)

    def _text_scores(self, query_text: str) -> np.ndarray:
        if not query_text.strip():
            return np.zeros(len(self.index), dtype=np.float32)
        if self.vectorizer is not None and self.report_matrix is not None:
            query_vector = self.vectorizer.transform([query_text])
            return np.asarray((self.report_matrix @ query_vector.T).toarray()).ravel()
        query = Counter(_tokens(query_text))
        return np.asarray(
            [
                _cosine(query, Counter(_tokens(str(text))))
                for text in self.index["report_text"].fillna("")
            ],
            dtype=np.float32,
        )

    def _condition_scores(
        self, labels: list[str], label_scores: dict[str, float]
    ) -> np.ndarray:
        weights = np.asarray(
            [max(0.0, float(label_scores.get(label, 1.0))) if label in labels else 0.0 for label in LABEL_IDS],
            dtype=np.float32,
        )
        query_weight = float(weights.sum())
        if not query_weight:
            return np.zeros(len(self.index), dtype=np.float32)
        intersection = self.label_matrix @ weights
        candidate_weight = self.label_matrix.sum(axis=1)
        mean_query_weight = query_weight / max(1, len(labels))
        union = query_weight + candidate_weight * mean_query_weight - intersection
        return np.divide(
            intersection,
            np.maximum(union, 1e-6),
            out=np.zeros_like(intersection),
            where=union > 0,
        )

    def _region_scores(
        self, anatomy_regions: list[str]
    ) -> tuple[np.ndarray, list[list[str]]]:
        normalized_regions = sorted(
            {" ".join(region.replace("_", " ").lower().split()) for region in anatomy_regions if region}
        )
        scores = np.zeros(len(self.index), dtype=np.float32)
        matches: list[list[str]] = [[] for _ in range(len(self.index))]
        if not normalized_regions:
            return scores, matches
        for index, report in enumerate(self.index["report_text"].fillna("")):
            normalized_report = " ".join(str(report).lower().replace("_", " ").split())
            found = [region for region in normalized_regions if region in normalized_report]
            matches[index] = found
            scores[index] = len(found) / len(normalized_regions)
        return scores, matches

    def retrieve(
        self,
        report_text: str,
        labels: list[str],
        top_k: int,
        query_case_id: str | None = None,
        label_scores: dict[str, float] | None = None,
        anatomy_regions: list[str] | None = None,
    ) -> list[SimilarCase]:
        labels = [label for label in labels if label in LABEL_BY_ID]
        if not report_text.strip() and not labels:
            return []
        label_scores = label_scores or {label: 1.0 for label in labels}
        anatomy_regions = anatomy_regions or []
        condition_query = self._condition_query(
            sorted(labels, key=lambda label: label_scores.get(label, 0.0), reverse=True)
        )
        query_text = report_text if report_text.strip() else condition_query
        text_scores = self._text_scores(query_text)
        condition_scores = self._condition_scores(labels, label_scores)
        region_scores, region_matches = self._region_scores(anatomy_regions)

        if report_text.strip() and labels:
            final_scores = 0.8 * text_scores + 0.2 * condition_scores
            strategy = "hybrid"
        elif report_text.strip():
            final_scores = text_scores
            strategy = "report_text"
        else:
            final_scores = 0.35 * text_scores + 0.65 * condition_scores
            strategy = "predicted_conditions"
        if anatomy_regions:
            final_scores = 0.9 * final_scores + 0.1 * region_scores
            strategy = "region_aware"

        candidate_count = min(len(self.index), max(200, top_k * 50))
        candidate_indices = set(_top_indices(final_scores, candidate_count).tolist())
        candidate_indices.update(_top_indices(text_scores, candidate_count // 2).tolist())
        candidate_indices.update(_top_indices(condition_scores, candidate_count // 2).tolist())
        ranked_indices = sorted(candidate_indices, key=lambda index: final_scores[index], reverse=True)

        query_patient_id = _patient_id(query_case_id) if query_case_id else ""
        seen_patients: set[str] = set()
        rows: list[SimilarCase] = []
        for index in ranked_indices:
            row = self.index.iloc[int(index)]
            row_case_id = str(row.get("case_id", "unknown"))
            patient_id = self.patient_ids[int(index)]
            if query_case_id and (
                row_case_id == query_case_id or patient_id == query_patient_id
            ):
                continue
            if patient_id in seen_patients:
                continue
            row_text = str(row.get("report_text", ""))
            row_labels = self.row_labels[int(index)]
            matched = sorted(set(labels) & row_labels)
            score = max(0.0, float(final_scores[int(index)]))
            if score <= 0 and rows:
                break
            rows.append(
                SimilarCase(
                    case_id=row_case_id,
                    score=round(float(score), 4),
                    matched_labels=matched,
                    summary=_summary(row_text),
                    patient_id=patient_id,
                    source="CT-RATE training reports and weak labels",
                    source_split=_source_split(row_case_id),
                    retrieval_strategy=strategy,
                    matched_regions=region_matches[int(index)],
                    score_breakdown={
                        "report_similarity": round(float(text_scores[int(index)]), 4),
                        "condition_overlap": round(float(condition_scores[int(index)]), 4),
                        "region_overlap": round(float(region_scores[int(index)]), 4),
                    },
                )
            )
            seen_patients.add(patient_id)
            if len(rows) >= top_k:
                break
        return rows
