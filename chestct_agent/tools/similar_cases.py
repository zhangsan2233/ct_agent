import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from chestct_agent.config import Settings
from chestct_agent.schemas import SimilarCase


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z_-]+", text.lower())


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    dot = sum(a[k] * b[k] for k in set(a) & set(b))
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


class SimilarCaseRetrieverTool:
    """Report/label similarity retriever over prepared CT-RATE cases."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.index_path = Path(settings.artifact_dir) / "prepared" / "case_index.csv"
        self.index = self._load_index()

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

    def retrieve(self, report_text: str, labels: list[str], top_k: int) -> list[SimilarCase]:
        query_vec = Counter(_tokens(report_text + " " + " ".join(labels)))
        rows: list[SimilarCase] = []
        for _, row in self.index.iterrows():
            row_text = str(row.get("report_text", ""))
            row_labels = [x for x in str(row.get("labels", "")).split(";") if x]
            score = _cosine(query_vec, Counter(_tokens(row_text + " " + " ".join(row_labels))))
            matched = sorted(set(labels) & set(row_labels))
            if matched:
                score += 0.2
            rows.append(
                SimilarCase(
                    case_id=str(row.get("case_id", "unknown")),
                    score=round(float(score), 4),
                    matched_labels=matched,
                    summary=row_text[:220],
                )
            )
        rows.sort(key=lambda item: item.score, reverse=True)
        return rows[:top_k]

