import math
import re
from collections import Counter

from chestct_agent.knowledge import LABEL_KNOWLEDGE
from chestct_agent.schemas import RetrievedDocument


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z_-]+", text.lower())


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    dot = sum(a[k] * b[k] for k in set(a) & set(b))
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


class MedicalRagTool:
    """Small local RAG store; replace with vector DB when embeddings are configured."""

    def __init__(self) -> None:
        self.documents = []
        for label, entry in LABEL_KNOWLEDGE.items():
            text = f"{entry['title']}. {entry['zh']} {entry['imaging']} Terms: {', '.join(entry['terms'])}"
            self.documents.append(
                RetrievedDocument(
                    doc_id=f"knowledge:{label}",
                    title=entry["title"],
                    text=text,
                    score=0.0,
                    metadata={"label": label, "source": "local_knowledge"},
                )
            )

    def retrieve(self, queries: list[str], top_k: int = 5) -> list[RetrievedDocument]:
        query_vec = Counter(_tokens(" ".join(queries)))
        scored: list[RetrievedDocument] = []
        for doc in self.documents:
            score = _cosine(query_vec, Counter(_tokens(doc.text)))
            scored.append(doc.model_copy(update={"score": score}))
        scored.sort(key=lambda item: item.score, reverse=True)
        return [doc for doc in scored[:top_k] if doc.score > 0]


def grade_retrieval(documents: list[RetrievedDocument], min_docs: int = 2) -> bool:
    return len(documents) >= min_docs and any(doc.score >= 0.1 for doc in documents)

