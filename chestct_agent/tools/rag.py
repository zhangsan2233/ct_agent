import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from rank_bm25 import BM25Okapi

from chestct_agent.config import Settings
from chestct_agent.labels import LABEL_SPECS
from chestct_agent.schemas import RetrievedDocument


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+|[\u4e00-\u9fff]", text.lower())


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    low = float(scores.min())
    high = float(scores.max())
    if math.isclose(low, high):
        return np.ones_like(scores) if high > 0 else np.zeros_like(scores)
    return (scores - low) / (high - low)


class MedicalRagTool:
    """BM25 + Qwen dense retrieval + Qwen reranking with explicit degradation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.documents = self._load_documents()
        self.corpus = [f"{doc.title}. {doc.text}" for doc in self.documents]
        self.bm25 = BM25Okapi([_tokens(text) for text in self.corpus])
        self.embedding_model = None
        self.reranker_model = None
        self.qdrant_client = None
        self.collection_name = self._collection_name()

    def _load_documents(self) -> list[RetrievedDocument]:
        documents: list[RetrievedDocument] = []
        for spec in LABEL_SPECS:
            sections = (
                (
                    "definition",
                    f"{spec.definition} Chinese label: {spec.zh}.",
                ),
                (
                    "imaging",
                    f"Imaging appearance: {spec.imaging} Relevant anatomy: {', '.join(spec.anatomy_regions)}.",
                ),
                (
                    "terminology",
                    f"Report terminology and synonyms: {', '.join(spec.terms)}.",
                ),
            )
            for section, text in sections:
                documents.append(
                    RetrievedDocument(
                        doc_id=f"knowledge:{spec.id}:{section}",
                        title=f"{spec.title} {section}",
                        text=text,
                        score=0.0,
                        metadata={
                            "label": spec.id,
                            "section": section,
                            "source": "chestct_agent_curated_label_registry",
                        },
                    )
                )

        knowledge_dir = Path(self.settings.knowledge_dir)
        if knowledge_dir.exists():
            for path in sorted(knowledge_dir.glob("*.jsonl")):
                documents.extend(self._read_jsonl(path))

        unique: dict[str, RetrievedDocument] = {}
        for document in documents:
            unique[document.doc_id] = document
        return list(unique.values())

    @staticmethod
    def _read_jsonl(path: Path) -> list[RetrievedDocument]:
        documents: list[RetrievedDocument] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            doc_id = str(item.get("doc_id") or f"{path.stem}:{line_number}")
            metadata = dict(item.get("metadata") or {})
            for key in ("label", "source", "url"):
                if item.get(key) and key not in metadata:
                    metadata[key] = item[key]
            documents.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    title=str(item.get("title") or doc_id),
                    text=text,
                    score=0.0,
                    metadata=metadata,
                )
            )
        return documents

    def _collection_name(self) -> str:
        payload = json.dumps(
            {
                "documents": [(doc.doc_id, doc.text) for doc in self.documents],
                "model": str(self.settings.embedding_model_path),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"medical_knowledge_{digest}"

    def _load_embedding_model(self):
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer

            model_path = Path(self.settings.embedding_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Embedding model is missing: {model_path}")
            self.embedding_model = SentenceTransformer(
                str(model_path), device=self.settings.local_rag_device
            )
        return self.embedding_model

    def _load_reranker(self):
        if self.reranker_model is None:
            from sentence_transformers import CrossEncoder

            model_path = Path(self.settings.reranker_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Reranker model is missing: {model_path}")
            self.reranker_model = CrossEncoder(
                str(model_path),
                device=self.settings.local_rag_device,
                max_length=self.settings.rag_reranker_max_length,
            )
        return self.reranker_model

    def _ensure_qdrant(self):
        if self.qdrant_client is not None:
            return self.qdrant_client
        from qdrant_client import QdrantClient, models

        path = Path(self.settings.qdrant_path)
        path.mkdir(parents=True, exist_ok=True)
        self.qdrant_client = QdrantClient(path=str(path))
        if not self.qdrant_client.collection_exists(self.collection_name):
            embedder = self._load_embedding_model()
            vectors = embedder.encode(
                self.corpus,
                prompt="",
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=int(vectors.shape[1]), distance=models.Distance.COSINE
                ),
            )
            points = [
                models.PointStruct(
                    id=index,
                    vector=vector.tolist(),
                    payload={"document_index": index, "doc_id": self.documents[index].doc_id},
                )
                for index, vector in enumerate(vectors)
            ]
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        return self.qdrant_client

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        raw = np.asarray(self.bm25.get_scores(_tokens(query)), dtype=np.float32)
        normalized = _normalize_scores(raw)
        order = np.argsort(normalized)[::-1]
        return [
            (int(index), float(normalized[index]))
            for index in order[:limit]
            if normalized[index] > 0
        ]

    def _dense_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        embedder = self._load_embedding_model()
        query_vector = embedder.encode(
            [query],
            prompt_name="query",
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        client = self._ensure_qdrant()
        result = client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        ).points
        return [
            (int(point.payload["document_index"]), float(point.score))
            for point in result
        ]

    @staticmethod
    def _rrf(
        bm25: list[tuple[int, float]], dense: list[tuple[int, float]], k: int = 60
    ) -> dict[int, float]:
        fused: dict[int, float] = {}
        for ranking in (bm25, dense):
            for rank, (index, _) in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank)
        return fused

    def _rerank(self, query: str, indices: list[int]) -> dict[int, float]:
        model = self._load_reranker()
        raw = np.asarray(
            model.predict(
                [(query, self.corpus[index]) for index in indices],
                batch_size=8,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        ).reshape(-1)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
        return {index: float(score) for index, score in zip(indices, probabilities, strict=True)}

    async def retrieve(
        self, queries: list[str], top_k: int = 5
    ) -> tuple[list[RetrievedDocument], str]:
        query = " ".join(str(item) for item in queries if str(item).strip()).strip()
        if not query or not self.documents:
            return [], "none"

        bm25 = self._bm25_search(query, self.settings.rag_bm25_candidates)
        backend = "bm25"
        dense: list[tuple[int, float]] = []
        reranked: dict[int, float] = {}
        if self.settings.embedding_backend == "hybrid-local":
            try:
                dense = self._dense_search(query, self.settings.rag_dense_candidates)
                fused = self._rrf(bm25, dense)
                candidate_indices = [
                    index
                    for index, _ in sorted(
                        fused.items(), key=lambda item: item[1], reverse=True
                    )[: self.settings.rag_rerank_candidates]
                ]
                reranked = self._rerank(query, candidate_indices)
                backend = "bm25+dense+qwen-reranker+qdrant"
            except Exception as exc:
                backend = f"bm25_degraded:{type(exc).__name__}"

        bm25_scores = dict(bm25)
        dense_scores = dict(dense)
        fused_scores = self._rrf(bm25, dense) if dense else {
            index: score for index, score in bm25
        }
        if reranked:
            ranked_indices = sorted(reranked, key=reranked.get, reverse=True)
        else:
            ranked_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)

        results: list[RetrievedDocument] = []
        for index in ranked_indices[:top_k]:
            final_score = reranked.get(index, fused_scores.get(index, 0.0))
            metadata = dict(self.documents[index].metadata)
            metadata["retrieval_scores"] = {
                "bm25": round(bm25_scores.get(index, 0.0), 6),
                "dense": round(dense_scores.get(index, 0.0), 6),
                "rrf": round(fused_scores.get(index, 0.0), 6),
                "reranker": round(reranked.get(index, 0.0), 6),
            }
            results.append(
                self.documents[index].model_copy(
                    update={"score": round(float(final_score), 6), "metadata": metadata}
                )
            )
        return results, backend


def grade_retrieval(
    documents: list[RetrievedDocument],
    expected_labels: set[str] | None = None,
    min_score: float = 0.08,
) -> bool:
    relevant = [document for document in documents if document.score >= min_score]
    if not relevant:
        return False
    expected = expected_labels or set()
    if not expected:
        return True
    covered = {
        str(document.metadata.get("label"))
        for document in relevant
        if document.metadata.get("label")
    }
    required = 1 if len(expected) <= 2 else max(2, (len(expected) + 1) // 2)
    return len(expected & covered) >= required
