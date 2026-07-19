import argparse
import asyncio
import json
import math
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.config import Settings
from chestct_agent.labels import LABEL_SPECS
from chestct_agent.tools.rag import MedicalRagTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BM25 and hybrid medical retrieval.")
    parser.add_argument("--out", default="artifacts/evaluation/rag_metrics.json")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def _queries() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for spec in LABEL_SPECS:
        rows.extend(
            [
                (spec.id, spec.title),
                (spec.id, f"CT imaging findings of {spec.title}"),
            ]
        )
    return rows


async def _evaluate(tool: MedicalRagTool, rows: list[tuple[str, str]], top_k: int):
    recall = 0
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    latencies: list[float] = []
    backends: dict[str, int] = {}
    failures: list[dict[str, object]] = []
    for expected_label, query in rows:
        started = time.perf_counter()
        docs, backend = await tool.retrieve([query], top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)
        backends[backend] = backends.get(backend, 0) + 1
        ranked_labels = [str(doc.metadata.get("label", "")) for doc in docs]
        ranks = [index + 1 for index, label in enumerate(ranked_labels) if label == expected_label]
        if ranks:
            recall += 1
            reciprocal_ranks.append(1.0 / ranks[0])
            dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks)
            ideal_hits = min(3, len(ranks))
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcgs.append(dcg / idcg if idcg else 0.0)
        else:
            reciprocal_ranks.append(0.0)
            ndcgs.append(0.0)
            failures.append(
                {"query": query, "expected_label": expected_label, "retrieved": ranked_labels}
            )
    ordered = sorted(latencies)
    return {
        "queries": len(rows),
        f"recall@{top_k}": recall / len(rows),
        "mrr": sum(reciprocal_ranks) / len(rows),
        f"ndcg@{top_k}": sum(ndcgs) / len(rows),
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p95": ordered[int(0.95 * (len(ordered) - 1))],
        },
        "backends": backends,
        "failures": failures,
    }


async def main() -> None:
    args = parse_args()
    rows = _queries()
    common = {
        "embedding_model_path": Path("models/qwen/Qwen3-Embedding-0.6B"),
        "reranker_model_path": Path("models/qwen/Qwen3-Reranker-0.6B"),
        "qdrant_path": Path("artifacts/qdrant"),
        "local_rag_device": "cpu",
        "rag_dense_candidates": 10,
        "rag_bm25_candidates": 10,
        "rag_rerank_candidates": 4,
    }
    baseline = await _evaluate(
        MedicalRagTool(Settings(embedding_backend="bm25", **common)), rows, args.top_k
    )
    hybrid = await _evaluate(
        MedicalRagTool(Settings(embedding_backend="hybrid-local", **common)), rows, args.top_k
    )
    result = {
        "dataset": "36 deterministic label-title/imaging queries",
        "baseline_bm25": baseline,
        "hybrid": hybrid,
        "ndcg_relative_improvement": (
            (hybrid[f"ndcg@{args.top_k}"] - baseline[f"ndcg@{args.top_k}"])
            / max(baseline[f"ndcg@{args.top_k}"], 1e-12)
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "hybrid"}, indent=2))
    print("hybrid", {key: value for key, value in hybrid.items() if key != "failures"})
    print(f"Wrote {output}")


if __name__ == "__main__":
    asyncio.run(main())
