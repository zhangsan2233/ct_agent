import asyncio
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import multiprocessing
import os
import re
import threading
from typing import Any

from chestct_agent.config import Settings
from chestct_agent.labels import LABEL_SPECS
from chestct_agent.schemas import (
    ReportEvidence,
    ReportGraph,
    ReportGraphEdge,
    ReportGraphNode,
)
from chestct_agent.tools.evidence_extractor import extract_evidence, split_sentences


_WORKER_MODEL = None
_WORKER_MODEL_KEY: tuple[str, str, str] | None = None


def _worker_extract(
    report_text: str,
    model_type: str,
    model_cache_dir: str,
    tokenizer_cache_dir: str,
) -> dict[str, Any]:
    global _WORKER_MODEL, _WORKER_MODEL_KEY
    key = (model_type, model_cache_dir, tokenizer_cache_dir)
    if _WORKER_MODEL is None or _WORKER_MODEL_KEY != key:
        from radgraph import RadGraph

        _WORKER_MODEL = RadGraph(
            model_type=model_type,
            cuda=-1,
            model_cache_dir=model_cache_dir,
            tokenizer_cache_dir=tokenizer_cache_dir,
        )
        _WORKER_MODEL_KEY = key
    return _WORKER_MODEL([report_text])


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _normalized_tokens(text: str) -> list[str]:
    return [_singular(item) for item in re.findall(r"[a-z0-9]+", text.lower())]


def _normalized_phrase(text: str) -> str:
    return " ".join(_normalized_tokens(text))


_HEAD_LABELS: dict[str, set[str]] = {}
for _spec in LABEL_SPECS:
    for _term in _spec.terms:
        _tokens = _normalized_tokens(_term)
        if _tokens:
            _HEAD_LABELS.setdefault(_tokens[-1], set()).add(_spec.id)


def _match_canonical_label(node_text: str, contexts: list[str]) -> str | None:
    normalized_contexts = [_normalized_phrase(value) for value in contexts if value]
    node_tokens = _normalized_tokens(node_text)
    if node_tokens and node_tokens[-1] == "opacity" and any(
        any(term in context.split() for term in ("lung", "pulmonary", "lobe"))
        for context in normalized_contexts
    ):
        return "lung_opacity"
    matches: list[tuple[int, str]] = []
    for spec in LABEL_SPECS:
        for term in spec.terms:
            normalized_term = _normalized_phrase(term)
            if normalized_term and any(
                re.search(rf"(?<!\w){re.escape(normalized_term)}(?!\w)", context)
                for context in normalized_contexts
            ):
                matches.append((len(normalized_term.split()), spec.id))
    if matches:
        return max(matches)[1]
    if not node_tokens:
        return None
    head_matches = _HEAD_LABELS.get(node_tokens[-1], set())
    return next(iter(head_matches)) if len(head_matches) == 1 else None


def _sentence_context(report_text: str, entity_text: str) -> tuple[int, str]:
    sentences = split_sentences(report_text)
    entity = _normalized_phrase(entity_text)
    for index, sentence in enumerate(sentences):
        if entity and entity in _normalized_phrase(sentence):
            return index, sentence
    return 0, sentences[0] if sentences else report_text.strip()


def parse_radgraph_annotations(
    annotations: dict[str, Any], report_text: str, model_type: str
) -> ReportGraph:
    annotation = annotations.get("0") or next(iter(annotations.values()), {})
    raw_entities = annotation.get("entities", {})
    nodes: list[ReportGraphNode] = []
    edges: list[ReportGraphEdge] = []
    allowed_relations = {"modify", "located_at", "suggestive_of"}
    for entity_id, entity in raw_entities.items():
        raw_label = str(entity.get("label", "Observation::uncertain"))
        entity_part, _, assertion_part = raw_label.partition("::")
        entity_type = "anatomy" if entity_part.lower() == "anatomy" else "observation"
        assertion = assertion_part.lower().replace(" ", "_")
        if assertion not in {"definitely_present", "definitely_absent", "uncertain"}:
            assertion = "uncertain"
        text = str(entity.get("tokens", "")).strip()
        sentence_index, sentence = _sentence_context(report_text, text)
        nodes.append(
            ReportGraphNode(
                node_id=str(entity_id),
                text=text,
                entity_type=entity_type,
                assertion=assertion,
                start_ix=max(0, int(entity.get("start_ix", 0))),
                end_ix=max(0, int(entity.get("end_ix", 0))),
                sentence_index=sentence_index,
                sentence=sentence,
            )
        )
        for relation in entity.get("relations", []):
            if len(relation) != 2 or relation[0] not in allowed_relations:
                continue
            edges.append(
                ReportGraphEdge(
                    source_id=str(entity_id),
                    target_id=str(relation[1]),
                    relation=relation[0],
                )
            )

    node_by_id = {node.node_id: node for node in nodes}
    for node in nodes:
        if node.entity_type != "observation":
            continue
        contexts = [node.text]
        for edge in edges:
            if (
                edge.source_id == node.node_id
                and edge.relation == "located_at"
                and edge.target_id in node_by_id
            ):
                target = node_by_id[edge.target_id]
                contexts.append(f"{target.text} {node.text}")
            elif (
                edge.target_id == node.node_id
                and edge.relation == "modify"
                and edge.source_id in node_by_id
            ):
                source = node_by_id[edge.source_id]
                contexts.append(f"{source.text} {node.text}")
        node.canonical_label = _match_canonical_label(node.text, contexts)

    valid_ids = set(node_by_id)
    edges = [
        edge
        for edge in edges
        if edge.source_id in valid_ids and edge.target_id in valid_ids
    ]
    return ReportGraph(
        backend=model_type,
        model_type=model_type,
        nodes=nodes,
        edges=edges,
        degraded=False,
    )


def build_rule_fallback_graph(report_text: str, reason: str) -> ReportGraph:
    evidence_by_label = extract_evidence(report_text)
    nodes: list[ReportGraphNode] = []
    seen: set[tuple[str, str, str]] = set()
    assertion_by_polarity = {
        "positive": "definitely_present",
        "negative": "definitely_absent",
        "uncertain": "uncertain",
        "historical": "uncertain",
    }
    for label, items in evidence_by_label.items():
        for evidence in items:
            key = (label, evidence.polarity, evidence.sentence)
            if key in seen:
                continue
            seen.add(key)
            nodes.append(
                ReportGraphNode(
                    node_id=f"rule_{len(nodes) + 1}",
                    text=evidence.matched_term or label.replace("_", " "),
                    entity_type="observation",
                    assertion=assertion_by_polarity[evidence.polarity],
                    sentence_index=evidence.sentence_index,
                    sentence=evidence.sentence,
                    canonical_label=label,
                )
            )
    return ReportGraph(
        backend="rule_fallback",
        model_type="none",
        nodes=nodes,
        edges=[],
        degraded=True,
        warning=reason,
    )


def report_graph_to_evidence(graph: ReportGraph) -> dict[str, list[ReportEvidence]]:
    evidence: dict[str, list[ReportEvidence]] = {}
    polarity_by_assertion = {
        "definitely_present": ("positive", 0.98),
        "definitely_absent": ("negative", 0.99),
        "uncertain": ("uncertain", 0.75),
    }
    if graph.backend == "rule_fallback":
        return evidence
    for node in graph.nodes:
        if node.entity_type != "observation" or not node.canonical_label:
            continue
        polarity, certainty = polarity_by_assertion[node.assertion]
        evidence.setdefault(node.canonical_label, []).append(
            ReportEvidence(
                sentence=node.sentence,
                label=node.canonical_label,
                polarity=polarity,
                certainty=certainty,
                matched_term=node.text,
                sentence_index=node.sentence_index,
                source="radgraph_xl",
            )
        )
    return evidence


class ReportGraphTool:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._executor: ProcessPoolExecutor | None = None
        self._executor_lock = threading.Lock()

    def readiness_error(self) -> str | None:
        if not self.settings.radgraph_enabled:
            return "disabled"
        if importlib.util.find_spec("radgraph") is None:
            return "radgraph package is not installed"
        model_dir = self.settings.radgraph_model_cache_dir / self.settings.radgraph_model_type
        for name in ("config.json", "weights.th"):
            if not (model_dir / name).exists():
                return f"missing {model_dir / name}"
        return None

    def _get_executor(self) -> ProcessPoolExecutor:
        with self._executor_lock:
            if self._executor is None:
                os.environ.setdefault("PYTHONUTF8", "1")
                os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
                os.environ.setdefault(
                    "HF_HOME", str(self.settings.radgraph_tokenizer_cache_dir.resolve())
                )
                self._executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=multiprocessing.get_context("spawn"),
                )
            return self._executor

    async def extract(self, report_text: str) -> ReportGraph:
        if not report_text.strip():
            return ReportGraph(backend="not_used", model_type="none")
        readiness_error = self.readiness_error()
        if readiness_error:
            return build_rule_fallback_graph(
                report_text, f"RadGraph-XL不可用：{readiness_error}。已使用规则图谱降级。"
            )
        loop = asyncio.get_running_loop()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(
                    self._get_executor(),
                    _worker_extract,
                    report_text,
                    self.settings.radgraph_model_type,
                    str(self.settings.radgraph_model_cache_dir.resolve()),
                    str(self.settings.radgraph_tokenizer_cache_dir.resolve()),
                ),
                timeout=self.settings.radgraph_timeout_seconds,
            )
            return parse_radgraph_annotations(
                raw, report_text, self.settings.radgraph_model_type
            )
        except Exception as exc:
            return build_rule_fallback_graph(
                report_text,
                f"RadGraph-XL推理失败（{type(exc).__name__}），已使用规则图谱降级。",
            )

    def close(self) -> None:
        with self._executor_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
