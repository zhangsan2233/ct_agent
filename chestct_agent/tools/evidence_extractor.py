import re

from chestct_agent.knowledge import HISTORY_TERMS, NEGATION_TERMS, UNCERTAIN_TERMS
from chestct_agent.labels import LABEL_BY_ID, LABEL_IDS
from chestct_agent.schemas import ReportEvidence


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = rf"(?<!\w){re.escape(phrase.lower())}(?!\w)"
    return re.search(pattern, text.lower()) is not None


def _clause_for_term(sentence: str, term: str) -> str:
    clauses = re.split(r"\s*(?:;|\bbut\b|\bhowever\b|\balthough\b)\s*", sentence, flags=re.I)
    return next((clause for clause in clauses if _contains_phrase(clause, term)), sentence)


def _polarity(clause: str) -> tuple[str, float]:
    if any(_contains_phrase(clause, term) for term in UNCERTAIN_TERMS):
        return "uncertain", 0.65
    if any(_contains_phrase(clause, term) for term in NEGATION_TERMS):
        return "negative", 0.98
    if any(_contains_phrase(clause, term) for term in HISTORY_TERMS):
        return "historical", 0.75
    return "positive", 0.95


def extract_evidence(
    report_text: str,
    labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, list[ReportEvidence]]:
    selected = tuple(labels or LABEL_IDS)
    sentences = split_sentences(report_text)
    evidence: dict[str, list[ReportEvidence]] = {label: [] for label in selected}
    for sentence_index, sentence in enumerate(sentences):
        for label in selected:
            spec = LABEL_BY_ID.get(label)
            terms = spec.terms if spec else (label.replace("_", " "),)
            for term in terms:
                if not _contains_phrase(sentence, term):
                    continue
                clause = _clause_for_term(sentence, term)
                polarity, certainty = _polarity(clause)
                evidence[label].append(
                    ReportEvidence(
                        sentence=sentence,
                        label=label,
                        polarity=polarity,
                        certainty=certainty,
                        matched_term=term,
                        sentence_index=sentence_index,
                    )
                )
                break
    return evidence
