import re

from chestct_agent.knowledge import LABEL_KNOWLEDGE


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def extract_evidence(report_text: str, labels: list[str]) -> dict[str, list[str]]:
    sentences = split_sentences(report_text)
    evidence: dict[str, list[str]] = {}
    for label in labels:
        terms = LABEL_KNOWLEDGE.get(label, {}).get("terms", [label.replace("_", " ")])
        hits: list[str] = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(term.lower() in lower for term in terms):
                hits.append(sentence)
        evidence[label] = hits[:3]
    return evidence

