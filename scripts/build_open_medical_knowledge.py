import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import sys
import time

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.labels import LABEL_IDS


EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
KNOWLEDGE_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "medical_material": (
        "central venous catheter",
        "pacemaker lead",
        "cardiac implantable device",
    ),
    "arterial_wall_calcification": (
        "thoracic aortic calcification",
        "aortic wall calcification",
    ),
    "cardiomegaly": ("cardiomegaly", "cardiac enlargement"),
    "pericardial_effusion": ("pericardial effusion",),
    "coronary_artery_wall_calcification": ("coronary artery calcification",),
    "hiatal_hernia": ("hiatal hernia",),
    "lymphadenopathy": ("mediastinal lymphadenopathy", "hilar lymphadenopathy"),
    "emphysema": ("pulmonary emphysema",),
    "atelectasis": ("pulmonary atelectasis", "lung atelectasis"),
    "pulmonary_nodule": ("pulmonary nodule", "lung nodule"),
    "lung_opacity": ("lung opacity", "ground glass opacity"),
    "pulmonary_fibrotic_sequela": ("pulmonary fibrosis", "fibrotic lung change"),
    "pleural_effusion": ("pleural effusion",),
    "mosaic_attenuation_pattern": ("mosaic attenuation",),
    "peribronchial_thickening": ("peribronchial thickening", "bronchial wall thickening"),
    "consolidation": ("pulmonary consolidation", "lung consolidation"),
    "bronchiectasis": ("bronchiectasis",),
    "interlobular_septal_thickening": ("interlobular septal thickening",),
}
EXCLUDED_CONTENT_MARKERS = (
    "dog",
    "dogs",
    "canine",
    "veterinary",
    "mouse model",
    "mice",
    "rat model",
    "lithium-ion",
    "battery",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a traceable chest CT knowledge corpus from Europe PMC Open Access metadata."
    )
    parser.add_argument(
        "--output",
        default="data/knowledge/europe_pmc_chest_ct.jsonl",
    )
    parser.add_argument("--per-label", type=int, default=3)
    parser.add_argument(
        "--labels",
        nargs="*",
        default=list(LABEL_IDS),
        help="Canonical label IDs. Defaults to all 18 labels.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep", type=float, default=0.15)
    return parser.parse_args()


def _clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def _query_for_label(label: str) -> str:
    terms = KNOWLEDGE_QUERY_TERMS[label]
    condition = " OR ".join(f'TITLE_ABS:"{term}"' for term in terms)
    modality = (
        'TITLE_ABS:"chest CT" OR TITLE_ABS:"computed tomography" '
        "OR TITLE_ABS:thoracic"
    )
    return f"OPEN_ACCESS:Y AND HAS_ABSTRACT:Y AND ({condition}) AND ({modality})"


def _relevance_score(label: str, item: dict[str, object]) -> float:
    title = _clean_text(item.get("title")).lower()
    abstract = _clean_text(item.get("abstractText")).lower()
    score = 0.0
    for term in KNOWLEDGE_QUERY_TERMS[label]:
        normalized = term.lower()
        if normalized in title:
            score += 5.0
        elif normalized in abstract:
            score += 2.0
    if any(term in title for term in ("ct", "computed tomography", "thoracic", "chest")):
        score += 2.0
    publication_types = item.get("pubTypeList") or {}
    publication_text = json.dumps(publication_types).lower()
    if "review" in publication_text or "guideline" in publication_text:
        score += 1.5
    return score


def _stable_article_id(item: dict[str, object]) -> tuple[str, str]:
    for key in ("pmcid", "pmid", "doi", "id"):
        value = _clean_text(item.get(key))
        if value:
            return key, value
    return "unknown", str(abs(hash(json.dumps(item, sort_keys=True, default=str))))


def _article_url(item: dict[str, object]) -> str:
    pmcid = _clean_text(item.get("pmcid"))
    if pmcid:
        return f"https://europepmc.org/articles/{pmcid}"
    pmid = _clean_text(item.get("pmid"))
    if pmid:
        return f"https://europepmc.org/article/MED/{pmid}"
    doi = _clean_text(item.get("doi"))
    return f"https://doi.org/{doi}" if doi else "https://europepmc.org/"


def _fetch_label(
    client: httpx.Client, label: str, per_label: int
) -> list[dict[str, object]]:
    response = client.get(
        EUROPE_PMC_SEARCH_URL,
        params={
            "query": _query_for_label(label),
            "format": "json",
            "resultType": "core",
            "pageSize": max(per_label * 10, 30),
        },
    )
    response.raise_for_status()
    results = response.json().get("resultList", {}).get("result", [])
    documents: list[dict[str, object]] = []
    retrieved_at = datetime.now(timezone.utc).isoformat()
    results = sorted(results, key=lambda item: _relevance_score(label, item), reverse=True)
    for item in results:
        abstract = _clean_text(item.get("abstractText"))
        title = _clean_text(item.get("title"))
        relevance = _relevance_score(label, item)
        combined = f"{title} {abstract}".lower()
        if (
            not abstract
            or not title
            or relevance < 2.0
            or any(marker in combined for marker in EXCLUDED_CONTENT_MARKERS)
        ):
            continue
        id_type, article_id = _stable_article_id(item)
        url = _article_url(item)
        documents.append(
            {
                "doc_id": f"europe_pmc:{id_type}:{article_id}:{label}",
                "title": title,
                "text": abstract,
                "label": label,
                "source": "Europe PMC Open Access",
                "url": url,
                "metadata": {
                    "label": label,
                    "source": "Europe PMC Open Access",
                    "source_type": "open_access_literature",
                    "url": url,
                    "pmid": _clean_text(item.get("pmid")),
                    "pmcid": _clean_text(item.get("pmcid")),
                    "doi": _clean_text(item.get("doi")),
                    "journal": _clean_text(item.get("journalTitle")),
                    "publication_year": _clean_text(item.get("pubYear")),
                    "license": _clean_text(item.get("license")) or "open_access",
                    "retrieved_at": retrieved_at,
                    "query": _query_for_label(label),
                    "relevance_score": relevance,
                },
            }
        )
        if len(documents) >= per_label:
            break
    return documents


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.labels) - set(LABEL_IDS))
    if unknown:
        raise ValueError(f"Unknown canonical labels: {unknown}")
    if args.per_label < 1 or args.per_label > 25:
        raise ValueError("--per-label must be between 1 and 25")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, object]] = []
    headers = {"User-Agent": "ChestCT-Agent/0.1 (academic research knowledge builder)"}
    with httpx.Client(timeout=args.timeout, headers=headers, follow_redirects=True) as client:
        for index, label in enumerate(args.labels, start=1):
            rows = _fetch_label(client, label, args.per_label)
            documents.extend(rows)
            print(f"[{index}/{len(args.labels)}] {label}: {len(rows)} documents", flush=True)
            time.sleep(max(0.0, args.sleep))

    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(f"Wrote {len(documents)} traceable documents to {output}")


if __name__ == "__main__":
    main()
