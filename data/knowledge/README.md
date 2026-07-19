# Medical knowledge corpus

Add curated documents as UTF-8 JSONL files. One document per line:

Build a small traceable chest CT corpus from the Europe PMC Open Access index:

```powershell
python scripts/build_open_medical_knowledge.py --per-label 3
```

This stores abstracts and source metadata only. It does not scrape paywalled article full text.

```json
{"doc_id":"source:pleural_effusion","title":"Pleural effusion","text":"...","label":"pleural_effusion","source":"Source name","url":"https://example.org/document"}
```

Required field: `text`. Recommended fields: `doc_id`, `title`, `label`, `source`, and `url`.
For literature, also keep `metadata.pmid`, `metadata.pmcid`, `metadata.doi`,
`metadata.license`, `metadata.publication_year`, and `metadata.retrieved_at`.
Keep source text concise and record redistribution permission before sharing the corpus.
