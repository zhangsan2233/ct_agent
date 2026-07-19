# ChestCT-Agent

ChestCT-Agent is a controlled Agentic RAG prototype for chest CT evidence integration.
It uses Qwen/Qwen3.6-35B-A3B as the agent model through an OpenAI-compatible endpoint,
LangGraph for workflow orchestration, CT-RATE for data, and deterministic fallbacks so
the project can run before the real dataset and remote model service are configured.

## What This Implements

- FastAPI backend with JSON `/api/analyze`, multipart `/api/analyze/upload`, and NDJSON
  `/api/analyze/upload/stream` endpoints.
- LangGraph workflow with Qwen tool planning, an allow-list policy, bounded retrieval rewrite,
  retry/degradation handling, typed output validation, and a human approval gate.
- A canonical 18-label contract across report classification, CT-CLIP, calibration, fusion,
  evidence, evaluation, and Chinese output. Report evidence retains positive, negative,
  uncertain, and historical polarity.
- Hybrid medical retrieval using BM25 + local Qwen3-Embedding-0.6B + Qdrant +
  Qwen3-Reranker-0.6B, plus CT-RATE similar-case retrieval.
- RadGenome organ and lesion masks with real slice/bbox/mask output. `lung nodule` and
  `lung effusion` are treated as lesion masks; other masks remain explicitly anatomy-level.
- SQLite audit memory, FastMCP tools, FastAPI approval endpoints, Streamlit, and Docker Compose.
- CT-RATE download and preparation scripts.
- Streamlit interface for patient/doctor uploads, with dataset cases isolated in developer mode.
- NIfTI upload validation and safe DICOM ZIP ingestion through SimpleITK. Uploaded files are
  stored under a deidentified content hash rather than their original filename.
- Evaluation scripts for classification, route completeness, evidence coverage, retrieval,
  latency, cache use, and safety checks.

## Model Download Policy

For exact training-flow reconstruction after authorized CT-RATE access, see [reproducibility/REBUILD.md](reproducibility/REBUILD.md). The public reproducibility package includes only the 500-case ID/path/weak-label manifest, the 450/50 split, seeds, parameters, and SHA256 values; it contains no CT images, report text, or CT-CLIP prediction payloads.

完整的模型、adapter、CT-RATE 数据下载来源、落盘路径、受限资产说明和离线验证方法见
[`docs/ASSET_SETUP.md`](docs/ASSET_SETUP.md)。仓库中实际包含和明确排除的资产清单见
[`docs/REPOSITORY_CONTENTS.md`](docs/REPOSITORY_CONTENTS.md)。

Do not download Qwen/Qwen3.6-35B-A3B to a normal laptop by default. Use an OpenAI-compatible
remote endpoint, vLLM/SGLang server, cloud model service, or a hosted endpoint.

Local downloads that are reasonable:

- `Qwen/Qwen3-Embedding-0.6B` and `Qwen/Qwen3-Reranker-0.6B`.
- A quantized Qwen3.5 4B/9B model for offline debugging.

### Local Stage-2 QLoRA backend

The final project-trained adapter can drive the Agent without an OpenAI-compatible
endpoint. Install the optional local inference dependencies, put the already downloaded
Qwen3.5-9B base model and the repository Stage-2 adapter at the configured paths, then set:

```text
MODEL_BACKEND=local-qlora
LOCAL_LLM_MODEL_DIR=./models/qwen3_5_9B/Qwen3.5-9B
LOCAL_LLM_ADAPTER_DIR=./artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep/adapter
LOCAL_LLM_DEVICE=auto
LOCAL_LLM_LOAD_IN_4BIT=true
```

Install the local QLoRA dependencies with `pip install -r requirements-llm-train.txt` in the
GPU environment.

The model is loaded lazily on its first Agent request, uses local files only, and never
downloads weights at runtime. If the base model or adapter is absent, the Agent reports an
explicit local-asset fallback rather than silently calling a remote model.

## Data Download Policy

CT-RATE is gated and large. Start with:

- `dataset/radiology_text_reports`
- `dataset/multi_abnormality_labels`
- `dataset/metadata`

Then download only a small CT volume subset for image rendering and CT model experiments.

Download an inclusive validation patient range with a gated Hugging Face token from `.env`.
The command is resumable and skips files already present in the local Hugging Face snapshot:

```powershell
python scripts/download_ct_rate_valid_range.py --start 1 --end 50 --dry-run
python scripts/download_ct_rate_valid_range.py --start 1 --end 50 --max-workers 2
python scripts/download_ct_rate_valid_range.py --start 51 --end 80 `
  --one-reconstruction --max-download-gb 8 --max-workers 2
```

## CT Models And Deployment Decision

The deployed CT-only tool uses the official 3D CT-CLIP architecture and `CT-CLIP_v2.pt`
checkpoint. It evaluates all 18 CT-RATE abnormalities with paired positive/negative prompts.
Positive output uses patient-level per-label thresholds fitted for at least 0.60 precision on
the calibration split; the uncertain boundary targets at least 0.40 precision. This
changes the task from high-recall all-label screening to selective key-finding output.

`CT_LiPro_v2.pt` is also implemented and evaluated. It is retained as an ablation rather than
silently presented as an upgrade: on the local 143-volume cohort its CT-only raw macro-AUPRC
was 0.402 versus 0.447 for zero-shot CT-CLIP. The checkpoint can be selected with
`CTCLIP_VARIANT=lipro`, but it is not the deployed default.

On this workstation CT inference runs in an isolated CUDA worker process, so each uncached
request reloads the model.
If dependencies or weights are missing, the Agent continues with report-only fusion and
returns an explicit warning.

Successful volume-level probabilities and rendered previews are cached by input/model
fingerprint. Repeating the same case does not reload CT-CLIP or decode the compressed volume.

Install the CUDA dependencies and download the official source and gated checkpoint:

```powershell
cd chestct-agent
powershell -ExecutionPolicy Bypass -File scripts\install_ctclip.ps1
python scripts/download_ctclip_assets.py --weights-only --variant lipro
```

The large assets are downloaded to ignored paths:

```text
external/CT-CLIP-main
models/ctclip/CT-CLIP_v2.pt
models/ctclip/CT_LiPro_v2.pt
```

The CT model accepts any valid three-dimensional NIfTI volume. The web interface also accepts
a ZIP containing one or more DICOM series, selects the largest series, and converts it to
compressed NIfTI. Preprocessing uses the
official target geometry `(240, 480, 480)`, HU clipping `[-1000, 1000]`, and target spacing
`(1.5, 0.75, 0.75)` mm. RTX 4060 Laptop inference is attempted with FP16 and batch size 1;
GPU memory failure is reported rather than silently replaced with fabricated scores.

CT-CLIP can run in a separate CUDA Python environment; the main FastAPI environment calls
it as an isolated Tool worker. Set `CTCLIP_PYTHON` to that environment's Python executable.

## Key Local Results

The current leakage-free split contains 48 calibration patients and 32 independent test
patients, with zero patient overlap. References are CT-RATE report-derived weak labels.

| Mode | Micro-P | Micro-R | Micro-F1 | Macro-F1 | Macro-AUROC | Macro-AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CT-only selective output | 0.545 | 0.313 | 0.398 | 0.273 | 0.731 | 0.483 |
| Report-only | 0.925 | 0.965 | 0.945 | 0.896 | 0.993 | 0.944 |
| CT + report fusion | 0.859 | 0.957 | 0.905 | 0.894 | 0.986 | 0.941 |

These results support the project claim that controlled multimodal integration is stronger
than the current CT classifier alone. They do not establish clinical performance: the test
cohort is small, labels are weak, and there is no external hospital validation.

## Setup

```powershell
cd chestct-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Set `.env` to your Qwen endpoint:

```text
MODEL_BACKEND=openai-compatible
OPENAI_COMPATIBLE_BASE_URL=https://your-qwen-endpoint/v1
OPENAI_COMPATIBLE_API_KEY=...
AGENT_MODEL=Qwen/Qwen3.6-35B-A3B
```

The default RAG backend is `hybrid-local`. It uses the two local Qwen 0.6B models and a
persistent Qdrant index. If dense retrieval or reranking fails, the request continues with
BM25 and records a `bm25_degraded:<error>` backend instead of fabricating retrieval scores.

```text
EMBEDDING_BACKEND=hybrid-local
EMBEDDING_MODEL_PATH=./models/qwen/Qwen3-Embedding-0.6B
RERANKER_MODEL_PATH=./models/qwen/Qwen3-Reranker-0.6B
LOCAL_RAG_DEVICE=cpu
```

Additional medical knowledge can be added as JSONL files under `data/knowledge`. Each line
accepts `doc_id`, `title`, `text`, `label`, `source`, and `url`. Only curated, traceable
sources should be used for reported experiments.

Build the small Europe PMC Open Access chest CT corpus used by the demo:

```powershell
python scripts/build_open_medical_knowledge.py --per-label 3
```

The generated JSONL contains abstracts plus PMID/PMCID, DOI, publication year, license,
retrieval time, and the original URL. It is supporting literature, not a substitute for a
versioned clinical guideline library.

## Run API

```powershell
uvicorn chestct_agent.api.main:app --reload --port 8080
```

Example request:

```powershell
Invoke-RestMethod -Method Post http://localhost:8080/api/analyze `
  -ContentType "application/json" `
  -Body '{"case_id":"demo","report_text":"Small bilateral pleural effusions are present. No pneumothorax.","question":"What abnormalities are present?"}'
```

For normal patient/doctor input, open the Streamlit interface or send multipart form data to
`POST /api/analyze/upload`. Accepted CT files are `.nii`, `.nii.gz`, and DICOM `.zip`; reports
can be pasted text or UTF-8/GB18030 `.txt`. CT-RATE selection is hidden behind the sidebar
developer toggle and is not the default product input.

The local UI uses `http://127.0.0.1:8080` by default. Set `CHESTCT_API_URL` only when the API is
served at another address.

The Streamlit interface consumes the streaming endpoint and displays both the start and completion
of every node while the Agent is running. Every final response includes `agent_plan`,
`execution_events`, and `rag_trace`. These fields expose
the selected tools and reasons, per-node status/latency/retries, query rewrites, retrieval
backend, BM25/Dense/RRF/Reranker scores, retrieved evidence, and sufficiency decisions.

Report inputs also run through the official Stanford AIMI `modern-radgraph-xl` model. The
`report_graph` response field contains Anatomy/Observation nodes, certainty/negation assertions,
and `modify`, `located_at`, and `suggestive_of` edges. RadGraph runs in an isolated persistent CPU
worker and contributes typed evidence to consistency checks. If its package or weights are absent,
the response uses `backend=rule_fallback` and `degraded=true` instead of presenting rule extraction
as model output.

Set a `session_id` on the first analysis to enable case-bound multi-turn conversation. Follow-up
questions use the saved structured result and dialogue history, then dynamically choose case
context, medical RAG, or similar-case context without rerunning the 3D CT model:

```powershell
Invoke-RestMethod -Method Post http://localhost:8080/api/chat `
  -ContentType "application/json" `
  -Body '{"session_id":"session-1","case_id":"demo","message":"胸腔积液有什么报告证据？"}'
```

`POST /api/chat/stream` returns NDJSON start/completion events plus the final answer. Conversation
history is stored in SQLite and can be inspected at
`GET /api/sessions/{session_id}/cases/{case_id}/messages`.

## Run Demo

```powershell
streamlit run demo/streamlit_app.py
```

## Download CT-RATE Metadata Files

After Hugging Face access is approved:

```powershell
python scripts/download_ct_rate.py --repo-id ibrahimhamamci/CT-RATE --data-dir data --metadata-only
```

For gated access, either log in once:

```powershell
.venv\Scripts\huggingface-cli.exe login
```

or put a read token in `.env`:

```text
HF_TOKEN=hf_...
```

Do not paste tokens into chat or commit them.

## Prepare Dataset

```powershell
python scripts/prepare_dataset.py --data-dir data --out-dir artifacts/prepared --top-labels 0
python scripts/build_similar_case_index.py
```

The case index contains CT-RATE **training** reports only. Similar-case retrieval supports
report-text, CT-predicted-condition, and grounded-region queries. It excludes the current
patient and deduplicates returned reconstruction variants by patient. CT-RATE weak labels are
used for candidate reranking and display only; they are not treated as ground truth for the
current input.

Evaluate report-to-case retrieval without feeding validation labels into the retriever:

```powershell
python scripts/evaluate_similar_case_retrieval.py --limit 100 --top-k 5
```

## Multimodal Evaluation

Build a manifest from the CT volumes that are currently available locally:

```powershell
python scripts/build_multimodal_manifest.py
```

Run the four ablation modes. Batch inference supports `--resume`, `--limit`, and `--no-llm`:

```powershell
python scripts/run_agent_batch.py --mode report_only --no-llm
python scripts/run_agent_batch.py --mode ct_only --no-llm
python scripts/run_agent_batch.py --mode multimodal --no-llm
```

For CT-CLIP evaluation on many volumes, keep one model instance resident on the GPU and
write each completed result to the same cache used by the Agent:

```powershell
D:\path\to\cuda-python.exe scripts/ctclip_worker.py `
  --manifest artifacts/evaluation/multimodal_manifest.csv `
  --checkpoint models/ctclip/CT-CLIP_v2.pt `
  --source-dir external/CT-CLIP-main `
  --variant zeroshot --device cuda --fp16 --resume

python scripts/evaluate_ctclip.py
python scripts/build_patient_splits.py
python scripts/evaluate_ablation.py --ct-threshold-method precision `
  --ct-target-precision 0.6 --ct-uncertain-target-precision 0.4
python scripts/evaluate_retrieval.py
python scripts/evaluate_report_evidence.py
python scripts/evaluate_agent_capabilities.py --use-llm
python scripts/summarize_experiments.py
```

Evaluate predictions against the dataset-provided validation labels:

```powershell
python scripts/evaluate_agent.py `
  --predictions artifacts/evaluation/multimodal_predictions.jsonl `
  --ground-truth data/dataset/multi_abnormality_labels/valid_predicted_labels.csv `
  --uncertain-as-positive `
  --out artifacts/evaluation/multimodal_metrics.json
```

The CT-RATE files are named `*_predicted_labels.csv`; report these as dataset-provided weak
labels rather than radiologist gold-standard annotations. AUROC/AUPRC from one or only a few
cases are not meaningful. Use at least 30-50 local validation volumes for a first comparison.

## RadGenome Grounding

After downloading the validation region/anatomy archives, extract only masks matching local
CT cases, then build an index. Empty or invalid masks are skipped and reported.

```powershell
python scripts/extract_radgenome_subset.py `
  --archive data/radgenome/archives/valid_region_mask.tar.gz
python scripts/extract_radgenome_subset.py `
  --archive data/radgenome/archives/valid_anatomy_mask.tar.gz
python scripts/index_radgenome_masks.py
```

RadGenome volumes use a processed grid. Unless shape and affine both match, outputs use
`normalized_index_resample` and set `alignment_verified=false`.

## MCP And Docker

Run the MCP server over stdio:

```powershell
python -m chestct_agent.mcp_server
```

Validate and start the container stack:

```powershell
docker compose config --quiet
docker compose up --build
```

See `docs/PPT_IMPLEMENTATION_STATUS.md` for a PPT-by-PPT implementation audit and the
explicit rationale for alternatives that were not duplicated.

## Safety Boundary

This project is for coursework and research demonstration only. It is not a clinical
diagnostic system. Default output includes: `仅用于课程设计和科研演示，不作为临床诊断依据。`
