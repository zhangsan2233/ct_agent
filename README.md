# ChestCT-Agent

ChestCT-Agent is a controlled Agentic RAG prototype for chest CT evidence integration.
It uses Qwen/Qwen3.6-35B-A3B as the agent model through an OpenAI-compatible endpoint,
LangGraph for workflow orchestration, CT-RATE for data, and deterministic fallbacks so
the project can run before the real dataset and remote model service are configured.

## What This Implements

- FastAPI backend with `/api/analyze`.
- LangGraph workflow:
  `parse_input -> parse_report -> run_text_classifier -> run_ct_classifier -> plan_rag_queries -> retrieve_medical_knowledge -> retrieve_similar_cases -> grade_retrieval -> rewrite_query_if_needed -> extract_evidence -> check_consistency -> generate_json -> validate_output -> generate_chinese_explanation`.
- Tool modules for report parsing, text classification, CT preprocessing, CT classification,
  RAG, similar case retrieval, visual evidence, consistency checking, and JSON validation.
- CT-RATE download and preparation scripts.
- Streamlit demo.
- Evaluation script for JSON validity, tool traces, latency, retrieval overlap, and safety checks.

## Model Download Policy

Do not download Qwen/Qwen3.6-35B-A3B to a normal laptop by default. Use an OpenAI-compatible
remote endpoint, vLLM/SGLang server, cloud model service, or a hosted endpoint.

Local downloads that are reasonable:

- `Qwen/Qwen3-Embedding-0.6B` or another small embedding model.
- A quantized Qwen3.5 4B/9B model for offline debugging.

## Data Download Policy

CT-RATE is gated and large. Start with:

- `dataset/radiology_text_reports`
- `dataset/multi_abnormality_labels`
- `dataset/metadata`

Then download only a small CT volume subset for image rendering and CT model experiments.

## CT-CLIP

The CT tool uses the official 3D CT-CLIP architecture and `CT-CLIP_v2.pt` checkpoint.
It evaluates all 18 CT-RATE abnormalities with paired positive/negative prompts. On this
workstation it runs in an isolated CUDA worker process, so each request reloads the model.
If dependencies or weights are missing, the Agent continues with report-only fusion and
returns an explicit warning.

Install the CUDA dependencies and download the official source and gated checkpoint:

```powershell
cd chestct-agent
powershell -ExecutionPolicy Bypass -File scripts\install_ctclip.ps1
```

The large assets are downloaded to ignored paths:

```text
external/CT-CLIP-main
models/ctclip/CT-CLIP_v2.pt
```

The expected CT input is a CT-RATE v2 `*_fixed/*.nii.gz` volume. Preprocessing uses the
official target geometry `(240, 480, 480)`, HU clipping `[-1000, 1000]`, and target spacing
`(1.5, 0.75, 0.75)` mm. RTX 4060 Laptop inference is attempted with FP16 and batch size 1;
GPU memory failure is reported rather than silently replaced with fabricated scores.

CT-CLIP can run in a separate CUDA Python environment; the main FastAPI environment calls
it as an isolated Tool worker. Set `CTCLIP_PYTHON` to that environment's Python executable.

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
python scripts/prepare_dataset.py --data-dir data --out-dir artifacts/prepared --top-labels 8
```

## Evaluate Agent Outputs

```powershell
python scripts/evaluate_agent.py --predictions artifacts/predictions.jsonl
```

## Safety Boundary

This project is for coursework and research demonstration only. It is not a clinical
diagnostic system. Default output includes: `仅用于课程设计和科研演示，不作为临床诊断依据。`
