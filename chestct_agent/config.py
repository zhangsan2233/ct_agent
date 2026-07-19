from functools import lru_cache
import os
from pathlib import Path
import sys

from pydantic import BaseModel, Field


class Settings(BaseModel):
    agent_model: str = "Qwen/Qwen3.6-35B-A3B"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_backend: str = "hybrid-local"
    embedding_model_path: Path = Path("./models/qwen/Qwen3-Embedding-0.6B")
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_model_path: Path = Path("./models/qwen/Qwen3-Reranker-0.6B")
    local_rag_device: str = "cpu"
    rag_dense_candidates: int = 30
    rag_bm25_candidates: int = 30
    rag_rerank_candidates: int = 20
    rag_reranker_max_length: int = Field(default=256, ge=64, le=2048)
    qdrant_path: Path = Path("./artifacts/qdrant")
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_timeout_seconds: float = 10.0
    model_backend: str = "openai-compatible"
    openai_compatible_base_url: str = "http://localhost:8000/v1"
    openai_compatible_api_key: str = "replace-me"
    local_llm_model_dir: Path = Path("./models/qwen3_5_9B/Qwen3.5-9B")
    local_llm_adapter_dir: Path = Path(
        "./artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep/adapter"
    )
    local_llm_device: str = "auto"
    local_llm_load_in_4bit: bool = True
    request_timeout_seconds: float = 45.0
    llm_json_max_tokens: int = Field(default=1024, ge=128, le=8192)
    llm_text_max_tokens: int = Field(default=512, ge=64, le=4096)
    llm_reasoning_effort: str = "auto"

    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    artifact_dir: Path = Path("./artifacts")
    static_dir: Path = Path("./static")
    ct_model_backend: str = "ct-clip"
    ctclip_variant: str = "zeroshot"
    ctclip_checkpoint: Path = Path("./models/ctclip/CT-CLIP_v2.pt")
    ctclip_source_dir: Path = Path("./external/CT-CLIP-main")
    ctclip_python: Path = Path(sys.executable)
    ctclip_device: str = "auto"
    ctclip_use_fp16: bool = True
    ctclip_timeout_seconds: float = 600.0
    ct_cache_enabled: bool = True
    ct_max_positive_labels: int = Field(default=11, ge=1, le=18)
    knowledge_dir: Path = Path("./data/knowledge")
    calibration_path: Path = Path("./artifacts/calibration/calibrators.joblib")
    radgraph_enabled: bool = True
    radgraph_model_type: str = "modern-radgraph-xl"
    radgraph_model_cache_dir: Path = Path("./models/radgraph")
    radgraph_tokenizer_cache_dir: Path = Path("./models/huggingface")
    radgraph_timeout_seconds: float = 180.0
    radgenome_mask_dir: Path = Path("./data/radgenome")
    radgenome_index_path: Path = Path("./artifacts/radgenome/mask_index.csv")
    radgenome_max_masks: int = Field(default=8, ge=1, le=200)
    memory_db_path: Path = Path("./artifacts/memory/agent_memory.sqlite3")
    agent_dynamic_planning: bool = True
    tool_max_retries: int = Field(default=1, ge=0, le=3)
    top_k_similar: int = 5
    rag_max_attempts: int = Field(default=2, ge=1, le=5)
    min_label_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    positive_label_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    strong_negative_threshold: float = Field(default=0.15, ge=0.0, le=1.0)

    disclaimer: str = "仅用于课程设计和科研演示，不作为临床诊断依据。"


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _setting_value(name: str, default):
    env_name = name.upper()
    raw = os.environ.get(env_name, _read_env_file(Path(".env")).get(env_name))
    if raw is None or raw == "":
        return default
    if isinstance(default, Path):
        return Path(raw)
    if isinstance(default, bool):
        return str(raw).lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@lru_cache
def get_settings() -> Settings:
    defaults = Settings()
    return Settings(
        agent_model=_setting_value("agent_model", defaults.agent_model),
        embedding_model=_setting_value("embedding_model", defaults.embedding_model),
        embedding_backend=_setting_value("embedding_backend", defaults.embedding_backend),
        embedding_model_path=_setting_value(
            "embedding_model_path", defaults.embedding_model_path
        ),
        reranker_model=_setting_value("reranker_model", defaults.reranker_model),
        reranker_model_path=_setting_value(
            "reranker_model_path", defaults.reranker_model_path
        ),
        local_rag_device=_setting_value("local_rag_device", defaults.local_rag_device),
        rag_dense_candidates=_setting_value(
            "rag_dense_candidates", defaults.rag_dense_candidates
        ),
        rag_bm25_candidates=_setting_value(
            "rag_bm25_candidates", defaults.rag_bm25_candidates
        ),
        rag_rerank_candidates=_setting_value(
            "rag_rerank_candidates", defaults.rag_rerank_candidates
        ),
        rag_reranker_max_length=_setting_value(
            "rag_reranker_max_length", defaults.rag_reranker_max_length
        ),
        qdrant_path=_setting_value("qdrant_path", defaults.qdrant_path),
        embedding_base_url=_setting_value("embedding_base_url", defaults.embedding_base_url),
        embedding_api_key=_setting_value("embedding_api_key", defaults.embedding_api_key),
        embedding_timeout_seconds=_setting_value(
            "embedding_timeout_seconds", defaults.embedding_timeout_seconds
        ),
        model_backend=_setting_value("model_backend", defaults.model_backend),
        openai_compatible_base_url=_setting_value(
            "openai_compatible_base_url", defaults.openai_compatible_base_url
        ),
        openai_compatible_api_key=_setting_value(
            "openai_compatible_api_key", defaults.openai_compatible_api_key
        ),
        local_llm_model_dir=_setting_value(
            "local_llm_model_dir", defaults.local_llm_model_dir
        ),
        local_llm_adapter_dir=_setting_value(
            "local_llm_adapter_dir", defaults.local_llm_adapter_dir
        ),
        local_llm_device=_setting_value("local_llm_device", defaults.local_llm_device),
        local_llm_load_in_4bit=_setting_value(
            "local_llm_load_in_4bit", defaults.local_llm_load_in_4bit
        ),
        request_timeout_seconds=_setting_value(
            "request_timeout_seconds", defaults.request_timeout_seconds
        ),
        llm_json_max_tokens=_setting_value(
            "llm_json_max_tokens", defaults.llm_json_max_tokens
        ),
        llm_text_max_tokens=_setting_value(
            "llm_text_max_tokens", defaults.llm_text_max_tokens
        ),
        llm_reasoning_effort=_setting_value(
            "llm_reasoning_effort", defaults.llm_reasoning_effort
        ),
        data_dir=_setting_value("data_dir", defaults.data_dir),
        upload_dir=_setting_value("upload_dir", defaults.upload_dir),
        artifact_dir=_setting_value("artifact_dir", defaults.artifact_dir),
        static_dir=_setting_value("static_dir", defaults.static_dir),
        ct_model_backend=_setting_value("ct_model_backend", defaults.ct_model_backend),
        ctclip_variant=_setting_value("ctclip_variant", defaults.ctclip_variant),
        ctclip_checkpoint=_setting_value("ctclip_checkpoint", defaults.ctclip_checkpoint),
        ctclip_source_dir=_setting_value("ctclip_source_dir", defaults.ctclip_source_dir),
        ctclip_python=_setting_value("ctclip_python", defaults.ctclip_python),
        ctclip_device=_setting_value("ctclip_device", defaults.ctclip_device),
        ctclip_use_fp16=_setting_value("ctclip_use_fp16", defaults.ctclip_use_fp16),
        ctclip_timeout_seconds=_setting_value(
            "ctclip_timeout_seconds", defaults.ctclip_timeout_seconds
        ),
        ct_cache_enabled=_setting_value("ct_cache_enabled", defaults.ct_cache_enabled),
        ct_max_positive_labels=_setting_value(
            "ct_max_positive_labels", defaults.ct_max_positive_labels
        ),
        knowledge_dir=_setting_value("knowledge_dir", defaults.knowledge_dir),
        calibration_path=_setting_value("calibration_path", defaults.calibration_path),
        radgraph_enabled=_setting_value("radgraph_enabled", defaults.radgraph_enabled),
        radgraph_model_type=_setting_value(
            "radgraph_model_type", defaults.radgraph_model_type
        ),
        radgraph_model_cache_dir=_setting_value(
            "radgraph_model_cache_dir", defaults.radgraph_model_cache_dir
        ),
        radgraph_tokenizer_cache_dir=_setting_value(
            "radgraph_tokenizer_cache_dir", defaults.radgraph_tokenizer_cache_dir
        ),
        radgraph_timeout_seconds=_setting_value(
            "radgraph_timeout_seconds", defaults.radgraph_timeout_seconds
        ),
        radgenome_mask_dir=_setting_value(
            "radgenome_mask_dir", defaults.radgenome_mask_dir
        ),
        radgenome_index_path=_setting_value(
            "radgenome_index_path", defaults.radgenome_index_path
        ),
        radgenome_max_masks=_setting_value(
            "radgenome_max_masks", defaults.radgenome_max_masks
        ),
        memory_db_path=_setting_value("memory_db_path", defaults.memory_db_path),
        agent_dynamic_planning=_setting_value(
            "agent_dynamic_planning", defaults.agent_dynamic_planning
        ),
        tool_max_retries=_setting_value("tool_max_retries", defaults.tool_max_retries),
        top_k_similar=_setting_value("top_k_similar", defaults.top_k_similar),
        rag_max_attempts=_setting_value("rag_max_attempts", defaults.rag_max_attempts),
        min_label_confidence=_setting_value(
            "min_label_confidence", defaults.min_label_confidence
        ),
        positive_label_threshold=_setting_value(
            "positive_label_threshold", defaults.positive_label_threshold
        ),
        strong_negative_threshold=_setting_value(
            "strong_negative_threshold", defaults.strong_negative_threshold
        ),
        disclaimer=_setting_value("disclaimer", defaults.disclaimer),
    )
