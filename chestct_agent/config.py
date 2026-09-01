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
    vision_openai_compatible_base_url: str = ""
    vision_openai_compatible_api_key: str = ""
    local_llm_model_dir: Path = Path("./models/Qwen3.5-9B")
    local_llm_adapter_dir: Path = Path(
        "./artifacts/llm_qlora/qwen3_5_9b_ctclip_stage2_500_2ep/adapter"
    )
    local_llm_device: str = "auto"
    local_llm_load_in_4bit: bool = True
    request_timeout_seconds: float = 45.0
    llm_json_max_tokens: int = Field(default=1024, ge=128, le=8192)
    llm_text_max_tokens: int = Field(default=512, ge=64, le=4096)
    llm_reasoning_effort: str = "auto"
    qwen_vision_enabled: bool = True
    qwen_vision_max_images: int = Field(default=7, ge=3, le=12)
    qwen_vision_min_confidence: float = Field(default=0.85, ge=0.5, le=1.0)
    qwen_grounding_enabled: bool = True
    qwen_grounding_alpha: float = Field(default=0.68, ge=0.1, le=0.9)
    slice_vlm_enabled: bool = True
    slice_vlm_model: str = "google/gemma-4-31b-it"
    slice_vlm_max_images: int = Field(default=4, ge=3, le=18)

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
    ctclip_service_enabled: bool = False
    ctclip_service_url: str = "http://127.0.0.1:8090"
    ctclip_service_api_key: str = "local-ctclip"
    ct_cache_enabled: bool = True
    ct_max_positive_labels: int = Field(default=11, ge=1, le=18)
    ct_attribution_enabled: bool = True
    ct_attribution_slices_per_label: int = Field(default=3, ge=1, le=5)
    ct_attribution_alpha: float = Field(default=0.65, ge=0.1, le=0.9)
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
    totalseg_enabled: bool = True
    totalseg_executable: Path = Path("./tools/TotalSegmentator")
    totalseg_device: str = "gpu"
    totalseg_timeout_seconds: float = 900.0
    totalseg_cache_dir: Path = Path("./artifacts/tool_cache/totalsegmentator")
    pleural_effusion_positive_ml: float = Field(default=15.0, ge=0.0)
    pleural_effusion_uncertain_ml: float = Field(default=3.0, ge=0.0)
    pericardial_effusion_positive_ml: float = Field(default=20.0, ge=0.0)
    pericardial_effusion_uncertain_ml: float = Field(default=3.0, ge=0.0)
    anatomy_quantification_fusion_enabled: bool = False
    cardiothoracic_ratio_positive: float = Field(default=0.5, ge=0.2, le=0.9)
    memory_db_path: Path = Path("./artifacts/memory/agent_memory.sqlite3")
    experience_memory_enabled: bool = False
    experience_memory_experiment_id: str = "patchchestct-feedback-memory-v1"
    experience_memory_fold: int = Field(default=-1, ge=-1)
    experience_memory_max_items: int = Field(default=9, ge=1, le=36)
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
        embedding_model_path=_setting_value("embedding_model_path", defaults.embedding_model_path),
        reranker_model=_setting_value("reranker_model", defaults.reranker_model),
        reranker_model_path=_setting_value("reranker_model_path", defaults.reranker_model_path),
        local_rag_device=_setting_value("local_rag_device", defaults.local_rag_device),
        rag_dense_candidates=_setting_value("rag_dense_candidates", defaults.rag_dense_candidates),
        rag_bm25_candidates=_setting_value("rag_bm25_candidates", defaults.rag_bm25_candidates),
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
        vision_openai_compatible_base_url=_setting_value(
            "vision_openai_compatible_base_url",
            defaults.vision_openai_compatible_base_url,
        ),
        vision_openai_compatible_api_key=_setting_value(
            "vision_openai_compatible_api_key",
            defaults.vision_openai_compatible_api_key,
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
        llm_json_max_tokens=_setting_value("llm_json_max_tokens", defaults.llm_json_max_tokens),
        llm_text_max_tokens=_setting_value("llm_text_max_tokens", defaults.llm_text_max_tokens),
        llm_reasoning_effort=_setting_value("llm_reasoning_effort", defaults.llm_reasoning_effort),
        qwen_vision_enabled=_setting_value(
            "qwen_vision_enabled", defaults.qwen_vision_enabled
        ),
        qwen_vision_max_images=_setting_value(
            "qwen_vision_max_images", defaults.qwen_vision_max_images
        ),
        qwen_vision_min_confidence=_setting_value(
            "qwen_vision_min_confidence", defaults.qwen_vision_min_confidence
        ),
        qwen_grounding_enabled=_setting_value(
            "qwen_grounding_enabled", defaults.qwen_grounding_enabled
        ),
        qwen_grounding_alpha=_setting_value(
            "qwen_grounding_alpha", defaults.qwen_grounding_alpha
        ),
        slice_vlm_enabled=_setting_value("slice_vlm_enabled", defaults.slice_vlm_enabled),
        slice_vlm_model=_setting_value("slice_vlm_model", defaults.slice_vlm_model),
        slice_vlm_max_images=_setting_value(
            "slice_vlm_max_images", defaults.slice_vlm_max_images
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
        ctclip_service_enabled=_setting_value(
            "ctclip_service_enabled", defaults.ctclip_service_enabled
        ),
        ctclip_service_url=_setting_value(
            "ctclip_service_url", defaults.ctclip_service_url
        ),
        ctclip_service_api_key=_setting_value(
            "ctclip_service_api_key", defaults.ctclip_service_api_key
        ),
        ct_cache_enabled=_setting_value("ct_cache_enabled", defaults.ct_cache_enabled),
        ct_max_positive_labels=_setting_value(
            "ct_max_positive_labels", defaults.ct_max_positive_labels
        ),
        ct_attribution_enabled=_setting_value(
            "ct_attribution_enabled", defaults.ct_attribution_enabled
        ),
        ct_attribution_slices_per_label=_setting_value(
            "ct_attribution_slices_per_label",
            defaults.ct_attribution_slices_per_label,
        ),
        ct_attribution_alpha=_setting_value("ct_attribution_alpha", defaults.ct_attribution_alpha),
        knowledge_dir=_setting_value("knowledge_dir", defaults.knowledge_dir),
        calibration_path=_setting_value("calibration_path", defaults.calibration_path),
        radgraph_enabled=_setting_value("radgraph_enabled", defaults.radgraph_enabled),
        radgraph_model_type=_setting_value("radgraph_model_type", defaults.radgraph_model_type),
        radgraph_model_cache_dir=_setting_value(
            "radgraph_model_cache_dir", defaults.radgraph_model_cache_dir
        ),
        radgraph_tokenizer_cache_dir=_setting_value(
            "radgraph_tokenizer_cache_dir", defaults.radgraph_tokenizer_cache_dir
        ),
        radgraph_timeout_seconds=_setting_value(
            "radgraph_timeout_seconds", defaults.radgraph_timeout_seconds
        ),
        radgenome_mask_dir=_setting_value("radgenome_mask_dir", defaults.radgenome_mask_dir),
        radgenome_index_path=_setting_value("radgenome_index_path", defaults.radgenome_index_path),
        radgenome_max_masks=_setting_value("radgenome_max_masks", defaults.radgenome_max_masks),
        totalseg_enabled=_setting_value("totalseg_enabled", defaults.totalseg_enabled),
        totalseg_executable=_setting_value(
            "totalseg_executable", defaults.totalseg_executable
        ),
        totalseg_device=_setting_value("totalseg_device", defaults.totalseg_device),
        totalseg_timeout_seconds=_setting_value(
            "totalseg_timeout_seconds", defaults.totalseg_timeout_seconds
        ),
        totalseg_cache_dir=_setting_value(
            "totalseg_cache_dir", defaults.totalseg_cache_dir
        ),
        pleural_effusion_positive_ml=_setting_value(
            "pleural_effusion_positive_ml", defaults.pleural_effusion_positive_ml
        ),
        pleural_effusion_uncertain_ml=_setting_value(
            "pleural_effusion_uncertain_ml", defaults.pleural_effusion_uncertain_ml
        ),
        pericardial_effusion_positive_ml=_setting_value(
            "pericardial_effusion_positive_ml", defaults.pericardial_effusion_positive_ml
        ),
        pericardial_effusion_uncertain_ml=_setting_value(
            "pericardial_effusion_uncertain_ml", defaults.pericardial_effusion_uncertain_ml
        ),
        anatomy_quantification_fusion_enabled=_setting_value(
            "anatomy_quantification_fusion_enabled",
            defaults.anatomy_quantification_fusion_enabled,
        ),
        cardiothoracic_ratio_positive=_setting_value(
            "cardiothoracic_ratio_positive", defaults.cardiothoracic_ratio_positive
        ),
        memory_db_path=_setting_value("memory_db_path", defaults.memory_db_path),
        experience_memory_enabled=_setting_value(
            "experience_memory_enabled", defaults.experience_memory_enabled
        ),
        experience_memory_experiment_id=_setting_value(
            "experience_memory_experiment_id",
            defaults.experience_memory_experiment_id,
        ),
        experience_memory_fold=_setting_value(
            "experience_memory_fold", defaults.experience_memory_fold
        ),
        experience_memory_max_items=_setting_value(
            "experience_memory_max_items", defaults.experience_memory_max_items
        ),
        agent_dynamic_planning=_setting_value(
            "agent_dynamic_planning", defaults.agent_dynamic_planning
        ),
        tool_max_retries=_setting_value("tool_max_retries", defaults.tool_max_retries),
        top_k_similar=_setting_value("top_k_similar", defaults.top_k_similar),
        rag_max_attempts=_setting_value("rag_max_attempts", defaults.rag_max_attempts),
        min_label_confidence=_setting_value("min_label_confidence", defaults.min_label_confidence),
        positive_label_threshold=_setting_value(
            "positive_label_threshold", defaults.positive_label_threshold
        ),
        strong_negative_threshold=_setting_value(
            "strong_negative_threshold", defaults.strong_negative_threshold
        ),
        disclaimer=_setting_value("disclaimer", defaults.disclaimer),
    )
