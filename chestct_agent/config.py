from functools import lru_cache
import os
from pathlib import Path
import sys

from pydantic import BaseModel, Field


class Settings(BaseModel):
    agent_model: str = "Qwen/Qwen3.6-35B-A3B"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    model_backend: str = "openai-compatible"
    openai_compatible_base_url: str = "http://localhost:8000/v1"
    openai_compatible_api_key: str = "replace-me"
    request_timeout_seconds: float = 45.0

    data_dir: Path = Path("./data")
    artifact_dir: Path = Path("./artifacts")
    static_dir: Path = Path("./static")
    ct_model_backend: str = "ct-clip"
    ctclip_checkpoint: Path = Path("./models/ctclip/CT-CLIP_v2.pt")
    ctclip_source_dir: Path = Path("./external/CT-CLIP-main")
    ctclip_python: Path = Path(sys.executable)
    ctclip_device: str = "auto"
    ctclip_use_fp16: bool = True
    ctclip_timeout_seconds: float = 600.0
    top_k_similar: int = 5
    min_label_confidence: float = Field(default=0.35, ge=0.0, le=1.0)

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
        model_backend=_setting_value("model_backend", defaults.model_backend),
        openai_compatible_base_url=_setting_value(
            "openai_compatible_base_url", defaults.openai_compatible_base_url
        ),
        openai_compatible_api_key=_setting_value(
            "openai_compatible_api_key", defaults.openai_compatible_api_key
        ),
        request_timeout_seconds=_setting_value(
            "request_timeout_seconds", defaults.request_timeout_seconds
        ),
        data_dir=_setting_value("data_dir", defaults.data_dir),
        artifact_dir=_setting_value("artifact_dir", defaults.artifact_dir),
        static_dir=_setting_value("static_dir", defaults.static_dir),
        ct_model_backend=_setting_value("ct_model_backend", defaults.ct_model_backend),
        ctclip_checkpoint=_setting_value("ctclip_checkpoint", defaults.ctclip_checkpoint),
        ctclip_source_dir=_setting_value("ctclip_source_dir", defaults.ctclip_source_dir),
        ctclip_python=_setting_value("ctclip_python", defaults.ctclip_python),
        ctclip_device=_setting_value("ctclip_device", defaults.ctclip_device),
        ctclip_use_fp16=_setting_value("ctclip_use_fp16", defaults.ctclip_use_fp16),
        ctclip_timeout_seconds=_setting_value(
            "ctclip_timeout_seconds", defaults.ctclip_timeout_seconds
        ),
        top_k_similar=_setting_value("top_k_similar", defaults.top_k_similar),
        min_label_confidence=_setting_value(
            "min_label_confidence", defaults.min_label_confidence
        ),
        disclaimer=_setting_value("disclaimer", defaults.disclaimer),
    )
