import asyncio

from chestct_agent.config import Settings
from chestct_agent.llm import QwenClient


def test_parse_json_content_accepts_plain_object():
    assert QwenClient._parse_json_content('{"answer": 1}') == {"answer": 1}


def test_parse_json_content_accepts_fenced_or_prefixed_object():
    content = 'Result:\n```json\n{"answer": 2}\n```\nDone.'

    assert QwenClient._parse_json_content(content) == {"answer": 2}


def test_local_qlora_missing_assets_uses_deterministic_fallback(tmp_path):
    client = QwenClient(
        Settings(
            model_backend="local-qlora",
            local_llm_model_dir=tmp_path / "missing-model",
            local_llm_adapter_dir=tmp_path / "missing-adapter",
        )
    )

    result = asyncio.run(
        client.chat_json("system", "user", fallback={"safe": True}, max_tokens=128)
    )

    assert result.value == {"safe": True}
    assert result.used_remote is False
    assert result.fallback_reason == f"local_model_missing:{tmp_path / 'missing-model'}"


def test_local_qlora_is_configured_when_base_and_adapter_directories_exist(tmp_path):
    model_dir = tmp_path / "Qwen3.5-9B"
    adapter_dir = tmp_path / "adapter"
    model_dir.mkdir()
    adapter_dir.mkdir()

    client = QwenClient(
        Settings(
            model_backend="local-qlora",
            local_llm_model_dir=model_dir,
            local_llm_adapter_dir=adapter_dir,
        )
    )

    assert client.is_configured is True
