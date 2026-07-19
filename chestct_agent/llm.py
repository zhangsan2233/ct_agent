import json
from dataclasses import dataclass
from typing import Generic, TypeVar
from typing import Any

import httpx

from chestct_agent.config import Settings


T = TypeVar("T")


@dataclass(frozen=True)
class LlmCallResult(Generic[T]):
    value: T
    used_remote: bool
    fallback_reason: str | None = None


class QwenClient:
    """OpenAI-compatible chat client for Qwen endpoints with deterministic fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def is_configured(self) -> bool:
        return (
            self.settings.model_backend == "openai-compatible"
            and self.settings.openai_compatible_api_key
            and self.settings.openai_compatible_api_key != "replace-me"
        )

    def _apply_reasoning_policy(self, payload: dict[str, Any]) -> None:
        effort = self.settings.llm_reasoning_effort.strip().lower()
        if effort == "auto":
            effort = (
                "none"
                if "openrouter.ai" in self.settings.openai_compatible_base_url.lower()
                else ""
            )
        if effort and effort not in {"default", "omit"}:
            payload["reasoning"] = {"effort": effort, "exclude": True}

    @staticmethod
    def _fallback_reason(exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            return f"HTTPStatusError:{exc.response.status_code}"
        message = str(exc)
        if message.startswith("empty_content:"):
            return message
        return type(exc).__name__

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        text = content.strip()
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise ValueError("invalid_json_content")

    async def chat_json(
        self,
        system: str,
        user: str,
        fallback: dict[str, Any],
        max_tokens: int | None = None,
    ) -> LlmCallResult[dict[str, Any]]:
        if not self.is_configured:
            return LlmCallResult(fallback, used_remote=False, fallback_reason="not_configured")

        url = self.settings.openai_compatible_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.agent_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens or self.settings.llm_json_max_tokens,
            "response_format": {"type": "json_object"},
        }
        self._apply_reasoning_policy(payload)
        headers = {"Authorization": f"Bearer {self.settings.openai_compatible_api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            choice = data["choices"][0]
            content = choice["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"empty_content:{choice.get('finish_reason', 'unknown')}")
            return LlmCallResult(self._parse_json_content(content), used_remote=True)
        except Exception as exc:
            return LlmCallResult(
                fallback,
                used_remote=False,
                fallback_reason=self._fallback_reason(exc),
            )

    async def chat_text(
        self,
        system: str,
        user: str,
        fallback: str,
        max_tokens: int | None = None,
    ) -> LlmCallResult[str]:
        if not self.is_configured:
            return LlmCallResult(fallback, used_remote=False, fallback_reason="not_configured")

        url = self.settings.openai_compatible_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.agent_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens or self.settings.llm_text_max_tokens,
        }
        self._apply_reasoning_policy(payload)
        headers = {"Authorization": f"Bearer {self.settings.openai_compatible_api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            choice = data["choices"][0]
            content = choice["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"empty_content:{choice.get('finish_reason', 'unknown')}")
            return LlmCallResult(content, used_remote=True)
        except Exception as exc:
            return LlmCallResult(
                fallback,
                used_remote=False,
                fallback_reason=self._fallback_reason(exc),
            )
