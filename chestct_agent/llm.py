import json
from typing import Any

import httpx

from chestct_agent.config import Settings


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

    async def chat_json(self, system: str, user: str, fallback: dict[str, Any]) -> dict[str, Any]:
        if not self.is_configured:
            return fallback

        url = self.settings.openai_compatible_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.agent_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.settings.openai_compatible_api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            return fallback

    async def chat_text(self, system: str, user: str, fallback: str) -> str:
        if not self.is_configured:
            return fallback

        url = self.settings.openai_compatible_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.agent_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.settings.openai_compatible_api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except Exception:
            return fallback

