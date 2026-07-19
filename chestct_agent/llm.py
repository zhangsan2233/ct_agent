import asyncio
import json
import threading
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


class LocalQloraRuntime:
    """Lazy local Qwen3.5 + PEFT adapter runtime.

    This deliberately imports CUDA-heavy dependencies only when the local backend is
    selected and a request actually needs an LLM.  It never calls Hugging Face or any
    other network service: both paths must already exist locally.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._lock = threading.Lock()

    def readiness_error(self) -> str | None:
        if not self.settings.local_llm_model_dir.is_dir():
            return f"local_model_missing:{self.settings.local_llm_model_dir}"
        if not self.settings.local_llm_adapter_dir.is_dir():
            return f"local_adapter_missing:{self.settings.local_llm_adapter_dir}"
        return None

    def _load(self) -> None:
        if self.model is not None:
            return
        error = self.readiness_error()
        if error:
            raise FileNotFoundError(error)

        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        processor = AutoProcessor.from_pretrained(
            self.settings.local_llm_model_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        tokenizer = processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_args: dict[str, Any] = {
            "device_map": "auto" if self.settings.local_llm_device == "auto" else self.settings.local_llm_device,
            "local_files_only": True,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        if self.settings.local_llm_load_in_4bit:
            model_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        base = AutoModelForImageTextToText.from_pretrained(
            self.settings.local_llm_model_dir, **model_args
        )
        self.model = PeftModel.from_pretrained(
            base, self.settings.local_llm_adapter_dir, local_files_only=True
        ).eval()
        self.tokenizer = tokenizer

    def generate(self, system: str, user: str, max_tokens: int) -> str:
        with self._lock:
            self._load()
            assert self.model is not None and self.tokenizer is not None
            import torch

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            return self.tokenizer.decode(
                generated[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )


class QwenClient:
    """OpenAI-compatible chat client for Qwen endpoints with deterministic fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._local_runtime: LocalQloraRuntime | None = None

    @property
    def is_configured(self) -> bool:
        if self.settings.model_backend == "local-qlora":
            return LocalQloraRuntime(self.settings).readiness_error() is None
        return (
            self.settings.model_backend == "openai-compatible"
            and self.settings.openai_compatible_api_key
            and self.settings.openai_compatible_api_key != "replace-me"
        )

    def _local_runtime_or_error(self) -> tuple[LocalQloraRuntime | None, str | None]:
        if self._local_runtime is None:
            self._local_runtime = LocalQloraRuntime(self.settings)
        return self._local_runtime, self._local_runtime.readiness_error()

    async def _local_generate(
        self, system: str, user: str, max_tokens: int
    ) -> tuple[str | None, str | None]:
        runtime, error = self._local_runtime_or_error()
        if error:
            return None, error
        try:
            assert runtime is not None
            return await asyncio.to_thread(runtime.generate, system, user, max_tokens), None
        except Exception as exc:
            return None, f"local_qlora_error:{type(exc).__name__}"

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
        if self.settings.model_backend == "local-qlora":
            content, error = await self._local_generate(
                system, user, max_tokens or self.settings.llm_json_max_tokens
            )
            if error:
                return LlmCallResult(fallback, used_remote=False, fallback_reason=error)
            try:
                assert content is not None
                return LlmCallResult(self._parse_json_content(content), used_remote=True)
            except Exception as exc:
                return LlmCallResult(
                    fallback,
                    used_remote=False,
                    fallback_reason=f"local_qlora_json:{type(exc).__name__}",
                )
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
        if self.settings.model_backend == "local-qlora":
            content, error = await self._local_generate(
                system, user, max_tokens or self.settings.llm_text_max_tokens
            )
            if error:
                return LlmCallResult(fallback, used_remote=False, fallback_reason=error)
            assert content is not None
            return LlmCallResult(content, used_remote=True)
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
