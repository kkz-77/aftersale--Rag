"""DeepSeek chat-completion provider."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config.settings import Settings
from app.core.exceptions import LLMProviderError
from app.core.retry import is_retryable_status, with_exponential_backoff
from app.providers.base import LLMResponse
from app.providers.call_trace import current_call_trace
from app.schemas.common import TokenUsage

logger = logging.getLogger(__name__)


class DeepSeekProvider:
    """OpenAI-compatible DeepSeek chat completions client.

    Timeouts are split so a hung connect does not block a worker forever.
    400/401/403 are not retried.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.deepseek_base_url.rstrip("/"),
            timeout=httpx.Timeout(
                connect=3.0,
                read=settings.llm_timeout_seconds,
                write=10.0,
                pool=5.0,
            ),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def ping(self) -> bool:
        """Do not call chat completions here — readiness must stay cheap."""
        return bool(self._settings.deepseek_api_key.get_secret_value())

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        api_key = self._settings.deepseek_api_key.get_secret_value()
        if not api_key:
            raise LLMProviderError("DeepSeek API key is not configured")

        payload: dict[str, object] = {
            "model": self._settings.deepseek_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        # httpx 直接发送 JSON：qwen-flash 用 enable_thinking=false 关闭思考。
        # 必须判断 is not None，否则 False 会被遗漏。
        if self._settings.llm_enable_thinking is not None:
            payload["enable_thinking"] = self._settings.llm_enable_thinking
        # 保留 Omni 的思考强度配置入口；本轮默认不发送此项。
        if self._settings.llm_reasoning_effort is not None:
            payload["reasoning_effort"] = self._settings.llm_reasoning_effort

        async def _call() -> LLMResponse:
            trace = current_call_trace.get()
            attempt: dict[str, Any] = {"usage": None, "response_model": None}
            if trace is not None:
                trace.append(attempt)
            try:
                response = await self._client.post(
                    "/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
            except httpx.TimeoutException as exc:
                raise LLMProviderError("DeepSeek request timed out", timeout=True) from exc
            except httpx.HTTPError as exc:
                raise LLMProviderError("DeepSeek transport error") from exc

            attempt["http_status"] = response.status_code
            if response.status_code in {400, 401, 403, 404, 422}:
                raise LLMProviderError(
                    f"DeepSeek HTTP {response.status_code}",
                    retryable=False,
                )
            if response.status_code == 429:
                raise LLMProviderError("DeepSeek rate limited", rate_limited=True)
            if is_retryable_status(response.status_code):
                raise LLMProviderError(f"DeepSeek HTTP {response.status_code}")
            if response.status_code >= 400:
                raise LLMProviderError(f"DeepSeek HTTP {response.status_code}")

            body = response.json()
            raw_usage = body.get("usage")
            if isinstance(raw_usage, dict):
                attempt["usage"] = {
                    k: raw_usage.get(k)
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
            attempt["response_model"] = body.get("model")
            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMProviderError("DeepSeek response missing choices") from exc
            usage_raw = body.get("usage") or {}
            usage = TokenUsage(
                prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
                completion_tokens=int(usage_raw.get("completion_tokens") or 0),
                total_tokens=int(usage_raw.get("total_tokens") or 0),
            )
            return LLMResponse(
                content=content or "", usage=usage, model=self._settings.deepseek_model
            )

        # return await with_exponential_backoff(
        #     _call,
        #     max_attempts=self._settings.llm_max_retries,
        #     operation_name="deepseek.generate",
        # )
        started = time.perf_counter()

        try:
            return await with_exponential_backoff(
                _call,
                max_attempts=self._settings.llm_max_retries,
                operation_name="deepseek.generate",
            )
        finally:
            logger.warning(
                "LLM generate duration: %.2f seconds",
                time.perf_counter() - started,
            )

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        api_key = self._settings.deepseek_api_key.get_secret_value()
        if not api_key:
            raise LLMProviderError("DeepSeek API key is not configured")

        payload: dict[str, object] = {
            "model": self._settings.deepseek_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        # 流式接口同步使用该配置，避免两种调用方式的思考模式不一致。
        if self._settings.llm_enable_thinking is not None:
            payload["enable_thinking"] = self._settings.llm_enable_thinking
        if self._settings.llm_reasoning_effort is not None:
            payload["reasoning_effort"] = self._settings.llm_reasoning_effort

        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            ) as response:
                if response.status_code in {400, 401, 403, 404, 422}:
                    raise LLMProviderError(
                        f"DeepSeek HTTP {response.status_code}",
                        retryable=False,
                    )
                if response.status_code == 429:
                    raise LLMProviderError("DeepSeek rate limited", rate_limited=True)
                if is_retryable_status(response.status_code):
                    raise LLMProviderError(f"DeepSeek HTTP {response.status_code}")
                if response.status_code >= 400:
                    raise LLMProviderError(f"DeepSeek HTTP {response.status_code}")
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {}).get("content") or ""
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue
                    if delta:
                        yield delta
        except httpx.TimeoutException as exc:
            raise LLMProviderError("DeepSeek stream timed out", timeout=True) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError("DeepSeek stream transport error") from exc
