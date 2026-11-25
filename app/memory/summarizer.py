"""Incremental summaries with explicit failure; never replace good data with snippets."""

import asyncio
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, StrictStr

from app.memory.budget import message_tokens
from app.prompts.summary import SUMMARY_SYSTEM, SUMMARY_USER
from app.providers.base import LLMProvider
from app.providers.fake import OfflineLLMProvider
from app.providers.json_parser import parse_model
from app.schemas.chat import ChatMessage
from app.schemas.common import TokenUsage


class SummaryPayload(BaseModel):
    summary: StrictStr = Field(min_length=1, max_length=4000)


@dataclass
class SummaryOutcome:
    summary: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    elapsed_ms: float = 0.0
    calls: int = 0
    error: str = ""


class ConversationSummarizer:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @staticmethod
    def input_messages(messages: list[ChatMessage], previous_summary: str) -> list[dict[str, str]]:
        # 不截掉候选头尾；游标由调用方根据实际输入ID决定，不交给模型猜测。
        history = "\n".join(f"[{m.id}] {m.role}: {m.content}" for m in messages)
        return [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": SUMMARY_USER.format(
                    previous_summary=previous_summary or "(无)",
                    history=history,
                ),
            },
        ]

    @classmethod
    def input_tokens(cls, messages: list[ChatMessage], previous_summary: str) -> int:
        return sum(
            message_tokens(m["content"]) for m in cls.input_messages(messages, previous_summary)
        )

    async def summarize(
        self,
        messages: list[ChatMessage],
        *,
        previous_summary: str = "",
        timeout_seconds: float = 15.0,
        max_tokens: int = 512,
    ) -> SummaryOutcome:
        outcome = SummaryOutcome()
        if isinstance(self._llm, OfflineLLMProvider):
            outcome.error = "offline_model"
            return outcome
        started = time.perf_counter()
        try:
            outcome.calls = 1  # 逻辑调用次数，不包含Provider内部HTTP重试。
            async with asyncio.timeout(timeout_seconds):
                response = await self._llm.generate(
                    self.input_messages(messages, previous_summary),
                    temperature=0.0,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            outcome.usage = response.usage
            if outcome.usage.completion_tokens > max_tokens:
                outcome.error = "summary_exceeds_model_output_budget"
                return outcome
            text = parse_model(response.content, SummaryPayload).summary.strip()
            if not text:
                raise ValueError("empty summary")
            outcome.summary = text
        except Exception as exc:
            # 摘要辅助失败不打断回答；CancelledError不在Exception内，仍向上传播。
            outcome.error = type(exc).__name__
        finally:
            outcome.elapsed_ms = (time.perf_counter() - started) * 1000
        return outcome
