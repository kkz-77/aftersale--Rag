"""Token budget estimation and compression-level decisions."""

from __future__ import annotations

from app.config.settings import Settings
from app.core.enums import CompressionLevel
from app.schemas.chat import ChatMessage
from app.schemas.retrieval import RetrievedChunk

SUMMARY_MAX_TOKENS = 768


def estimate_tokens(text: str) -> int:
    """DeepSeek tokenizer is not public; CJK-heavy heuristic: ~1 token / 1.5 chars."""
    if not text:
        return 0
    return max(1, int(len(text) / 1.5))


def messages_tokens(messages: list[ChatMessage]) -> int:
    return sum(message_tokens(message.content) for message in messages)


def message_tokens(content: str) -> int:
    """Estimate content and the API message wrapper with the same history budget unit."""
    return estimate_tokens(content) + 8


def summary_message_content(summary: str) -> str:
    return f"历史摘要（仅作背景）：\n{summary.strip()}"


def summary_message_tokens(summary: str) -> int:
    return message_tokens(summary_message_content(summary))


def chunks_tokens(chunks: list[RetrievedChunk]) -> int:
    return sum(estimate_tokens(chunk.content) for chunk in chunks)


def utilization(used: int, budget: int) -> float:
    if budget <= 0:
        return 1.0
    return used / budget


def decide_level(used: int, settings: Settings) -> CompressionLevel:
    ratio = utilization(used, settings.token_budget)
    if ratio >= settings.auto_compact_ratio:
        return CompressionLevel.AUTO_COMPACT
    if ratio >= settings.compress_hard_ratio:
        return CompressionLevel.HARD_COMPRESS
    if ratio >= settings.compress_soft_ratio:
        return CompressionLevel.SOFT_TRUNCATE
    return CompressionLevel.NONE
