"""Dialogue grouping plus legacy progressive compression helpers.

Rolling memory uses complete-turn grouping below. The legacy tool truncation
helpers remain available, but the after-answer node no longer truncates tool
outputs or removes history before it has been summarized successfully.

50–70% budget: truncate bulky tool payloads.
70–85%: shrink older turns to bullets.
>85%: auto-compact — keep last K turns + a running summary.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.core.enums import CompressionLevel
from app.memory.budget import decide_level, estimate_tokens, messages_tokens
from app.schemas.agent import ToolResult
from app.schemas.chat import ChatMessage

KEEP_RECENT_TURNS = 4
TOOL_SOFT_CHARS = 400
TOOL_HARD_CHARS = 160


def group_dialogue_turns(messages: list[ChatMessage]) -> list[list[ChatMessage]]:
    """按用户发起分组，窗口/摘要共用，不删除不完整的用户轮次。"""
    turns: list[list[ChatMessage]] = []
    for message in messages:
        if message.role == "user":
            turns.append([message])
        elif message.role == "assistant":
            if turns:
                turns[-1].append(message)
            else:
                turns.append([message])  # 异常孤立消息也保留，摘要不能跨过它推进。
        else:
            turns.append([message])
    return turns


def persisted_complete_prefix(
    turns: list[list[ChatMessage]], *, after_id: int = 0
) -> list[list[ChatMessage]]:
    """仅取连续完整的已保存前缀，遇到缺失ID或不完整问答即停止。"""
    complete: list[list[ChatMessage]] = []
    for turn in turns:
        if (
            len(turn) != 2
            or turn[0].role != "user"
            or turn[1].role != "assistant"
            or turn[0].id is None
            or turn[1].id is None
            or not after_id < turn[0].id < turn[1].id
        ):
            break
        complete.append(turn)
        after_id = turn[1].id
    return complete


def compress_tool_results(
    results: list[ToolResult],
    level: CompressionLevel,
) -> list[ToolResult]:
    limit = {
        CompressionLevel.NONE: 10_000,
        CompressionLevel.SOFT_TRUNCATE: TOOL_SOFT_CHARS,
        CompressionLevel.HARD_COMPRESS: TOOL_HARD_CHARS,
        CompressionLevel.AUTO_COMPACT: TOOL_HARD_CHARS,
    }[level]
    compressed: list[ToolResult] = []
    for result in results:
        cloned = result.model_copy(deep=True)
        text = str(cloned.output)
        if len(text) > limit:
            cloned.output = {"truncated": True, "preview": text[:limit]}
            cloned.truncated = True
        compressed.append(cloned)
    return compressed


def sliding_window(
    messages: list[ChatMessage], *, keep: int = KEEP_RECENT_TURNS
) -> list[ChatMessage]:
    if len(messages) <= keep * 2:
        return list(messages)
    return list(messages[-(keep * 2) :])


def apply_compression(
    messages: list[ChatMessage],
    tool_results: list[ToolResult],
    summary: str,
    settings: Settings,
) -> tuple[list[ChatMessage], list[ToolResult], CompressionLevel, str]:
    used = messages_tokens(messages) + estimate_tokens(summary)
    level = decide_level(used, settings)
    tools = compress_tool_results(tool_results, level)
    if level in {CompressionLevel.HARD_COMPRESS, CompressionLevel.AUTO_COMPACT}:
        window = sliding_window(messages)
        if summary:
            window = [ChatMessage(role="system", content=f"对话摘要：{summary}")] + window
        return window, tools, level, summary
    if level == CompressionLevel.SOFT_TRUNCATE:
        return messages, tools, level, summary
    return messages, tools, level, summary
