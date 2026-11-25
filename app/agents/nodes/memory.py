"""After-answer rolling summary; only committed complete turns can advance the cursor."""

import logging
from typing import Any

from app.agents.state import AgentState
from app.config.settings import Settings
from app.memory.budget import SUMMARY_MAX_TOKENS, messages_tokens, summary_message_tokens
from app.memory.compression import group_dialogue_turns, persisted_complete_prefix
from app.memory.summarizer import ConversationSummarizer
from app.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)


async def memory_node(
    state: AgentState,
    *,
    settings: Settings,
    summarizer: ConversationSummarizer,
) -> dict:
    cursor = state.get("summary_upto_message_id") or 0
    messages = list(state.get("messages") or [])
    messages.append(ChatMessage(role="user", content=state["user_query"]))
    final_answer = state.get("final_answer")
    if final_answer:
        messages.append(ChatMessage(role="assistant", content=final_answer))
    summary = state.get("summary") or ""
    before = messages_tokens(messages)
    stats: dict[str, Any] = {
        "trigger": "none",
        "calls": 0,
        "elapsed_ms": 0.0,
        "before_tokens_estimated": before,
        "after_tokens_estimated": before,
        "covered_ids": [],
        "summary_usage": {},
        "remaining_over_target": before > settings.memory_summary_target_tokens,
    }
    update = {
        "messages": messages,
        "summary": summary,
        "summary_updated": False,
        "summary_upto_message_id": cursor,
        "memory_stats": stats,
    }
    if not (settings.memory_in_generation or settings.memory_in_decisions):
        return update
    if before <= settings.memory_summary_trigger_tokens:
        return update
    stats["trigger"] = "token_pressure"

    eligible = persisted_complete_prefix(group_dialogue_turns(messages), after_id=cursor)
    stats["pending_turns"] = len(eligible)
    if not eligible:
        stats["error"] = "no_persisted_complete_prefix"
        return update

    batch: list[ChatMessage] = []
    remaining_tokens = before
    for turn in eligible:
        proposed = batch + turn
        if summarizer.input_tokens(proposed, summary) > settings.memory_summary_input_tokens:
            break
        batch = proposed
        remaining_tokens -= messages_tokens(turn)
        if remaining_tokens <= settings.memory_summary_target_tokens:
            break
    if not batch:
        stats["error"] = "complete_turn_exceeds_summary_input_budget"
        logger.warning(
            "rolling_summary_input_budget_exceeded", extra={"session_id": state.get("session_id")}
        )
        return update
    stats["candidate_ids"] = [m.id for m in batch]
    stats["summary_input_tokens_estimated"] = summarizer.input_tokens(batch, summary)
    outcome = await summarizer.summarize(
        batch,
        previous_summary=summary,
        timeout_seconds=settings.memory_summary_timeout_seconds,
        max_tokens=settings.memory_summary_max_tokens,
    )
    stats.update(
        calls=outcome.calls,
        elapsed_ms=outcome.elapsed_ms,
        summary_usage=outcome.usage.model_dump(),
        error=outcome.error,
    )
    if outcome.summary is not None and summary_message_tokens(outcome.summary) > SUMMARY_MAX_TOKENS:
        outcome.summary = None
        stats["error"] = "summary_exceeds_injection_budget"
    if outcome.summary is not None:
        new_cursor = batch[-1].id
        assert new_cursor is not None
        # 移出的只是成功摘要覆盖的消息；本轮无ID消息和未处理缓冲仍保留。
        remaining = [m for m in messages if m.id is None or m.id > new_cursor]
        update.update(
            messages=remaining,
            summary=outcome.summary,
            summary_updated=True,
            summary_upto_message_id=new_cursor,
        )
        stats["covered_ids"] = [m.id for m in batch]
        stats["after_tokens_estimated"] = messages_tokens(remaining)
        stats["remaining_over_target"] = (
            stats["after_tokens_estimated"] > settings.memory_summary_target_tokens
        )
    log = logger.info if outcome.summary is not None else logger.warning
    log(
        "rolling_summary_completed" if outcome.summary is not None else "rolling_summary_failed",
        extra={
            "session_id": state.get("session_id"),
            "memory_stats": stats,
        },
    )
    return update
