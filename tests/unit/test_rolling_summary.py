"""Mock-only rolling summary tests: message IDs, trigger, failures, budgets."""

import asyncio
import json
import logging

import pytest

from app.agents.nodes.memory import memory_node
from app.config.settings import Settings
from app.core.exceptions import LLMProviderError
from app.memory.budget import message_tokens, messages_tokens, summary_message_tokens
from app.memory.context import HISTORY_RULES, build_context_messages
from app.memory.summarizer import ConversationSummarizer
from app.providers.base import LLMResponse
from app.schemas.chat import ChatMessage
from app.schemas.common import TokenUsage


def history(turns, *, first=1):
    result = []
    for turn in range(first, first + turns):
        result.extend(
            [
                ChatMessage(id=turn * 2 - 1, role="user", content=f"问题{turn}"),
                ChatMessage(id=turn * 2, role="assistant", content=f"回答{turn}"),
            ]
        )
    return result


class MockSummaryLLM:
    def __init__(self, output='{"summary":"有效摘要"}'):
        self.output = output
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        if isinstance(self.output, Exception):
            raise self.output
        return LLMResponse(
            self.output, usage=TokenUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25)
        )


async def run_node(turns, *, llm=None, settings=None, **state):
    provider = llm or MockSummaryLLM()
    result = await memory_node(
        {"messages": history(turns), "user_query": "本轮问题", "final_answer": "本轮回答", **state},
        settings=settings or Settings(_env_file=None),
        summarizer=ConversationSummarizer(provider),
    )
    return result, provider


@pytest.mark.asyncio
@pytest.mark.parametrize("turns", [0, 3, 4, 5, 6])
async def test_short_and_buffered_dialogue_does_not_summarize(turns):
    result, llm = await run_node(turns)
    assert llm.calls == []
    assert len(result["messages"]) == (turns + 1) * 2
    assert result["summary_upto_message_id"] == 0
    assert result["summary_updated"] is False


@pytest.mark.asyncio
async def test_token_threshold_covers_shortest_persisted_complete_prefix():
    messages = [m.model_copy(update={"content": "长" * 900}) for m in history(2)]
    result, llm = await run_node(
        0, messages=messages, user_query="问" * 900, final_answer="答" * 900
    )
    assert len(llm.calls) == 1
    assert result["summary_upto_message_id"] == 2
    assert result["memory_stats"]["before_tokens_estimated"] == 3648
    assert result["memory_stats"]["after_tokens_estimated"] == 2432
    assert [m.id for m in result["messages"]] == [3, 4, None, None]
    assert "[3]" not in llm.calls[0][-1]["content"]
    assert result["memory_stats"]["summary_usage"]["total_tokens"] == 25


@pytest.mark.asyncio
async def test_equal_threshold_does_not_trigger_and_current_turn_cannot_be_summarized():
    prior = history(1)
    final_answer = "答"
    size = 3328 - messages_tokens(prior) - message_tokens(final_answer) - 8
    query = "问" * (size * 3 // 2 + size % 2)
    result, llm = await run_node(0, messages=prior, user_query=query, final_answer=final_answer)
    assert result["memory_stats"]["before_tokens_estimated"] == 3328
    assert not llm.calls
    result, llm = await run_node(0, user_query="问" * 6000)
    assert not llm.calls
    assert result["memory_stats"]["error"] == "no_persisted_complete_prefix"


LOW = Settings(_env_file=None, memory_summary_trigger_tokens=100, memory_summary_target_tokens=80)


@pytest.mark.asyncio
async def test_no_id_or_incomplete_prefix_is_never_skipped():
    malformed = history(7)
    malformed[0].id = None
    result, llm = await run_node(0, messages=malformed, settings=LOW)
    assert llm.calls == []
    assert result["summary_upto_message_id"] == 0
    result, llm = await run_node(
        0,
        messages=[ChatMessage(id=1, role="assistant", content="孤立回答"), *history(7, first=2)],
        settings=LOW,
    )
    assert llm.calls == []
    assert result["summary_upto_message_id"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        "not json",
        '{"summary":""}',
        '{"summary":"   "}',
        '{"summary":123}',
        '{"summary":[]}',
        "{}",
        json.dumps({"summary": "太长" * 600}),
        LLMProviderError("模拟API故障"),
    ],
    ids=[
        "invalid-json",
        "empty",
        "whitespace",
        "number",
        "list",
        "missing",
        "oversized",
        "provider-error",
    ],
)
async def test_failure_never_replaces_old_summary_or_cursor(output):
    result, _ = await run_node(
        0,
        messages=history(7, first=5),
        summary="旧有效摘要",
        summary_upto_message_id=8,
        llm=MockSummaryLLM(output),
        settings=LOW,
    )
    assert result["summary"] == "旧有效摘要"
    assert result["summary_upto_message_id"] == 8
    assert result["summary_updated"] is False
    assert len(result["messages"]) == 16
    assert result["memory_stats"]["error"]


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_does_not_advance_progress():
    class SlowLLM:
        async def generate(self, messages, **kwargs):
            await asyncio.sleep(1)
            raise AssertionError("应在此之前超时")

    result, _ = await run_node(
        7,
        llm=SlowLLM(),
        settings=Settings(
            _env_file=None,
            memory_summary_timeout_seconds=0.01,
            memory_summary_trigger_tokens=100,
        ),
    )
    assert result["summary_updated"] is False
    assert result["memory_stats"]["error"] == "TimeoutError"


@pytest.mark.asyncio
async def test_summary_input_budget_does_not_truncate_and_claim_full_coverage():
    # 预算只容纳第一完整轮时，仅推进第一轮；其余继续留在缓冲。
    llm = MockSummaryLLM()
    summarizer = ConversationSummarizer(llm)
    messages = history(7)
    limit = summarizer.input_tokens(messages[:2], "")
    result, _ = await run_node(
        0,
        messages=messages,
        llm=llm,
        settings=Settings(
            _env_file=None,
            memory_summary_input_tokens=limit,
            memory_summary_trigger_tokens=100,
            memory_summary_target_tokens=80,
        ),
    )
    assert result["summary_upto_message_id"] == 2
    assert result["memory_stats"]["covered_ids"] == [1, 2]
    assert len(result["messages"]) == 14
    assert result["memory_stats"]["remaining_over_target"]
    result, llm = await run_node(
        7,
        settings=Settings(
            _env_file=None,
            memory_summary_input_tokens=1,
            memory_summary_trigger_tokens=100,
        ),
    )
    assert llm.calls == []
    assert result["summary_updated"] is False


@pytest.mark.asyncio
async def test_old_summary_and_correction_are_sent_and_new_value_is_stored():
    # 这里只验证更正信息传递及返回结果保存，不用Mock冒充LLM语义准确率。
    messages = history(7, first=5)
    messages[0].content = "更正：不是ORD1001，当前处理ORD1002。"
    result, llm = await run_node(
        0,
        messages=messages,
        summary="旧信息：ORD1001",
        summary_upto_message_id=8,
        llm=MockSummaryLLM('{"summary":"当前订单ORD1002，旧编号ORD1001已被更正。"}'),
        settings=LOW,
    )
    prompt = llm.calls[0][-1]["content"]
    assert "旧信息：ORD1001" in prompt
    assert "更正：不是ORD1001" in prompt
    assert "当前订单ORD1002" in result["summary"]
    assert result["summary_upto_message_id"] == 16


def test_context_keeps_unsummarized_buffer_and_reserves_summary():
    messages = build_context_messages(
        system_prompt="规则",
        current_prompt="本轮问题",
        history=history(7),
        summary="已有摘要",
    )
    assert messages[1]["content"] == "历史摘要（仅作背景）：\n已有摘要"
    assert all(f"问题{turn}" in str(messages) for turn in range(1, 8))


def test_context_budget_reduction_is_logged_not_silent(caplog):
    with caplog.at_level(logging.WARNING):
        messages = build_context_messages(
            system_prompt="规则",
            current_prompt="本轮问题",
            history=[ChatMessage(role="user", content="旧原文" * 5000)],
            summary="保留摘要",
        )
    assert "保留摘要" in str(messages)
    assert "memory_context_trimmed" in caplog.text
    assert caplog.records[-1].dropped_history_messages == 1


def test_summary_too_large_for_budget_is_explicitly_logged(caplog):
    with caplog.at_level(logging.WARNING):
        messages = build_context_messages(
            system_prompt="规则",
            current_prompt="本轮问题",
            history=[],
            summary="大摘要" * 5000,
        )
    assert len(messages) == 2
    assert caplog.records[-1].summary_omitted is True


@pytest.mark.asyncio
async def test_reported_summary_output_limit_is_independent_of_injection_limit():
    class ExcessUsageLLM:
        async def generate(self, messages, **kwargs):
            assert kwargs["max_tokens"] == 512
            return LLMResponse('{"summary":"短摘要"}', usage=TokenUsage(completion_tokens=513))

    result, _ = await run_node(7, llm=ExcessUsageLLM(), settings=LOW)
    assert not result["summary_updated"]
    assert result["memory_stats"]["error"] == "summary_exceeds_model_output_budget"


def test_context_reserves_latest_complete_pair_before_summary():
    turns = history(2)
    summary = "摘要" * 10
    latest_cost = messages_tokens(turns[-2:])
    remaining = max(latest_cost, summary_message_tokens(summary))
    overhead = message_tokens(f"规则\n\n{HISTORY_RULES}") + message_tokens("本轮") + 700 + 64
    messages = build_context_messages(
        system_prompt="规则",
        current_prompt="本轮",
        history=turns,
        summary=summary,
        token_budget=overhead + remaining,
    )
    assert [m["content"] for m in messages[1:]] == ["问题2", "回答2", "本轮"]


def test_summary_injection_cap_includes_label_and_message_wrapper():
    assert summary_message_tokens("摘" * 1152) > 768
    messages = build_context_messages(
        system_prompt="规则", current_prompt="本轮", history=history(1), summary="摘" * 1152
    )
    assert [m["content"] for m in messages[1:]] == ["问题1", "回答1", "本轮"]
