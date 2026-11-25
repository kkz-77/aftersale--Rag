"""Generate format repair and read-only fallback; no network, model weights or DB."""

import json
import logging

import pytest

from app.agents.nodes.generate import _offline_answer, generate_node
from app.core.enums import RetrievalStrategy, ToolRisk
from app.core.exceptions import LLMProviderError
from app.providers.base import LLMResponse
from app.schemas.agent import ToolResult
from app.schemas.chat import ActionProposal, ChatMessage
from app.schemas.common import TokenUsage
from app.schemas.retrieval import RetrievedChunk


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def completion(content, prompt=10, output=3):
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return LLMResponse(
        content,
        TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=output,
            total_tokens=prompt + output,
        ),
    )


def shipment(**updates):
    data = {
        "order_id": "ORD10001",
        "tracking_no": "TEST-TRACK",
        "events": [
            {"status": "warehouse", "at": "2026-11-16T10:00:00"},
            {"status": "in_transit", "at": "2026-11-17T10:00:00"},
        ],
    }
    data.update(updates)
    return ToolResult(name="track_shipment", success=True, output=data)


def state(tools=None, **updates):
    return {
        "user_query": "它的物流怎么样？",
        "strategy": RetrievalStrategy.AGENT,
        "tool_results": [shipment()] if tools is None else tools,
        "trace_id": "generate-mock",
        **updates,
    }


def invalid_completion():
    return completion([{"name": "track_shipment", "output": {"order_id": "ORD10001"}}])


@pytest.mark.asyncio
async def test_normal_payload_one_call_preserves_citations_refusal_and_prior_usage():
    chunk = RetrievedChunk(
        chunk_id="c1", document_id="policy", title="政策", content="政策原文", score=1
    )
    llm = ScriptedLLM(
        completion(
            {
                "answer": "证据不足，无法确认。",
                "citation_chunk_ids": ["c1", "fake"],
                "refused": True,
            }
        )
    )
    result = await generate_node(
        state(
            [],
            retrieved_chunks=[chunk],
            token_usage=TokenUsage(
                prompt_tokens=20,
                completion_tokens=2,
                total_tokens=22,
            ),
        ),
        llm,
    )
    assert len(llm.calls) == 1
    assert result["final_answer"] == "证据不足，无法确认。"
    assert result["refused"] is True
    assert [c.chunk_id for c in result["citations"]] == ["c1"]
    assert result["token_usage"] == TokenUsage(
        prompt_tokens=30, completion_tokens=5, total_tokens=35
    )


@pytest.mark.asyncio
async def test_tool_array_repaired_once_with_original_evidence_history_and_schema(caplog):
    raw = invalid_completion()
    llm = ScriptedLLM(
        raw,
        completion(
            {
                "answer": "订单 ORD10001 的物流状态：运输中。",
                "citation_chunk_ids": [],
                "refused": False,
            },
            prompt=30,
            output=5,
        ),
    )
    with caplog.at_level(logging.WARNING):
        result = await generate_node(
            state(
                messages=[ChatMessage(role="user", content="查ORD10001")],
                summary="用户咨询订单物流",
            ),
            llm,
        )
    assert len(llm.calls) == 2
    original, repaired = llm.calls[0][0], llm.calls[1][0]
    assert repaired[:-1] == original
    assert any("TEST-TRACK" in m["content"] for m in repaired[:-1])
    assert any("查ORD10001" in m["content"] for m in repaired[:-1])
    repair_prompt = repaired[-1]["content"]
    assert json.dumps(raw.content, ensure_ascii=False) in repair_prompt
    assert "Final answer JSON root" in repair_prompt
    assert "additionalProperties" in repair_prompt and "refused" in repair_prompt
    assert "不得执行工具或补造事实" in repair_prompt
    assert llm.calls[1][1]["temperature"] == 0.0
    assert llm.calls[1][1]["response_format"] == {"type": "json_object"}
    assert result["final_answer"].endswith("运输中。")
    assert result["token_usage"] == TokenUsage(
        prompt_tokens=40, completion_tokens=8, total_tokens=48
    )
    assert result["latency"].llm_ms > 0
    assert [r.phase for r in caplog.records if r.message == "answer_schema_failed"] == ["generate"]


@pytest.mark.asyncio
async def test_two_invalid_outputs_fallback_to_real_shipment_and_keep_both_usages():
    llm = ScriptedLLM(invalid_completion(), completion("not-json", prompt=20, output=4))
    result = await generate_node(state(), llm)
    assert len(llm.calls) == 2
    assert result["final_answer"] == "订单 ORD10001 的物流状态：运输中。"
    assert result["token_usage"] == TokenUsage(
        prompt_tokens=30, completion_tokens=7, total_tokens=37
    )
    assert result["citations"] == [] and result["refused"] is False
    assert "到达" not in result["final_answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        '{"answer":""}',
        '{"answer":"   "}',
        '{"answer":123}',
        '{"answer":"答复","refused":"false"}',
        '{"answer":"答复","tool":"track_shipment"}',
        '[{"answer":"数组不能被作为合法对象"}]',
        '```json\n[{"answer":"数组不能被作为合法对象"}]\n```',
        '{"answer":"答复","citation_chunk_ids":[123]}',
    ],
)
async def test_invalid_payloads_get_exactly_one_repair(raw):
    llm = ScriptedLLM(completion(raw), completion({"answer": "已修复", "refused": True}))
    result = await generate_node(state(), llm)
    assert len(llm.calls) == 2
    assert result["final_answer"] == "已修复" and result["refused"] is True


@pytest.mark.asyncio
async def test_failed_tool_never_claims_success_and_skips_llm():
    llm = ScriptedLLM()
    failed = shipment().model_copy(update={"success": False, "error": "not found"})
    result = await generate_node(state([failed]), llm)
    assert not llm.calls
    assert "暂时无法查到" in result["final_answer"]
    assert "运输中" not in result["final_answer"]


@pytest.mark.asyncio
async def test_no_tools_or_evidence_keeps_safe_fallback():
    llm = ScriptedLLM(invalid_completion(), invalid_completion())
    result = await generate_node(state([]), llm)
    assert len(llm.calls) == 2
    assert "暂时无法基于现有信息" in result["final_answer"]
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_evidence_fallback_keeps_real_citation():
    chunk = RetrievedChunk(
        chunk_id="c1", document_id="policy", title="政策", content="原文可申请售后", score=1
    )
    llm = ScriptedLLM(invalid_completion(), invalid_completion())
    result = await generate_node(state([], retrieved_chunks=[chunk]), llm)
    assert "原文可申请售后" in result["final_answer"]
    assert [c.chunk_id for c in result["citations"]] == ["c1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tools",
    [
        [shipment(events=[])],
        [shipment(events="not a list")],
        [shipment(events=[None])],
        [shipment(events=[{"status": "invented"}])],
        [shipment(order_id="")],
        [shipment(), shipment(order_id="ORD10002")],
        [shipment(), shipment(events=[{"status": "delivered"}])],
        [shipment(), shipment().model_copy(update={"success": False})],
        [shipment().model_copy(update={"truncated": True})],
        [
            ToolResult(
                name="cancel_order",
                success=True,
                risk=ToolRisk.DESTRUCTIVE,
                output={"order_id": "ORD10001"},
            )
        ],
        [ToolResult(name="create_refund_request", success=True, output={"order_id": "ORD10001"})],
        [
            ToolResult(
                name="get_order",
                success=True,
                output={"order_id": "ORD10001", "status": "invented"},
            )
        ],
        [
            ToolResult(
                name="get_order_items",
                success=True,
                output={"items": [{"name": "杯子", "quantity": True}]},
            )
        ],
        [ToolResult(name="get_order_items", success=True, output={"items": []})],
    ],
)
async def test_missing_conflicting_or_write_data_does_not_use_deterministic_success(tools):
    llm = ScriptedLLM(invalid_completion(), invalid_completion())
    result = await generate_node(state(tools), llm)
    assert "现有业务数据不足" in result["final_answer"]
    assert "运输中" not in result["final_answer"]
    assert "已取消" not in result["final_answer"] and "申请已提交" not in result["final_answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        (shipment(), "订单 ORD10001 的物流状态：运输中。"),
        (
            ToolResult(
                name="get_order", success=True, output={"order_id": "ORD10001", "status": "shipped"}
            ),
            "订单 ORD10001 当前状态：已发货。",
        ),
        (
            ToolResult(
                name="get_order_items",
                success=True,
                output={"items": [{"name": "杯子", "quantity": 2}]},
            ),
            "查询到的订单商品（共1项）：杯子 × 2。",
        ),
    ],
)
async def test_readonly_fallback_and_offline_share_formatter(tool, expected):
    llm = ScriptedLLM(invalid_completion(), invalid_completion())
    current = state([tool])
    result = await generate_node(current, llm)
    assert result["final_answer"] == _offline_answer(current)[0] == expected


@pytest.mark.asyncio
async def test_initial_api_failure_has_no_format_retry_or_fabricated_usage(caplog):
    llm = ScriptedLLM(LLMProviderError("mock unavailable"))
    with caplog.at_level(logging.WARNING):
        result = await generate_node(state(token_usage=TokenUsage(total_tokens=9)), llm)
    assert len(llm.calls) == 1
    assert result["token_usage"] == TokenUsage(total_tokens=9)
    assert result["final_answer"].endswith("运输中。")
    assert any(r.message == "answer_api_failed" and r.phase == "generate" for r in caplog.records)
    assert not any(r.message == "answer_schema_failed" for r in caplog.records)


@pytest.mark.asyncio
async def test_repair_api_failure_keeps_first_successful_usage(caplog):
    llm = ScriptedLLM(invalid_completion(), LLMProviderError("mock timeout", timeout=True))
    with caplog.at_level(logging.WARNING):
        result = await generate_node(state(), llm)
    assert len(llm.calls) == 2
    assert result["token_usage"] == TokenUsage(
        prompt_tokens=10, completion_tokens=3, total_tokens=13
    )
    assert result["final_answer"].endswith("运输中。")
    assert any(
        r.message == "answer_api_failed" and r.phase == "format_repair" for r in caplog.records
    )


@pytest.mark.asyncio
async def test_pending_write_confirmation_still_skips_generation_and_repair():
    pending = ActionProposal(
        action_id="a1",
        tool_name="cancel_order",
        arguments={"order_id": "ORD10001"},
        risk=ToolRisk.DESTRUCTIVE,
        summary="确认后才能取消订单。",
    )
    llm = ScriptedLLM()
    result = await generate_node(state([], pending_action=pending), llm)
    assert not llm.calls and result["final_answer"] == pending.summary
