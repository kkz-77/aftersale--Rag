"""Test context wiring and safeguards with scripted providers, not model quality."""

import json

import pytest

from app.agents.nodes.generate import generate_node
from app.agents.nodes.intent import IntentClassifier
from app.agents.nodes.planner import Planner, planner_node
from app.agents.nodes.retrieval import retrieval_node
from app.agents.nodes.router import intent_node, router_node
from app.config.settings import Settings
from app.core.enums import IntentLabel, RetrievalStrategy
from app.core.exceptions import LLMProviderError
from app.providers.base import LLMResponse
from app.retrieval.adaptive import AdaptiveRouter
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.self_rag import SelfRAGLoop
from app.schemas.agent import IntentResult
from app.schemas.chat import ChatMessage
from app.schemas.retrieval import RetrievedChunk
from app.tools.aftersale_tools import CancelOrderTool
from app.tools.knowledge_tools import SearchKnowledgeTool
from app.tools.order_tools import GetOrderItemsTool, GetOrderTool
from app.tools.registry import ToolRegistry

HISTORY = [
    ChatMessage(role="user", content="我的订单是ORD10003。"),
    ChatMessage(role="assistant", content="请说明要查询什么。"),
]


class ScriptedLLM:
    """输出由测试预先设定；捕获真实节点传来的messages，不发送HTTP。"""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return LLMResponse(json.dumps(payload, ensure_ascii=False))


def _intent(label=IntentLabel.ORDER_INQUIRY):
    return IntentResult(label=label, confidence=0.9, prompt_version="memory_test")


def _chunk():
    return RetrievedChunk(
        chunk_id="return_policy_01",
        document_id="return_policy_hash",
        title="退货政策",
        content="定制商品质量问题可以申请售后。",
        score=1.0,
    )


class RecordingRetriever:
    def __init__(self):
        self.calls = []

    async def retrieve(self, query, *, top_k):
        self.calls.append(query)
        return [_chunk()]


class RecordingReranker:
    def __init__(self):
        self.calls = []

    async def rerank(self, query, chunks, *, top_n):
        self.calls.append(query)
        return chunks[:top_n]


def _registry():
    registry = ToolRegistry()
    for tool in [
        GetOrderTool(),
        GetOrderItemsTool(),
        CancelOrderTool(),
        SearchKnowledgeTool(RecordingRetriever()),
    ]:
        registry.register(tool)
    return registry


@pytest.mark.asyncio
async def test_history_recall_stays_direct_and_only_includes_summary_as_history():
    summary = "原订单 ORD10003 已替换，当前订单 ORD10001，待核对是否为头戴式。"
    llm = ScriptedLLM([{"answer": "ORD10001，待核对是否为头戴式。"}])
    state = {
        "user_query": "请只回顾更正后的订单编号和待办，不用查询订单。",
        "intent": _intent(),
        "summary": summary,
    }
    state.update(await router_node(state, AdaptiveRouter(llm)))
    assert state["strategy"] == RetrievalStrategy.DIRECT
    assert not llm.calls
    result = await generate_node(state, llm)
    assert summary in llm.calls[0][1]["content"]
    assert "历史原文和历史摘要可用于" in llm.calls[0][0]["content"]
    assert "ORD10001" in result["final_answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query", ["那笔订单现在发货了吗？", "只回顾之前的订单号，顺便查一下现在的物流。"]
)
async def test_live_order_request_with_memory_still_routes_to_agent(query):
    llm = ScriptedLLM([{"complexity": "simple", "strategy": "direct", "confidence": 0.9}])
    result = await router_node(
        {"user_query": query, "intent": _intent(), "messages": HISTORY}, AdaptiveRouter(llm)
    )
    assert result["strategy"] == RetrievalStrategy.AGENT
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_empty_history_does_not_turn_recall_into_order_lookup():
    llm = ScriptedLLM([])
    result = await router_node(
        {"user_query": "请只复述刚才提到的订单号，不用查订单。", "intent": _intent()},
        AdaptiveRouter(llm),
    )
    assert result["strategy"] == RetrievalStrategy.DIRECT
    assert not llm.calls


@pytest.mark.asyncio
async def test_no_tool_step_does_not_trigger_order_tracking_fallback():
    llm = ScriptedLLM([{"steps": [{"step_id": 1, "goal": "回顾对话", "tool_name": "none"}]}])
    result = await planner_node(
        {"user_query": "看看我之前说过什么", "intent": _intent()}, Planner(llm, _registry())
    )
    assert len(result["plan"].steps) == 1 and result["plan"].steps[0].tool_name is None


@pytest.mark.asyncio
async def test_order_plan_without_required_id_asks_before_executor():
    llm = ScriptedLLM([{"steps": [{"step_id": 1, "goal": "查状态", "tool_name": "get_order"}]}])
    state = {"user_query": "查一下订单状态", "intent": _intent()}
    state.update(await planner_node(state, Planner(llm, _registry())))
    assert not state["plan"].steps
    result = await generate_node(state, llm)
    assert result["final_answer"] == "请提供要查询的订单号。"


@pytest.mark.asyncio
async def test_order_item_followup_with_multiple_orders_asks_without_tool_plan():
    llm = ScriptedLLM([])
    state = {
        "user_query": "这个商品是什么？",
        "intent": _intent(IntentLabel.PRODUCT_CONSULT),
        "messages": [*HISTORY, ChatMessage(role="user", content="也查一下ORD10001。")],
    }
    state.update(await router_node(state, AdaptiveRouter(llm)))
    assert state["strategy"] == RetrievalStrategy.AGENT
    state.update(await planner_node(state, Planner(llm, _registry())))
    assert state["plan"].steps == []
    result = await generate_node(state, llm)
    assert "多笔订单" in result["final_answer"]
    assert not llm.calls


@pytest.mark.asyncio
async def test_explicit_order_item_query_takes_priority_over_history():
    llm = ScriptedLLM([])
    plan = await Planner(llm, _registry()).create_plan(
        "帮我查一下 ORD10001 里有哪些商品",
        "product_consult",
        "",
        history=HISTORY,
    )
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "get_order_items"
    assert plan.steps[0].arguments == {"order_id": "ORD10001"}


@pytest.mark.asyncio
async def test_intent_receives_history_but_raw_query_is_unchanged():
    llm = ScriptedLLM(
        [
            {
                "label": "order_inquiry",
                "confidence": 0.9,
                "rationale": "read status",
                "prompt_version": "memory_test",
            }
        ]
    )
    state = {"user_query": "这单现在什么状态？", "messages": HISTORY, "summary": "用户咨询订单。"}
    result = await intent_node(state, IntentClassifier(llm))
    assert result["intent"].label == IntentLabel.ORDER_INQUIRY
    assert "ORD10003" in str(llm.calls[0][:-1])
    assert "这单现在什么状态？" in llm.calls[0][-1]["content"]
    assert state["user_query"] == "这单现在什么状态？"


@pytest.mark.asyncio
async def test_router_receives_same_history_and_selects_tool_branch():
    llm = ScriptedLLM(
        [
            {
                "complexity": "tool_required",
                "strategy": "agent",
                "confidence": 0.9,
                "rationale": "order lookup",
            }
        ]
    )
    result = await router_node(
        {"user_query": "它现在是什么状态？", "messages": HISTORY, "intent": _intent()},
        AdaptiveRouter(llm),
    )
    assert result["strategy"] == RetrievalStrategy.AGENT
    assert "ORD10003" in str(llm.calls[0][:-1])


@pytest.mark.asyncio
async def test_planner_passes_history_and_keeps_read_tool_order_id():
    llm = ScriptedLLM(
        [
            {
                "steps": [
                    {
                        "step_id": 1,
                        "goal": "查询当前订单",
                        "tool_name": "get_order",
                        "arguments": {"order_id": "ORD10003"},
                    }
                ]
            }
        ]
    )
    result = await planner_node(
        {"user_query": "这单还在吗？", "messages": HISTORY, "intent": _intent()},
        Planner(llm, _registry()),
    )
    assert result["plan"].steps[0].arguments == {"order_id": "ORD10003"}
    assert result["pending_action"] is None
    assert "ORD10003" in str(llm.calls[0][:-1])
    assert "这单还在吗？" in llm.calls[0][-1]["content"]


@pytest.mark.asyncio
async def test_historical_cancel_cannot_authorize_a_new_write():
    # 故意模拟意图误判并产生写计划，使cancel_order在目录中，专门检查本轮授权过滤。
    llm = ScriptedLLM(
        [
            {
                "steps": [
                    {
                        "step_id": 1,
                        "goal": "错误重放取消",
                        "tool_name": "cancel_order",
                        "arguments": {"order_id": "ORD10003"},
                    }
                ]
            }
        ]
    )
    result = await planner_node(
        {
            "user_query": "这单现在什么状态？",
            "intent": _intent(IntentLabel.CANCEL_ORDER),
            "messages": [
                ChatMessage(role="user", content="取消订单ORD10003。"),
                ChatMessage(role="assistant", content="已取消ORD10003。"),
            ],
        },
        Planner(llm, _registry()),
    )
    assert result["plan"].steps == []
    assert result["pending_action"] is None


@pytest.mark.asyncio
async def test_current_explicit_write_with_historical_id_still_needs_confirmation():
    llm = ScriptedLLM(
        [
            {
                "steps": [
                    {
                        "step_id": 1,
                        "goal": "取消订单",
                        "tool_name": "cancel_order",
                        "arguments": {"order_id": "ORD10003"},
                    }
                ]
            }
        ]
    )
    result = await planner_node(
        {
            "user_query": "帮我取消这笔订单。",
            "intent": _intent(IntentLabel.CANCEL_ORDER),
            "messages": HISTORY,
        },
        Planner(llm, _registry()),
    )
    assert result["pending_action"].arguments == {"order_id": "ORD10003"}
    assert result["pending_action"].tool_name == "cancel_order"


@pytest.mark.asyncio
async def test_single_searches_each_distinct_rewrite_with_one_model_call():
    query = "数码保修多久，进水能修吗？"
    alternatives = ["数码配件保修期限", "进液免保条款", "进液付费维修"]
    llm = ScriptedLLM(
        [{"queries": [query, alternatives[0], alternatives[0], *alternatives[1:], "第四条"]}]
    )
    retriever, reranker = RecordingRetriever(), RecordingReranker()

    result = await retrieval_node(
        {"user_query": query, "strategy": RetrievalStrategy.SINGLE},
        retriever=retriever,
        reranker=reranker,
        rewriter=QueryRewriter(llm),
        self_rag=None,
        settings=Settings(),
    )

    assert retriever.calls == [query, *alternatives]
    assert result["rewritten_queries"] == retriever.calls
    assert reranker.calls == [query]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_single_uses_contextual_query_for_retrieval_and_reranking():
    expanded = "定制马克杯 超过7天 质量问题 退货政策"
    llm = ScriptedLLM([{"queries": [expanded]}])
    retriever, reranker = RecordingRetriever(), RecordingReranker()
    state = {
        "user_query": "过了7天有质量问题还能退吗？",
        "strategy": RetrievalStrategy.SINGLE,
        "messages": [ChatMessage(role="user", content="我买的是定制马克杯。")],
        "summary": "签收已经10天。",
    }
    result = await retrieval_node(
        state,
        retriever=retriever,
        reranker=reranker,
        rewriter=QueryRewriter(llm),
        self_rag=None,
        settings=Settings(memory_in_decisions=True),
    )
    assert expanded in retriever.calls
    assert reranker.calls == [expanded]
    assert result["rewritten_queries"] == [state["user_query"], expanded]
    assert "定制马克杯" in str(llm.calls[0][:-1])
    assert "签收已经10天" in str(llm.calls[0][:-1])


@pytest.mark.asyncio
async def test_multi_passes_history_to_rewriter_and_uses_contextual_grading_query():
    expanded = "定制马克杯 质量问题 退货条件"
    llm = ScriptedLLM(
        [
            {"queries": [expanded]},
            {
                "relevant": True,
                "sufficient": True,
                "missing_aspect": "",
                "checks": [
                    {
                        "aspect": "退货条件",
                        "supporting_chunk_ids": ["return_policy_01"],
                        "missing_aspect": "",
                    }
                ],
            },
        ]
    )
    retriever, reranker = RecordingRetriever(), RecordingReranker()
    loop = SelfRAGLoop(retriever, reranker, QueryRewriter(llm), llm, Settings())
    await loop.retrieve(
        "它有问题能退吗？",
        top_k=8,
        history=[ChatMessage(role="user", content="我买的是定制马克杯。")],
    )
    assert len(llm.calls) == 2  # 原有改写和grade各一次，没有额外Resolver。
    assert expanded in retriever.calls
    assert reranker.calls == [expanded]
    assert expanded in llm.calls[1][-1]["content"]


@pytest.mark.asyncio
async def test_rewrite_failure_uses_original_query_without_extra_call():
    llm = ScriptedLLM([LLMProviderError("模拟超时")])
    result = await QueryRewriter(llm).rewrite("它能退吗？", history=HISTORY)
    assert result == ["它能退吗？"]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_invalid_rewrite_list_is_not_split_into_characters():
    llm = ScriptedLLM([{"queries": "退货政策"}])
    assert await QueryRewriter(llm).rewrite("它能退吗？", history=HISTORY) == ["它能退吗？"]


@pytest.mark.asyncio
async def test_decision_switch_removes_history_from_intent_and_router():
    intent_llm = ScriptedLLM(
        [
            {
                "label": "order_inquiry",
                "confidence": 0.9,
                "prompt_version": "test",
            }
        ]
    )
    state = {"user_query": "这单什么状态？", "messages": HISTORY, "summary": "订单ORD10003"}
    await intent_node(state, IntentClassifier(intent_llm), use_history=False)
    assert len(intent_llm.calls[0]) == 2
    assert "ORD10003" not in str(intent_llm.calls[0])
    router_llm = ScriptedLLM(
        [
            {
                "complexity": "tool_required",
                "strategy": "agent",
                "confidence": 0.9,
            }
        ]
    )
    await router_node({**state, "intent": _intent()}, AdaptiveRouter(router_llm), use_history=False)
    assert "ORD10003" not in str(router_llm.calls[0])


@pytest.mark.asyncio
async def test_decision_switch_removes_history_from_planner():
    llm = ScriptedLLM([{"steps": []}])
    await planner_node(
        {
            "user_query": "这单什么状态？",
            "messages": HISTORY,
            "summary": "订单ORD10003",
            "intent": _intent(),
        },
        Planner(llm, _registry()),
        use_history=False,
    )
    assert len(llm.calls[0]) == 2
    assert "ORD10003" not in str(llm.calls[0])


@pytest.mark.asyncio
async def test_generation_switch_removes_history_for_baseline_a():
    llm = ScriptedLLM([{"answer": "请补充信息。", "citation_chunk_ids": []}])
    await generate_node(
        {"user_query": "这是什么？", "messages": HISTORY, "summary": "订单ORD10003"},
        llm,
        use_history=False,
    )
    assert len(llm.calls[0]) == 2
    assert "ORD10003" not in str(llm.calls[0])


@pytest.mark.asyncio
async def test_decision_switch_keeps_original_query_for_single_reranking():
    llm = ScriptedLLM([{"queries": ["退货政策"]}])
    reranker = RecordingReranker()
    await retrieval_node(
        {
            "user_query": "它能退吗？",
            "messages": HISTORY,
            "summary": "订单ORD10003",
            "strategy": RetrievalStrategy.SINGLE,
        },
        retriever=RecordingRetriever(),
        reranker=reranker,
        rewriter=QueryRewriter(llm),
        self_rag=None,
        settings=Settings(memory_in_decisions=False),
    )
    assert "ORD10003" not in str(llm.calls[0])
    assert reranker.calls == ["它能退吗？"]
