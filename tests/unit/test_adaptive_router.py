import json

import pytest

from app.agents.nodes.intent import rule_based_intent
from app.core.enums import IntentLabel, RetrievalStrategy
from app.providers.base import LLMResponse
from app.retrieval.adaptive import _ORDER_ID, AdaptiveRouter, rule_based_complexity
from app.schemas.agent import IntentResult


def test_greeting_goes_direct() -> None:
    intent = rule_based_intent("你好")
    assert intent.label == IntentLabel.GREETING
    routed = rule_based_complexity("你好", intent)
    assert routed.strategy == RetrievalStrategy.DIRECT


def test_order_goes_agent() -> None:
    intent = IntentResult(label=IntentLabel.ORDER_INQUIRY, confidence=0.9, prompt_version="t")
    routed = rule_based_complexity("查一下订单 ORD10001", intent)
    assert routed.strategy == RetrievalStrategy.AGENT


def test_policy_multi_hop() -> None:
    intent = IntentResult(label=IntentLabel.FAQ_POLICY, confidence=0.8, prompt_version="t")
    query = "定制马克杯过了7天但是质量问题还能退吗"
    routed = rule_based_complexity(query, intent)
    assert routed.strategy in {RetrievalStrategy.MULTI, RetrievalStrategy.SINGLE}


def test_general_return_policy_does_not_request_order_tools() -> None:
    intent = IntentResult(label=IntentLabel.RETURN_EXCHANGE, confidence=0.8, prompt_version="t")
    routed = rule_based_complexity("七天无理由退货的规则是什么？", intent)
    assert routed.strategy == RetrievalStrategy.SINGLE


def test_specific_order_return_uses_tools() -> None:
    intent = IntentResult(label=IntentLabel.RETURN_EXCHANGE, confidence=0.8, prompt_version="t")
    routed = rule_based_complexity("订单 ORD10002 还能退货吗？", intent)
    assert routed.strategy == RetrievalStrategy.AGENT


# 本次新增：只用固定模型输出验证路由纠正，不访问真实 API。
class StubRouterLLM:
    def __init__(self, strategy: RetrievalStrategy) -> None:
        self.strategy = strategy

    async def generate(self, messages, **kwargs) -> LLMResponse:
        complexity = {
            RetrievalStrategy.DIRECT: "simple",
            RetrievalStrategy.SINGLE: "single_hop",
            RetrievalStrategy.MULTI: "multi_hop",
            RetrievalStrategy.AGENT: "tool_required",
            RetrievalStrategy.REFUSE: "out_of_scope",
        }[self.strategy]
        return LLMResponse(
            content=json.dumps(
                {
                    "complexity": complexity,
                    "strategy": self.strategy.value,
                    "confidence": 0.9,
                    "rationale": "stub",
                }
            )
        )

    async def ping(self) -> bool:
        return True

    async def stream(self, messages, **kwargs):
        yield ""


@pytest.mark.parametrize(
    "query",
    [
        "查一下订单ORD10001",
        "ORD10003发货了吗",
        "ORD10002的物流到哪了",
        "取消订单ord10003",
        "订单ORD-10003",
        "订单ORD_10003",
        "订单123456789012到哪了",
    ],
)
def test_order_id_next_to_chinese(query: str) -> None:
    # 防止中文相邻场景的正则边界问题再次出现。
    assert _ORDER_ID.search(query) is not None


@pytest.mark.parametrize("query", ["XORD10001A", "_ORD10001", "ORD10001_", "ORD123"])
def test_order_id_does_not_match_identifier_fragment(query: str) -> None:
    assert _ORDER_ID.search(query) is None


@pytest.mark.parametrize(
    ("query", "label"),
    [
        ("查一下订单ORD10001", IntentLabel.ORDER_INQUIRY),
        ("ORD10003发货了吗", IntentLabel.ORDER_INQUIRY),
        ("ORD10002的物流到哪了", IntentLabel.LOGISTICS),
        ("取消订单ORD10003", IntentLabel.CANCEL_ORDER),
        ("转人工", IntentLabel.HUMAN_HANDOFF),
        ("T恤还有库存吗", IntentLabel.PRODUCT_CONSULT),
        ("帮我退货，订单ORD10002", IntentLabel.RETURN_EXCHANGE),
    ],
)
@pytest.mark.parametrize(
    "wrong_strategy",
    [
        RetrievalStrategy.DIRECT,
        RetrievalStrategy.SINGLE,
        RetrievalStrategy.MULTI,
        RetrievalStrategy.REFUSE,
    ],
)
async def test_business_requests_override_wrong_model_route(query, label, wrong_strategy) -> None:
    # 模型即使返回合法但不适合业务的策略，也必须进入工具流程。
    router = AdaptiveRouter(StubRouterLLM(wrong_strategy))
    intent = IntentResult(label=label, confidence=0.9, prompt_version="test")
    result = await router.route(query, intent)
    assert result.strategy == RetrievalStrategy.AGENT


@pytest.mark.parametrize("wrong_strategy", [RetrievalStrategy.DIRECT, RetrievalStrategy.REFUSE])
async def test_faq_cannot_skip_retrieval(wrong_strategy) -> None:
    router = AdaptiveRouter(StubRouterLLM(wrong_strategy))
    intent = IntentResult(label=IntentLabel.FAQ_POLICY, confidence=0.9, prompt_version="test")
    result = await router.route("电子发票多久能开", intent)
    assert result.strategy == RetrievalStrategy.SINGLE


@pytest.mark.parametrize("strategy", [RetrievalStrategy.SINGLE, RetrievalStrategy.MULTI])
async def test_valid_policy_retrieval_choice_is_preserved(strategy) -> None:
    router = AdaptiveRouter(StubRouterLLM(strategy))
    intent = IntentResult(label=IntentLabel.RETURN_EXCHANGE, confidence=0.9, prompt_version="test")
    result = await router.route("七天无理由退货的规则是什么？", intent)
    assert result.strategy == strategy


async def test_general_cancellation_policy_does_not_force_tools() -> None:
    router = AdaptiveRouter(StubRouterLLM(RetrievalStrategy.AGENT))
    intent = IntentResult(label=IntentLabel.CANCEL_ORDER, confidence=0.9, prompt_version="test")
    result = await router.route("取消订单的规则是什么？", intent)
    assert result.strategy == RetrievalStrategy.SINGLE


async def test_inventory_policy_does_not_force_tools() -> None:
    router = AdaptiveRouter(StubRouterLLM(RetrievalStrategy.SINGLE))
    intent = IntentResult(label=IntentLabel.PRODUCT_CONSULT, confidence=0.9, prompt_version="test")
    result = await router.route("库存更新规则是什么？", intent)
    assert result.strategy == RetrievalStrategy.SINGLE


async def test_out_of_scope_cannot_be_overridden_by_model() -> None:
    router = AdaptiveRouter(StubRouterLLM(RetrievalStrategy.AGENT))
    intent = IntentResult(label=IntentLabel.OUT_OF_SCOPE, confidence=0.9, prompt_version="test")
    result = await router.route("超出业务范围的问题", intent)
    assert result.strategy == RetrievalStrategy.REFUSE
