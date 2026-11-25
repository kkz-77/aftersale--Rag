"""Adaptive-RAG complexity routing.

Paper mapping (Jeong et al., ACL 2024):
- simple      -> DIRECT (no retrieval) or REFUSE
- single_hop  -> SINGLE (one-shot hybrid retrieval)
- multi_hop   -> MULTI  (Self-RAG iterative retrieval)
- tool needs  -> AGENT  (planner + tools, optional retrieval)

LLM classifier is preferred. Rule fallback keeps the system testable and
available when the model is down.
"""

from __future__ import annotations

import re

from app.core.constants import PROMPT_VERSION_COMPLEXITY
from app.core.enums import IntentLabel, QueryComplexity, RetrievalStrategy
from app.core.exceptions import LLMOutputError, LLMProviderError
from app.memory.context import build_decision_messages
from app.prompts.complexity import COMPLEXITY_SYSTEM, COMPLEXITY_USER
from app.providers.base import LLMProvider
from app.providers.fake import OfflineLLMProvider
from app.providers.json_parser import parse_model
from app.schemas.agent import ComplexityResult, IntentResult
from app.schemas.chat import ChatMessage

_MULTI_HOP_CUES = (
    "同时",
    "并且",
    "还是",
    "如果",
    "但是已经",
    "过了",
    "既",
    "又",
    " besides",
    "and then",
)
_TOOL_CUES = (
    "订单",
    "物流",
    "快递",
    "运单",
    "退款进度",
    "取消订单",
    "我的单",
    "order",
    "tracking",
)
# 本次修复：中文也属于 Unicode 的单词字符，\b 会漏掉“订单ORD10001”。
# 改为检查两侧不是 ASCII 标识符字符；允许紧邻中文，避免从 XORD10001A 中截取。
_ORDER_ID = re.compile(
    r"(?<![A-Za-z0-9_])(?:ORD[-_]?[0-9]{4,}|[0-9]{12,})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_PERSONAL_REFUND = re.compile(
    r"(?:我的|我(?:这|那|上)(?:一)?笔|(?:这|那)(?:一)?笔|刚才那笔)的?退款|我退的款"
)
_RECALL_ONLY = re.compile(r"(?:只|仅)(?:需要|需|要|用)?(?:回顾|复述|回想|回忆|总结)")
_RECALL_ACTION = re.compile(
    r"(?:现在|目前|实时).{0,12}(?:查|发货|送到|到账|状态|库存|取消|退款|退货)"
    r"|(?:^|再|顺便|同时|另外|帮我|请|给我|然后|并且|并)(?:帮我|给我|执行)?(?:查|取消|申请退款|退款|退货|下单)"
)


def _history_recall_only(query: str) -> bool:
    clauses = re.split(r"[，,。；;！？!?\n]", query)
    affirmative = [
        clause.strip()
        for clause in clauses
        if not clause.strip().startswith(("不用", "无需", "不要", "别", "不必"))
    ]
    return any(_RECALL_ONLY.search(clause) for clause in affirmative) and not any(
        _RECALL_ACTION.search(clause) for clause in affirmative
    )


def _needs_inventory_lookup(query: str, intent: IntentResult) -> bool:
    # 本次补充：询问商品实时库存需要工具；库存规则/更新机制仍属于知识问答。
    return (
        intent.label == IntentLabel.PRODUCT_CONSULT
        and any(cue in query for cue in ("库存", "有货", "现货", "缺货", "补货"))
        and not any(
            cue in query
            for cue in ("规则", "政策", "机制", "含义", "什么意思", "怎么更新", "多久更新")
        )
    )


# 6中情况，根据意图判断复杂程度
def rule_based_complexity(query: str, intent: IntentResult) -> ComplexityResult:
    text = query.strip()
    # 超出处理范围的
    if intent.label == IntentLabel.OUT_OF_SCOPE:
        return ComplexityResult(
            complexity=QueryComplexity.OUT_OF_SCOPE,
            strategy=RetrievalStrategy.REFUSE,
            confidence=0.95,
            rationale="intent_out_of_scope",
        )
    # 问候或聊天
    if intent.label in {IntentLabel.GREETING, IntentLabel.CHITCHAT}:
        return ComplexityResult(
            complexity=QueryComplexity.SIMPLE,
            strategy=RetrievalStrategy.DIRECT,
            confidence=0.9,
            rationale="social_or_greeting",
        )
    aftersale_intent = intent.label in {IntentLabel.REFUND, IntentLabel.RETURN_EXCHANGE}
    # “到账”也出现在通用时效问题里，只有指向具体退款时才据此查账户数据。
    personal_refund_status = (
        aftersale_intent
        and _PERSONAL_REFUND.search(text) is not None
        and any(cue in text for cue in ("到账", "进度", "状态", "审核", "查", "处理到"))
    )
    # 没有订单号，但是有订单相关的关键词
    policy_question = not _ORDER_ID.search(text) and any(
        cue in text
        for cue in (
            "规则",
            "政策",
            "条件",
            "怎么算",
            "可以吗",
            "能不能",
            "怎么退",
            "能退吗",
            "能取消吗",
        )
    )
    # 如果是政策，但是没有订单号，且意图是取消订单，退款，退货
    if (
        policy_question
        and not personal_refund_status
        and intent.label
        in {
            IntentLabel.CANCEL_ORDER,
            IntentLabel.REFUND,
            IntentLabel.RETURN_EXCHANGE,
        }
    ):
        return ComplexityResult(
            complexity=QueryComplexity.SINGLE_HOP,
            strategy=RetrievalStrategy.SINGLE,
            confidence=0.78,
            rationale="general_policy_without_order",
        )
    transactional = intent.label in {
        IntentLabel.ORDER_INQUIRY,
        IntentLabel.LOGISTICS,
        IntentLabel.CANCEL_ORDER,
        IntentLabel.HUMAN_HANDOFF,
    }
    needs_account_data = aftersale_intent and (
        bool(_ORDER_ID.search(text))
        or personal_refund_status
        or any(cue in text for cue in ("帮我退", "申请退款", "退款进度"))
    )

    if (
        transactional
        or needs_account_data
        # 本次补充：让库存查询的备用策略也是 AGENT，模型失败时仍走工具流程。
        or _needs_inventory_lookup(text, intent)
        or _ORDER_ID.search(text)
        or any(cue in text.lower() for cue in _TOOL_CUES)
    ):
        return ComplexityResult(
            complexity=QueryComplexity.TOOL_REQUIRED,
            strategy=RetrievalStrategy.AGENT,
            confidence=0.8,
            rationale="entity_or_transactional_intent",
        )

    hop_cues = sum(1 for cue in _MULTI_HOP_CUES if cue in text)
    if hop_cues >= 1 and len(text) > 18:
        return ComplexityResult(
            complexity=QueryComplexity.MULTI_HOP,
            strategy=RetrievalStrategy.MULTI,
            confidence=0.7,
            rationale="multi_constraint_query",
        )
    return ComplexityResult(
        complexity=QueryComplexity.SINGLE_HOP,
        strategy=RetrievalStrategy.SINGLE,
        confidence=0.75,
        rationale="single_policy_or_faq",
    )


class AdaptiveRouter:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def route(
        self,
        query: str,
        intent: IntentResult,
        *,
        history: list[ChatMessage] | None = None,
        summary: str = "",
        diagnostics: dict | None = None,
    ) -> ComplexityResult:
        details = diagnostics if diagnostics is not None else {}
        fallback = rule_based_complexity(query, intent)
        details.update(
            pure_rule_strategy=fallback.strategy.value,
            pure_rule_reason=fallback.rationale,
            model_strategy=None,
            model_complexity=None,
            applied_rule=None,
            fallback_used=False,
            failure_reason=None,
            logical_model_calls=0,
        )
        if intent.label != IntentLabel.OUT_OF_SCOPE and _history_recall_only(query):
            details["applied_rule"] = "history_recall_only"
            return ComplexityResult(
                complexity=QueryComplexity.SIMPLE,
                strategy=RetrievalStrategy.DIRECT,
                confidence=1.0,
                rationale="history_recall_only",
            )
        if isinstance(self._llm, OfflineLLMProvider):
            details.update(
                fallback_used=True, failure_reason="offline_rules", applied_rule="offline_rules"
            )
            return fallback
        try:
            details["logical_model_calls"] = 1
            response = await self._llm.generate(
                # 记忆改造第2步：历史用于理解本轮是否需要业务数据，不拼进规则字符串。
                build_decision_messages(
                    system_prompt=COMPLEXITY_SYSTEM,
                    current_prompt=COMPLEXITY_USER.format(
                        query=query,
                        intent=intent.label.value,
                        prompt_version=PROMPT_VERSION_COMPLEXITY,
                    ),
                    history=history,
                    summary=summary,
                ),
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            parsed = parse_model(response.content, ComplexityResult)
            details.update(
                model_strategy=parsed.strategy.value, model_complexity=parsed.complexity.value
            )
            # 本次补充：越界意图不能被后续路由模型改成工具操作或普通回答。
            if intent.label == IntentLabel.OUT_OF_SCOPE:
                details["applied_rule"] = "out_of_scope_protection"
                return fallback

            # 本次调整：通用售后政策保留 SINGLE/MULTI 选择，但不直接回答、拒答或调订单工具。
            if fallback.rationale == "general_policy_without_order" and parsed.strategy not in {
                RetrievalStrategy.SINGLE,
                RetrievalStrategy.MULTI,
            }:
                details["applied_rule"] = "general_policy_without_order"
                return fallback

            # 本次补充：明确业务意图必须进入已有的 AGENT 分支，不允许模型绕过工具。
            # 前面的政策分支优先，避免把“取消订单的规则”误当成执行取消。
            tool_intent = intent.label in {
                IntentLabel.ORDER_INQUIRY,
                IntentLabel.LOGISTICS,
                IntentLabel.CANCEL_ORDER,
                IntentLabel.HUMAN_HANDOFF,
                IntentLabel.REFUND,
                IntentLabel.RETURN_EXCHANGE,
            }
            if fallback.strategy == RetrievalStrategy.AGENT and (
                _ORDER_ID.search(query) or tool_intent or _needs_inventory_lookup(query, intent)
            ):
                details["applied_rule"] = "business_requires_agent"
                return fallback

            # 本次补充：平台政策以知识库为依据，不能因模型认为“简单”就跳过检索。
            # 仅纠正 DIRECT/REFUSE，正常 SINGLE/MULTI 的自适应选择保持不变。
            if intent.label == IntentLabel.FAQ_POLICY and parsed.strategy in {
                RetrievalStrategy.DIRECT,
                RetrievalStrategy.REFUSE,
            }:
                details["applied_rule"] = "policy_requires_retrieval"
                return ComplexityResult(
                    complexity=QueryComplexity.SINGLE_HOP,
                    strategy=RetrievalStrategy.SINGLE,
                    confidence=fallback.confidence,
                    rationale="policy_requires_retrieval",
                )
            return parsed
        except (LLMProviderError, LLMOutputError) as exc:
            details.update(
                fallback_used=True,
                failure_reason="request_failed"
                if isinstance(exc, LLMProviderError)
                else "parse_or_validation_failed",
                error_type=type(exc).__name__,
                applied_rule="error_fallback",
            )
            return fallback
