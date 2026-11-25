"""Intent and complexity routing nodes."""

from __future__ import annotations

import time
from typing import Any

from app.agents.nodes.intent import IntentClassifier
from app.agents.state import AgentState
from app.core.enums import QueryComplexity, RetrievalStrategy
from app.memory.order_reference import order_item_references
from app.retrieval.adaptive import AdaptiveRouter
from app.schemas.agent import ComplexityResult
from app.schemas.common import LatencyBreakdown


def _latency(state: AgentState) -> LatencyBreakdown:
    return state.get("latency") or LatencyBreakdown()


async def intent_node(
    state: AgentState, classifier: IntentClassifier, *, use_history: bool = True
) -> dict:
    started = time.perf_counter()
    memory: dict[str, Any] = (
        {"history": list(state.get("messages") or []), "summary": state.get("summary") or ""}
        if use_history and (state.get("messages") or state.get("summary"))
        else {}
    )
    intent = await classifier.classify(state["user_query"], **memory)
    latency = _latency(state)
    latency.intent_ms = (time.perf_counter() - started) * 1000
    return {"intent": intent, "latency": latency}


async def router_node(
    state: AgentState, router: AdaptiveRouter, *, use_history: bool = True
) -> dict:
    started = time.perf_counter()
    intent = state["intent"]
    assert intent is not None
    memory: dict[str, Any] = (
        {"history": list(state.get("messages") or []), "summary": state.get("summary") or ""}
        if use_history and (state.get("messages") or state.get("summary"))
        else {}
    )
    references = order_item_references(
        state["user_query"], intent.label.value, memory.get("history") or []
    )
    if state.get("confirmed_action") or references:
        complexity = ComplexityResult(
            complexity=QueryComplexity.TOOL_REQUIRED,
            strategy=RetrievalStrategy.AGENT,
            confidence=1.0,
            rationale="confirmed_action" if state.get("confirmed_action") else "order_items_lookup",
        )
    else:
        complexity = await router.route(state["user_query"], intent, **memory)
    latency = _latency(state)
    latency.routing_ms = (time.perf_counter() - started) * 1000
    return {
        # 包含分类度，策略，置信度和理由
        "complexity": complexity,
        "strategy": complexity.strategy,
        "query_complexity": complexity.complexity,
        "latency": latency,
    }
