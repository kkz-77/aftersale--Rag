"""Document retrieval node."""

from __future__ import annotations

import time
from typing import Any

from app.agents.state import AgentState
from app.config.settings import Settings
from app.core.enums import RetrievalStrategy
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.reranker import DocumentReranker
from app.retrieval.self_rag import SelfRAGLoop
from app.schemas.common import LatencyBreakdown
from app.schemas.retrieval import RetrievedChunk


async def retrieval_node(
    state: AgentState,
    *,
    retriever: HybridRetriever,
    reranker: DocumentReranker,
    rewriter: QueryRewriter,
    self_rag: SelfRAGLoop,
    settings: Settings,
) -> dict:
    started = time.perf_counter()
    query = state["user_query"]
    # 记忆改造第2步：只把历史传给模型，始终保留user_query原话。
    memory: dict[str, Any] = (
        {"history": list(state.get("messages") or []), "summary": state.get("summary") or ""}
        if settings.memory_in_decisions and (state.get("messages") or state.get("summary"))
        else {}
    )
    strategy = state.get("strategy")
    top_k = (
        settings.retrieval_top_k_complex
        if strategy == RetrievalStrategy.MULTI
        else settings.retrieval_top_k_simple
    )
    if strategy == RetrievalStrategy.MULTI:
        chunks = await self_rag.retrieve(query, top_k=top_k, **memory)
        rewritten = [query]
    else:
        # 一次改写调用，保留原问及最多三条不同的补充检索表达。
        suggestions = await rewriter.rewrite(query, **memory)
        rewritten = list(dict.fromkeys([query, *suggestions]))[:4]
        by_id: dict[str, RetrievedChunk] = {}
        for search_query in rewritten:
            for candidate in await retriever.retrieve(search_query, top_k=top_k):
                by_id.setdefault(candidate.chunk_id, candidate)
        # 有历史时第一条改写作为完整检索表达，也用于精排；失败则原问题兜底。
        ranking_query = suggestions[0] if memory and suggestions else query
        chunks = await reranker.rerank(
            ranking_query, list(by_id.values()), top_n=settings.rerank_top_n
        )
    latency = state.get("latency") or LatencyBreakdown()
    latency.retrieval_ms = (time.perf_counter() - started) * 1000
    return {"retrieved_chunks": chunks, "rewritten_queries": rewritten, "latency": latency}
