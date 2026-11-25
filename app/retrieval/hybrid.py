"""Hybrid retriever fusing BM25 and dense vectors with RRF."""

from __future__ import annotations

import logging
import time

from app.core.exceptions import RetrievalError
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.cache import RetrievalCache
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.vector import VectorRetriever
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class HybridRetriever:
    """BM25 + dense search, fused with RRF.

    Each retriever is allowed to fail independently. If both fail, raise.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        *,
        rrf_k: int,
        cache: RetrievalCache | None = None,
    ) -> None:
        self._bm25 = bm25
        self._vector = vector
        self._rrf_k = rrf_k
        self._cache = cache

    # 这个地方传入的是关键词和top-k
    async def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        # 首先是查缓存，如果有就直接返回缓存结果
        if self._cache is not None:
            cached = await self._cache.get(query, top_k)
            if cached is not None:
                return cached
        started = time.perf_counter()
        lexical: list[RetrievedChunk] = []
        dense: list[RetrievedChunk] = []
        lexical_error: Exception | None = None
        dense_error: Exception | None = None
        # bm25搜索，精准匹配
        try:
            lexical = self._bm25.search(query, top_k=top_k)
        except Exception as exc:  # BM25 is in-process; unexpected errors still isolated
            lexical_error = exc
            logger.exception("bm25_failed")
        # dense搜索，语义匹配，通过向量相似度计算
        try:
            dense = await self._vector.search(query, top_k=top_k)
        except Exception as exc:
            dense_error = exc
            logger.exception("vector_search_failed")
        if not lexical and not dense:
            raise RetrievalError("both lexical and dense retrievers failed") from (
                dense_error or lexical_error
            )
        # 融合bm25和dense搜索结果，返回排好的序RRF融合
        fused = reciprocal_rank_fusion([lexical, dense], k=self._rrf_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "hybrid_retrieval_completed",
            extra={
                "top_k": top_k,
                "lexical": len(lexical),
                "dense": len(dense),
                "fused": len(fused),
                "latency_ms": round(elapsed_ms, 2),
            },
        )
        # 截断融合结果，只返回top-k个
        trimmed = fused[:top_k]
        if self._cache is not None:
            await self._cache.set(query, top_k, trimmed)
        return trimmed
