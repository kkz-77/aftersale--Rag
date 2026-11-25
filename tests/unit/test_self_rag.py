"""Unit tests for the Self-RAG retrieve → grade → hop loop.

The loop is driven with stubs so no real retrieval, rerank or LLM call happens.
"""

from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.core.exceptions import LLMProviderError
from app.providers.fake import FakeLLMProvider
from app.retrieval.self_rag import AspectCheck, GradeResult, SelfRAGLoop
from app.schemas.retrieval import RetrievedChunk

_SUFFICIENT = (
    '{"relevant": true, "sufficient": true, "missing_aspect": "", "rationale": "ok", '
    '"checks": [{"aspect":"退货条件","supporting_chunk_ids":["c1"],"missing_aspect":""}]}'
)
_NOT_SUFFICIENT = (
    '{"relevant": true, "sufficient": false, "missing_aspect": "赔付", "rationale": "partial"}'
)


def _chunk(chunk_id: str, content: str = "政策正文") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"policy_{chunk_id}",
        title="政策",
        content=content,
        score=1.0,
        rank=1,
        source="bm25",
    )


class _StubRetriever:
    def __init__(self, per_query: dict[str, list[RetrievedChunk]]) -> None:
        self._per_query = per_query
        self.calls: list[str] = []

    async def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        self.calls.append(query)
        return list(self._per_query.get(query, []))


class _StubReranker:
    async def rerank(self, query, chunks, *, top_n):
        return list(chunks)[:top_n]


class _PassthroughReranker:
    """Ignores top_n so the accumulator grows past it — used to test the final cap."""

    async def rerank(self, query, chunks, *, top_n):
        return list(chunks)


class _StubRewriter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def rewrite(self, query: str, *, missing_aspect: str = "") -> list[str]:
        self.calls.append(missing_aspect)
        return [query]


class _RaisingLLM:
    async def generate(self, messages, **kwargs):
        raise LLMProviderError("grading llm down")


def _loop(*, llm, retriever, hops: int = 3, rerank_top_n: int = 5) -> SelfRAGLoop:
    return SelfRAGLoop(
        retriever=retriever,
        reranker=_StubReranker(),
        rewriter=_StubRewriter(),
        llm=llm,
        settings=Settings(max_retrieval_hops=hops, rerank_top_n=rerank_top_n),
    )


@pytest.mark.asyncio
async def test_loop_stops_when_first_grade_is_sufficient() -> None:
    llm = FakeLLMProvider(canned={"": _SUFFICIENT})
    retriever = _StubRetriever({"退货": [_chunk("c1")]})
    loop = _loop(llm=llm, retriever=retriever)

    chunks = await loop.retrieve("退货", top_k=3)

    assert [c.chunk_id for c in chunks] == ["c1"]
    assert len(retriever.calls) == 1
    assert len(llm.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "checks",
    [
        "[]",
        '[{"aspect":"退货条件","supporting_chunk_ids":["c1","unknown"],"missing_aspect":""}]',
        '[{"aspect":"退款时效","supporting_chunk_ids":["c1"],"missing_aspect":"银行卡到账时效"}]',
    ],
)
async def test_positive_grade_without_complete_support_continues(checks) -> None:
    content = '{"relevant":true,"sufficient":true,"checks":' + checks + "}"
    llm = FakeLLMProvider(canned={"": content})
    retriever = _StubRetriever({"退货": [_chunk("c1")]})
    loop = _loop(llm=llm, retriever=retriever)
    rewriter = _StubRewriter()
    loop._rewriter = rewriter
    await loop.retrieve("退货", top_k=3)
    assert len(llm.calls) == 3
    assert rewriter.calls[1]
    if "银行卡" in checks:
        assert rewriter.calls[1] == "银行卡到账时效"


@pytest.mark.asyncio
async def test_grade_cannot_cite_chunk_outside_prompt_window() -> None:
    llm = FakeLLMProvider(canned={"": _SUFFICIENT.replace('"c1"', '"c7"')})
    loop = _loop(llm=llm, retriever=_StubRetriever({}))
    grade = await loop._grade("退货", [_chunk(f"c{i}") for i in range(1, 8)])
    assert not grade.sufficient
    assert grade.missing_aspect == "退货条件"


@pytest.mark.asyncio
async def test_continuation_exposes_new_support_to_grader_and_generation() -> None:
    first = [_chunk(f"c{i}") for i in range(6)]
    complete = _SUFFICIENT.replace('["c1"]', '["c1","late"]')
    llm = FakeLLMProvider(canned={"": complete})
    retriever = _StubRetriever({"退货": [_chunk("late")]})
    loop = _loop(llm=llm, retriever=retriever)
    initial_grade = GradeResult(
        relevant=True,
        sufficient=False,
        checks=[
            AspectCheck(aspect="退货资格", supporting_chunk_ids=["c1"], missing_aspect=""),
            AspectCheck(aspect="退费条件", supporting_chunk_ids=[], missing_aspect="退费条件"),
        ],
    )
    result = await loop._retrieve("退货", top_k=8, first_hop=(first, initial_grade))
    assert len(llm.calls) == 1
    assert "[late]" in llm.calls[0][-1]["content"]
    assert {"c1", "late"} <= {chunk.chunk_id for chunk in result[:6]}


@pytest.mark.asyncio
async def test_loop_hops_again_and_dedupes_chunks() -> None:
    llm = FakeLLMProvider(canned={"": _NOT_SUFFICIENT})
    retriever = _StubRetriever({"退货": [_chunk("c1"), _chunk("c2")]})
    loop = _loop(llm=llm, retriever=retriever, hops=2)

    chunks = await loop.retrieve("退货", top_k=3)

    # Same chunks come back on both hops, so the accumulator must dedupe.
    assert [c.chunk_id for c in chunks] == ["c1", "c2"]
    assert len(retriever.calls) == 2
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_loop_respects_max_hops() -> None:
    llm = FakeLLMProvider(canned={"": _NOT_SUFFICIENT})
    retriever = _StubRetriever({"退货": [_chunk("c1")]})
    loop = _loop(llm=llm, retriever=retriever, hops=1)

    await loop.retrieve("退货", top_k=3)

    assert len(retriever.calls) == 1


@pytest.mark.asyncio
async def test_loop_truncates_to_twice_rerank_top_n() -> None:
    llm = FakeLLMProvider(canned={"": _SUFFICIENT})
    chunks = [_chunk(f"c{i}") for i in range(10)]
    retriever = _StubRetriever({"退货": chunks})
    loop = SelfRAGLoop(
        retriever=retriever,
        reranker=_PassthroughReranker(),
        rewriter=_StubRewriter(),
        llm=llm,
        settings=Settings(max_retrieval_hops=1, rerank_top_n=2),
    )

    got = await loop.retrieve("退货", top_k=10)

    assert len(got) == 4  # 10 accumulated, capped at rerank_top_n * 2


@pytest.mark.asyncio
async def test_loop_degrades_when_grading_llm_fails() -> None:
    retriever = _StubRetriever({"退货": [_chunk("c1")]})
    loop = _loop(llm=_RaisingLLM(), retriever=retriever, hops=3)

    chunks = await loop.retrieve("退货", top_k=3)

    assert [c.chunk_id for c in chunks] == ["c1"]
    assert len(retriever.calls) == 3


@pytest.mark.asyncio
async def test_loop_returns_empty_when_nothing_retrieved_and_grading_fails() -> None:
    retriever = _StubRetriever({})  # every query yields nothing
    loop = _loop(llm=_RaisingLLM(), retriever=retriever, hops=1)

    assert await loop.retrieve("退货", top_k=3) == []
