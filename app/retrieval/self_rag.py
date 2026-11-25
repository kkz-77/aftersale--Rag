"""Self-RAG style retrieve → grade → optionally hop again.

Reflection is implemented as structured LLM JSON, not special tokens,
so it stays schema-validated and testable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config.settings import Settings
from app.core.exceptions import LLMOutputError, LLMProviderError
from app.prompts.self_rag import GRADE_SYSTEM, GRADE_USER
from app.providers.base import LLMProvider
from app.providers.json_parser import parse_model
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.reranker import DocumentReranker
from app.schemas.chat import ChatMessage
from app.schemas.retrieval import RetrievedChunk


class AspectCheck(BaseModel):
    aspect: str
    supporting_chunk_ids: list[str]
    missing_aspect: str


class GradeResult(BaseModel):
    relevant: bool
    sufficient: bool
    missing_aspect: str = ""
    rationale: str = ""
    checks: list[AspectCheck] = Field(default_factory=list)

    def validate_evidence(self, visible_ids: set[str]) -> GradeResult:
        gaps = [self.missing_aspect.strip()]
        if not self.checks:
            gaps.append("逐项核对问题所需的政策条款，补查没有直接证据的要点")
        for check in self.checks:
            gaps.append(check.missing_aspect.strip())
            if (
                not check.aspect.strip()
                or not check.supporting_chunk_ids
                or any(chunk_id not in visible_ids for chunk_id in check.supporting_chunk_ids)
            ):
                gaps.append(
                    check.missing_aspect.strip()
                    or check.aspect.strip()
                    or "补查未获证据支持的政策要点"
                )
        missing = "; ".join(dict.fromkeys(gap for gap in gaps if gap))
        return self.model_copy(
            update={"sufficient": self.sufficient and not missing, "missing_aspect": missing}
        )


class SelfRAGLoop:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: DocumentReranker,
        rewriter: QueryRewriter,
        llm: LLMProvider,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._rewriter = rewriter
        self._llm = llm
        self._settings = settings

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        history: list[ChatMessage] | None = None,
        summary: str = "",
    ) -> list[RetrievedChunk]:
        return await self._retrieve(query, top_k=top_k, history=history, summary=summary)

    async def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
        history: list[ChatMessage] | None = None,
        summary: str = "",
        first_hop: tuple[list[RetrievedChunk], GradeResult] | None = None,
    ) -> list[RetrievedChunk]:
        initial, grade = first_hop if first_hop else ([], None)
        accumulated = {chunk.chunk_id: chunk for chunk in initial}
        visible = initial[:6]
        if grade is not None:
            grade = grade.validate_evidence({chunk.chunk_id for chunk in visible})
        # 记忆改造第2步：每轮共享同一份会话背景，缺失项仍由grade单独反馈。
        memory: dict[str, Any] = (
            {"history": history or [], "summary": summary} if history or summary else {}
        )
        for _hop in range(int(first_hop is not None), self._settings.max_retrieval_hops):
            if grade is not None and grade.sufficient:
                break
            # 这个里面的missing是上一轮的反馈
            rewritten = await self._rewriter.rewrite(
                query,
                missing_aspect=(grade.missing_aspect or "补查尚未覆盖的政策条款") if grade else "",
                **memory,
            )
            # query放第一个，重写的放后面根据dict进行去重，提取前2个
            queries = list(dict.fromkeys([query, *rewritten]))[:2]
            candidates_by_id: dict[str, RetrievedChunk] = {}
            for search_query in queries:
                for candidate in await self._retriever.retrieve(search_query, top_k=top_k):
                    candidates_by_id.setdefault(candidate.chunk_id, candidate)
            ranking_query = rewritten[0] if memory and rewritten else query
            reranked = await self._reranker.rerank(
                ranking_query, list(candidates_by_id.values()), top_n=self._settings.rerank_top_n
            )
            new = [chunk for chunk in reranked if chunk.chunk_id not in accumulated]
            for chunk in reranked:
                accumulated[chunk.chunk_id] = chunk
            # Keep cited support and make room for newly found evidence in the grading window.
            cited = self._cited_chunks(grade, visible)
            ordered = dict.fromkeys(
                chunk.chunk_id for chunk in [*cited, *new, *accumulated.values()]
            )
            visible = [accumulated[chunk_id] for chunk_id in ordered][:6]
            grade = await self._grade(ranking_query, visible)
        ordered = dict.fromkeys(
            chunk.chunk_id
            for chunk in [*self._cited_chunks(grade, visible), *visible, *accumulated.values()]
        )
        return [accumulated[chunk_id] for chunk_id in ordered][: self._settings.rerank_top_n * 2]

    @staticmethod
    def _cited_chunks(
        grade: GradeResult | None, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if grade is None:
            return []
        ids = {chunk_id for check in grade.checks for chunk_id in check.supporting_chunk_ids}
        return [chunk for chunk in chunks if chunk.chunk_id in ids]

    async def _grade(self, query: str, chunks: list[RetrievedChunk]) -> GradeResult:
        visible = chunks[:6]
        evidence = "\n\n".join(f"[{c.chunk_id}] {c.content}" for c in visible) or "(empty)"
        try:
            response = await self._llm.generate(
                [
                    {"role": "system", "content": GRADE_SYSTEM},
                    {"role": "user", "content": GRADE_USER.format(query=query, evidence=evidence)},
                ],
                temperature=0.0,
                max_tokens=768,
                response_format={"type": "json_object"},
            )
            return parse_model(response.content, GradeResult).validate_evidence(
                {c.chunk_id for c in visible}
            )
        except (LLMProviderError, LLMOutputError):
            return GradeResult(
                relevant=bool(chunks),
                sufficient=False,
                rationale="fallback",
                missing_aspect="重新核对问题所需的具体政策条款",
            )
