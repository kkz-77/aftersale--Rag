"""Query rewriting for multi-hop and ambiguous queries."""

from __future__ import annotations

from app.core.exceptions import LLMOutputError, LLMProviderError
from app.memory.context import build_decision_messages
from app.prompts.query_rewrite import REWRITE_SYSTEM, REWRITE_USER
from app.providers.base import LLMProvider
from app.providers.fake import OfflineLLMProvider
from app.providers.json_parser import extract_json_object
from app.schemas.chat import ChatMessage


class QueryRewriter:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def rewrite(
        self,
        query: str,
        *,
        missing_aspect: str = "",
        history: list[ChatMessage] | None = None,
        summary: str = "",
    ) -> list[str]:
        if isinstance(self._llm, OfflineLLMProvider):
            return [query]
        try:
            response = await self._llm.generate(
                # 记忆改造第2步：复用原来这次改写调用，不额外请求一次LLM。
                build_decision_messages(
                    system_prompt=REWRITE_SYSTEM,
                    current_prompt=REWRITE_USER.format(
                        query=query, missing_aspect=missing_aspect or "none"
                    ),
                    history=history,
                    summary=summary,
                ),
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            # 解析 JSON 响应
            payload = extract_json_object(response.content)
            # 提取 queries 字段 是关键词列表
            queries = payload.get("queries") or []
            # 非列表不能当成字符序列使用；错误结果退回原问题。
            if not isinstance(queries, list):
                return [query]
            # 清理关键词列表，移除空字符串
            cleaned = [item.strip() for item in queries if isinstance(item, str) and item.strip()]
            return list(dict.fromkeys(item for item in cleaned if item != query))[:3] or [query]
        except (LLMProviderError, LLMOutputError):
            return [query]
