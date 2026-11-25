"""Knowledge-base ingestion into BM25 and the vector store."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config.settings import Settings, get_settings
from app.core.enums import DocumentCategory
from app.providers.base import EmbeddingProvider
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.chunking import document_id_from_path
from app.retrieval.cleaning import dedupe_chunks
from app.retrieval.unstructured_parser import list_knowledge_files, parse_path
from app.retrieval.vector import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorRecord,
    build_records,
)
from app.schemas.retrieval import RetrievedChunk

# Prefix rules are ordered longest-intent first. Unknown stems fall back to FAQ.
_CATEGORY_PREFIXES: tuple[tuple[str, DocumentCategory], ...] = (
    ("return_", DocumentCategory.RETURN_POLICY),
    ("exchange_", DocumentCategory.RETURN_POLICY),
    ("missing_", DocumentCategory.RETURN_POLICY),
    ("refund_", DocumentCategory.RETURN_POLICY),
    ("shipping", DocumentCategory.SHIPPING),
    ("logistics", DocumentCategory.SHIPPING),
    ("pickup", DocumentCategory.SHIPPING),
    ("warranty", DocumentCategory.WARRANTY),
    ("repair", DocumentCategory.WARRANTY),
    ("coupon", DocumentCategory.COUPON),
    ("promo", DocumentCategory.COUPON),
    ("gift_card", DocumentCategory.COUPON),
    ("installment", DocumentCategory.PAYMENT),
    ("size_", DocumentCategory.PRODUCT),
    ("product_", DocumentCategory.PRODUCT),
    ("install", DocumentCategory.PRODUCT),
    ("payment", DocumentCategory.PAYMENT),
    ("invoice", DocumentCategory.PAYMENT),
    ("account", DocumentCategory.ACCOUNT),
    ("privacy", DocumentCategory.ACCOUNT),
    ("membership", DocumentCategory.ACCOUNT),
    ("points_", DocumentCategory.ACCOUNT),
    ("complaint", DocumentCategory.AFTERSALE),
    ("handoff", DocumentCategory.AFTERSALE),
    ("aftersale", DocumentCategory.AFTERSALE),
    ("live_", DocumentCategory.COUPON),
    ("cross_border", DocumentCategory.SHIPPING),
    ("restock", DocumentCategory.PRODUCT),
    ("packaging", DocumentCategory.PRODUCT),
    ("price_protect", DocumentCategory.AFTERSALE),
    ("faq", DocumentCategory.FAQ),
)


def category_for_stem(stem: str) -> DocumentCategory:
    lowered = stem.lower()
    for prefix, category in _CATEGORY_PREFIXES:
        if lowered == prefix.rstrip("_") or lowered.startswith(prefix):
            return category
    return DocumentCategory.FAQ


class KnowledgeIngestor:
    def __init__(
        self,
        knowledge_dir: str,
        embedder: EmbeddingProvider,
        settings: Settings | None = None,
    ) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._embedder = embedder
        self._settings = settings or get_settings()

    # 加载知识库 chunck纯文本块，records文本块+向量
    async def load(self) -> tuple[list[RetrievedChunk], list[VectorRecord]]:
        # 列出所有文件路径
        files = list_knowledge_files(self._knowledge_dir)
        chunks: list[RetrievedChunk] = []
        for path in files:
            # 按照文件名的前缀贴分类标签
            category = category_for_stem(path.stem)
            # 生成文档id
            document_id = document_id_from_path(path, root=self._knowledge_dir)
            title = path.stem
            # asyncio.to_thread 是为了异步调用 parse_path 函数，因为 parse_path 是同步的
            file_chunks = await asyncio.to_thread(
                parse_path,
                path,
                document_id=document_id,
                title=title,
                category=category,
                settings=self._settings,
            )
            # 每个文件的块增加到总表
            chunks.extend(file_chunks)
        # 去重
        chunks = dedupe_chunks(chunks)
        if not chunks:
            return [], []
        embeddings = await self._embedder.embed([chunk.content for chunk in chunks])
        records = build_records(chunks, embeddings)
        return chunks, records


async def ingest_to_memory(
    knowledge_dir: str,
    embedder: EmbeddingProvider,
    settings: Settings | None = None,
) -> tuple[BM25Retriever, InMemoryVectorStore, list[RetrievedChunk]]:
    ingestor = KnowledgeIngestor(knowledge_dir, embedder, settings=settings)
    chunks, records = await ingestor.load()
    bm25 = BM25Retriever(chunks)
    store = InMemoryVectorStore()
    store.upsert(records)
    return bm25, store, chunks


async def ingest_knowledge(
    knowledge_dir: str,
    embedder: EmbeddingProvider,
    *,
    settings: Settings | None = None,
    qdrant_url: str = "",
    qdrant_collection: str = "cs_knowledge",
    embedding_dim: int = 512,
) -> tuple[BM25Retriever, InMemoryVectorStore | QdrantVectorStore, list[RetrievedChunk]]:
    """Prefer Qdrant when configured; fall back to memory if Qdrant is down."""
    # 日志和配置
    logger = logging.getLogger(__name__)
    resolved = settings or get_settings()

    # 这是离线构造知识库的总入口，传入知识库的路径，embedder还有设置
    ingestor = KnowledgeIngestor(knowledge_dir, embedder, settings=resolved)
    # 加载知识库
    chunks, records = await ingestor.load()
    bm25 = BM25Retriever(chunks)
    if qdrant_url:
        # 如果配置了qdrant，就用qdrant存储，服务器地址，表名，向量维度
        store = QdrantVectorStore(qdrant_url, qdrant_collection, embedding_dim)
        try:
            # 确保表存在，不存在则创建，存在则不创建，把records更新到表中
            await asyncio.to_thread(store.ensure_collection)
            await asyncio.to_thread(store.upsert, records)
            return bm25, store, chunks
        except Exception:
            logger.warning("qdrant_ingest_failed_using_memory")
    memory = InMemoryVectorStore()
    # 内存存储，有则更新，没有则插入，records
    memory.upsert(records)
    return bm25, memory, chunks
