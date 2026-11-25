"""Real service/SQLite transactions with a mock graph+LLM, never paid APIs."""

import asyncio
import json
import sqlite3
from weakref import WeakValueDictionary

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.nodes.memory import memory_node
from app.config.settings import Settings
from app.core.exceptions import InvalidRequestError
from app.memory.summarizer import ConversationSummarizer
from app.providers.base import LLMResponse
from app.providers.fake import FakeLLMProvider
from app.repositories.db import init_schema
from app.repositories.models import MessageRow, SessionMemoryRow
from app.repositories.session_repository import SessionRepository
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import TokenUsage
from app.schemas.retrieval import RetrievedChunk
from app.services.chat_service import ChatService
from app.tools.registry import ToolRegistry
from scripts.migrate_memory import create_session_memory


class MockLLM:
    def __init__(self):
        self.calls = []
        self.fail = False

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        if self.fail:
            return LLMResponse('{"summary":null}')
        return LLMResponse(
            json.dumps({"summary": f"滚动摘要{len(self.calls)}"}, ensure_ascii=False),
            usage=TokenUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        )


class MockGraph:
    """业务回答固定，但调用真实memory_node及真实服务持久化。"""

    def __init__(self, settings, llm):
        self.settings, self.llm = settings, llm
        self.states, self.updates = [], []

    async def ainvoke(self, state):
        self.states.append(state.copy())
        answer = "回答：" + state["user_query"]
        update = await memory_node(
            {**state, "final_answer": answer},
            settings=self.settings,
            summarizer=ConversationSummarizer(self.llm),
        )
        self.updates.append(update)
        return {**state, **update, "final_answer": answer}

    async def astream(self, state, **kwargs):
        result = await self.ainvoke(state)
        yield {"compact_memory": result}


class AllowAll:
    async def allow(self, key):
        return True


def service(factory, llm=None, **options):
    # 不初始化任何远程Provider/Embedding，明确注入无网络实现。
    instance = ChatService.__new__(ChatService)
    instance._settings = Settings(_env_file=None, **options)
    instance._session_factory = factory
    instance._rate_limiter = AllowAll()
    instance._pending_actions = {}
    instance._session_locks = WeakValueDictionary()
    instance._graph = MockGraph(instance._settings, llm or MockLLM())
    return instance


@pytest.fixture
async def sessions(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}")
    await init_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def ask(instance, turn, session_id="rolling", user_id="owner", *, long=False):
    return await instance.chat(
        ChatRequest(
            message=f"问题{turn}" + ("长" * 1300 if long else ""),
            session_id=session_id,
            user_id=user_id,
        ),
        trace_id=f"trace_{turn}",
    )


@pytest.mark.asyncio
async def test_short_conversation_never_creates_memory(sessions):
    instance = service(sessions)
    for turn in range(1, 13):
        await ask(instance, turn)
    assert not instance._graph.llm.calls
    async with sessions() as db:
        assert await SessionRepository(db).get_memory("rolling") is None
        assert await db.scalar(select(func.count()).select_from(MessageRow)) == 24


@pytest.mark.asyncio
async def test_first_and_incremental_summary_preserve_full_messages(sessions):
    instance = service(sessions)
    await ask(instance, 1, long=True)
    async with sessions() as db:
        assert await SessionRepository(db).get_memory("rolling") is None
    await ask(instance, 2, long=True)
    async with sessions() as db:
        memory = await SessionRepository(db).get_memory("rolling")
        assert (memory.summary, memory.summary_upto_message_id) == ("滚动摘要1", 2)
        assert memory.updated_at is not None
    await ask(instance, 3, long=True)
    async with sessions() as db:
        memory = await SessionRepository(db).get_memory("rolling")
        assert (memory.summary, memory.summary_upto_message_id) == ("滚动摘要2", 4)
        assert await db.scalar(select(func.count()).select_from(MessageRow)) == 6
        assert await db.scalar(select(func.count()).select_from(SessionMemoryRow)) == 1
    first, second = [call[-1]["content"] for call in instance._graph.llm.calls]
    assert "[1] user:" in first and "[3] user:" not in first
    assert "滚动摘要1" in second and "[3] user:" in second and "[1] user:" not in second


@pytest.mark.asyncio
async def test_failure_keeps_old_summary_then_retries_from_same_cursor(sessions):
    instance = service(sessions)
    await ask(instance, 1, long=True)
    await ask(instance, 2, long=True)
    llm = instance._graph.llm
    llm.fail = True
    await ask(instance, 3, long=True)
    async with sessions() as db:
        row = await db.get(SessionMemoryRow, "rolling")
        assert (row.summary, row.summary_upto_message_id) == ("滚动摘要1", 2)
        assert await db.scalar(select(func.count()).select_from(MessageRow)) == 6
    llm.fail = False
    await ask(instance, 4, long=True)
    assert instance._graph.updates[-1]["memory_stats"]["covered_ids"] == [3, 4, 5, 6]
    assert "[3] user:" in llm.calls[-1][-1]["content"]


@pytest.mark.asyncio
async def test_sessions_and_users_remain_isolated(sessions):
    instance = service(sessions)
    for turn in range(1, 9):
        await ask(instance, turn, "session_a", "a")
    await ask(instance, 1, "session_b", "b")
    assert instance._graph.states[-1]["summary"] == ""
    assert instance._graph.states[-1]["messages"] == []
    with pytest.raises(InvalidRequestError):
        await ask(instance, 2, "session_a", "b")


@pytest.mark.asyncio
async def test_file_db_survives_engine_and_service_restart(tmp_path):
    url = f"sqlite+aiosqlite:///{(tmp_path / 'restart.db').as_posix()}"
    first_engine = create_async_engine(url)
    await init_schema(first_engine)
    first = service(async_sessionmaker(first_engine, expire_on_commit=False))
    for turn in (1, 2):
        await ask(first, turn, long=True)
    await first_engine.dispose()
    second_engine = create_async_engine(url)
    try:
        second = service(async_sessionmaker(second_engine, expire_on_commit=False))
        await ask(second, 3)
        restored = second._graph.states[-1]
        assert restored["summary"] == "滚动摘要1"
        assert restored["summary_upto_message_id"] == 2
        assert [m.id for m in restored["messages"]] == [3, 4]
        assert second._graph.llm.calls == []
    finally:
        await second_engine.dispose()


@pytest.mark.asyncio
async def test_summary_and_current_turn_rollback_together(sessions):
    instance = service(sessions)
    await ask(instance, 1, long=True)
    async with sessions() as db:
        result = await memory_node(
            {
                "messages": await SessionRepository(db).unsummarized_messages(
                    "rolling", after_id=0
                ),
                "user_query": "问" * 1300,
                "final_answer": "答" * 1300,
            },
            settings=instance._settings,
            summarizer=ConversationSummarizer(MockLLM()),
        )
        await instance._persist(
            db,
            ChatRequest(message="问题8", user_id="owner"),
            ChatResponse(answer="回答8", session_id="rolling", trace_id="rollback"),
            summary=result["summary"],
            summary_updated=True,
            summary_upto_message_id=result["summary_upto_message_id"],
        )
        await db.flush()
        await db.rollback()  # 模拟最终提交前失败。
    async with sessions() as db:
        row = await db.get(SessionMemoryRow, "rolling")
        assert row is None
        assert await db.scalar(select(func.count()).select_from(MessageRow)) == 2


@pytest.mark.asyncio
async def test_cursor_rejects_other_session_boundary(sessions):
    instance = service(sessions)
    await ask(instance, 1, "a")
    await ask(instance, 1, "b")
    async with sessions() as db:
        with pytest.raises(InvalidRequestError):
            await SessionRepository(db).save_summary("a", "无效摘要", upto_message_id=4)


@pytest.mark.asyncio
async def test_sse_uses_same_summary_save_path(sessions):
    instance = service(sessions)
    await ask(instance, 1, long=True)
    events = [
        event
        async for event in instance.stream(
            ChatRequest(message="问" * 1300, session_id="rolling", user_id="owner"), trace_id="sse"
        )
    ]
    assert events[-1]["event"] == "done"
    async with sessions() as db:
        row = await db.get(SessionMemoryRow, "rolling")
        assert row.summary_upto_message_id == 2
        assert await db.scalar(select(func.count()).select_from(MessageRow)) == 4


@pytest.mark.asyncio
async def test_same_process_session_requests_are_serialized(sessions):
    instance = service(sessions)
    await asyncio.gather(ask(instance, 1), ask(instance, 2))
    assert len(instance._graph.states[0]["messages"]) == 0
    assert len(instance._graph.states[1]["messages"]) == 2


def test_migration_adds_only_memory_table_and_is_repeatable(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, summary TEXT)")
        db.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")
        db.execute("INSERT INTO sessions VALUES ('legacy', '已有摘要')")
        db.execute("INSERT INTO messages VALUES (1, '原始历史')")
    with sqlite3.connect(path) as db:
        assert create_session_memory(db) is True
        assert "summary_upto_message_id" not in {
            r[1] for r in db.execute("PRAGMA table_info(sessions)")
        }
        assert create_session_memory(db, apply=True) is True
        assert create_session_memory(db, apply=True) is False
        assert db.execute("SELECT summary FROM sessions").fetchone() == ("已有摘要",)
        assert db.execute("PRAGMA foreign_key_list(session_memory)").fetchone()[2:5] == (
            "sessions",
            "session_id",
            "session_id",
        )
        assert db.execute("SELECT content FROM messages").fetchone() == ("原始历史",)


@pytest.mark.asyncio
async def test_migrated_old_sqlite_is_readable_by_actual_service(tmp_path):
    path = tmp_path / "old_schema.db"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, summary TEXT NOT NULL,
            created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)""")
        db.execute("""CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at DATETIME NOT NULL)""")
        db.execute(
            "INSERT INTO sessions VALUES ('old', 'owner', '旧摘要独有标记', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
        db.execute(
            "INSERT INTO messages VALUES (1, 'old', 'user', ?, CURRENT_TIMESTAMP)",
            ("旧问题" * 440,),
        )
        db.execute(
            "INSERT INTO messages VALUES (2, 'old', 'assistant', ?, CURRENT_TIMESTAMP)",
            ("旧回答" * 440,),
        )
    engine = create_async_engine(f"sqlite+aiosqlite:///{path.as_posix()}")
    try:
        await init_schema(engine)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        instance = service(factory)
        await ask(instance, 1, "new")
        assert instance._graph.states[-1]["summary"] == ""
        await ask(instance, 2, "old", long=True)
        initial = instance._graph.states[-1]
        assert initial["summary"] == ""
        assert initial["summary_upto_message_id"] == 0
        assert [m.id for m in initial["messages"]] == [1, 2]
        assert "旧摘要独有标记" not in instance._graph.llm.calls[0][-1]["content"]
        async with factory() as db:
            repo = SessionRepository(db)
            assert await repo.get_memory("new") is None
            assert (await repo.get_memory("old")).summary_upto_message_id == 2
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT summary FROM sessions WHERE session_id='old'").fetchone() == (
                "旧摘要独有标记",
            )
            assert "summary_upto_message_id" not in {
                r[1] for r in db.execute("PRAGMA table_info(sessions)")
            }
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_real_langgraph_preserves_summary_update_through_service_commit(sessions):
    """不只验证MockGraph：实际Graph编排、memory更新与事务提交也连通。"""
    llm = FakeLLMProvider(
        canned={
            '"label"': '{"label":"return_exchange","confidence":0.9,"prompt_version":"test"}',
            '"complexity"': '{"complexity":"single_hop","strategy":"single","confidence":0.9}',
            '"queries"': '{"queries":["退货政策"]}',
            '"answer"': '{"answer":"请按政策处理。","citation_chunk_ids":[]}',
            '"supported"': '{"supported":true,"hallucinated_spans":[]}',
            '"summary"': '{"summary":"已讨论退货政策，继续核实用户条件。"}',
        }
    )

    class Retriever:
        async def retrieve(self, query, *, top_k):
            return [
                RetrievedChunk(
                    chunk_id="policy_01",
                    document_id="policy",
                    title="政策",
                    content="质量问题可按规则处理。",
                    score=1.0,
                )
            ]

    class Reranker:
        async def rerank(self, query, chunks, *, top_n):
            return chunks[:top_n]

    instance = ChatService(
        settings=Settings(_env_file=None),
        llm=llm,
        retriever=Retriever(),
        reranker=Reranker(),
        session_factory=sessions,
        rate_limiter=AllowAll(),
        registry=ToolRegistry(),
    )
    for turn in (1, 2):
        response = await instance.chat(
            ChatRequest(
                message=f"退货政策问题{turn}" + "长" * 2600,
                user_id="owner",
                session_id="real_graph",
            ),
            trace_id=f"graph_{turn}",
        )
        assert response.answer == "请按政策处理。"
    async with sessions() as db:
        row = await db.get(SessionMemoryRow, "real_graph")
        assert row.summary_upto_message_id == 2
        assert row.summary == "已讨论退货政策，继续核实用户条件。"
        assert await db.scalar(select(func.count()).select_from(MessageRow)) == 4
