"""Session restoration uses an isolated temporary SQLite database, never Qwen."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.exceptions import InvalidRequestError
from app.repositories.db import init_schema
from app.repositories.session_repository import SessionRepository
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService


@pytest.fixture
async def memory_sessions(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'conversation.db').as_posix()}")
    await init_schema(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_next_request_restores_persisted_summary_and_history(memory_sessions) -> None:
    service = ChatService.__new__(ChatService)
    request = ChatRequest(message="我咨询的是定制马克杯。", user_id="user_memory")
    async with memory_sessions() as db:
        await SessionRepository(db).get_or_create("session_memory", request.user_id)
        await service._persist(
            db,
            request,
            ChatResponse(
                answer="请说明问题。", session_id="session_memory", trace_id="trace_first"
            ),
        )
        repo = SessionRepository(db)
        saved = await repo.unsummarized_messages("session_memory", after_id=0)
        await repo.save_summary(
            "session_memory", "用户购买了定制马克杯。", upto_message_id=saved[-1].id
        )
        await repo.append_message("session_memory", "user", "还有另一个问题。")
        await repo.append_message("session_memory", "assistant", "请继续。")
        await db.commit()
    # 模拟下一次请求使用新的DB Session恢复，不依赖上次请求的Python变量。
    async with memory_sessions() as db:
        state = await service._initial_state(
            db,
            ChatRequest(message="杯柄断了。", user_id=request.user_id, session_id="session_memory"),
            "session_memory",
            "trace_next",
        )
        assert state["summary"] == "用户购买了定制马克杯。"
        assert [message.content for message in state["messages"]] == [
            "还有另一个问题。",
            "请继续。",
        ]
        assert state["user_query"] == "杯柄断了。"


@pytest.mark.asyncio
async def test_another_session_has_no_history_or_summary(memory_sessions) -> None:
    service = ChatService.__new__(ChatService)
    async with memory_sessions() as db:
        repo = SessionRepository(db)
        await repo.get_or_create("first_session", "owner")
        await repo.append_message("first_session", "user", "第一会话的消息")
        await repo.append_message("first_session", "assistant", "第一会话的回答")
        saved = await repo.unsummarized_messages("first_session", after_id=0)
        await repo.save_summary(
            "first_session", "不应出现在其他会话的摘要", upto_message_id=saved[-1].id
        )
        await db.commit()
    async with memory_sessions() as db:
        state = await service._initial_state(
            db, ChatRequest(message="你好", user_id="owner"), "second_session", "trace_second"
        )
        assert state["messages"] == []
        assert state["summary"] == ""


@pytest.mark.asyncio
async def test_other_user_cannot_load_existing_session_memory(memory_sessions) -> None:
    service = ChatService.__new__(ChatService)
    async with memory_sessions() as db:
        await SessionRepository(db).get_or_create("private_session", "owner")
        await db.commit()
    async with memory_sessions() as db:
        with pytest.raises(InvalidRequestError, match="会话不存在或不属于当前用户"):
            await service._initial_state(
                db,
                ChatRequest(message="你好", user_id="someone_else"),
                "private_session",
                "trace_rejected",
            )
