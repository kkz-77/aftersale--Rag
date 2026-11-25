"""Conversation session and message persistence."""

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import InvalidRequestError
from app.core.time import utcnow
from app.repositories.models import MessageRow, SessionMemoryRow, SessionRow
from app.schemas.chat import ChatMessage


class SessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create(self, session_id: str, user_id: str) -> SessionRow:
        row = await self._session.get(SessionRow, session_id)
        # 记忆改造第1步：读取历史之前校验归属，避免拿到其他用户的会话上下文。
        if row is not None and row.user_id != user_id:
            raise InvalidRequestError("会话不存在或不属于当前用户。")
        if row is None:
            row = SessionRow(session_id=session_id, user_id=user_id)
            if await self._requires_legacy_summary_placeholder():
                now = utcnow()
                await self._session.execute(
                    text(
                        "INSERT INTO sessions (session_id, user_id, created_at, updated_at, summary) "
                        "VALUES (:session_id, :user_id, :now, :now, '')"
                    ),
                    {"session_id": session_id, "user_id": user_id, "now": now},
                )
                row = await self._session.get(SessionRow, session_id)
                assert row is not None
            else:
                self._session.add(row)
                await self._session.flush()
        return row

    async def _requires_legacy_summary_placeholder(self) -> bool:
        if self._session.get_bind().dialect.name != "sqlite":
            return False
        columns = await self._session.execute(text("PRAGMA table_info(sessions)"))
        # Old SQLite schemas had a NOT NULL summary column without a server default.
        return any(column[1] == "summary" and column[3] and column[4] is None for column in columns)

    async def get_memory(self, session_id: str) -> SessionMemoryRow | None:
        return await self._session.get(SessionMemoryRow, session_id)

    async def recent_messages(self, session_id: str, *, limit: int = 20) -> list[ChatMessage]:
        result = await self._session.execute(
            select(MessageRow)
            .where(MessageRow.session_id == session_id)
            .order_by(MessageRow.id.desc())
            .limit(limit)
        )
        rows = list(reversed(result.scalars().all()))
        return [ChatMessage(id=row.id, role=row.role, content=row.content) for row in rows]  # type: ignore[arg-type]

    async def unsummarized_messages(self, session_id: str, *, after_id: int) -> list[ChatMessage]:
        """读取游标后的缓冲历史，不用last-20切掉尚未摘要的数据。"""
        result = await self._session.execute(
            select(MessageRow)
            .where(MessageRow.session_id == session_id, MessageRow.id > after_id)
            .order_by(MessageRow.id)
        )
        return [
            ChatMessage.model_validate({"id": row.id, "role": row.role, "content": row.content})
            for row in result.scalars()
        ]

    async def append_message(self, session_id: str, role: str, content: str) -> None:
        self._session.add(MessageRow(session_id=session_id, role=role, content=content))
        row = await self._session.get(SessionRow, session_id)
        if row is not None:
            row.updated_at = utcnow()

    async def save_summary(self, session_id: str, summary: str, *, upto_message_id: int) -> None:
        boundary = await self._session.get(MessageRow, upto_message_id)
        if (
            boundary is None
            or boundary.session_id != session_id
            or boundary.role != "assistant"
            or not summary.strip()
        ):
            raise InvalidRequestError("摘要边界必须是本会话已保存的完整问答末尾。")
        row = await self.get_memory(session_id)
        if row is None:
            self._session.add(
                SessionMemoryRow(
                    session_id=session_id, summary=summary, summary_upto_message_id=upto_message_id
                )
            )
            return
        if upto_message_id < row.summary_upto_message_id:
            raise InvalidRequestError("摘要进度不能倒退。")
        if upto_message_id == row.summary_upto_message_id:
            if summary != row.summary:
                raise InvalidRequestError("同一摘要进度不能覆盖为不同结果。")
            return
        row.summary = summary
        row.summary_upto_message_id = upto_message_id
        row.updated_at = utcnow()
