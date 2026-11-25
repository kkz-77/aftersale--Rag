"""Task-local API attempt accounting; inactive outside evaluation."""

from contextvars import ContextVar

current_call_trace: ContextVar[list[dict] | None] = ContextVar("llm_call_trace", default=None)
