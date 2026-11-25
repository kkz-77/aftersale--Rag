"""Answer generation and refusal nodes."""

from __future__ import annotations

import json
import logging
import re
import time

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.state import AgentState
from app.core.enums import RetrievalStrategy, ToolRisk
from app.core.exceptions import LLMOutputError, LLMProviderError
from app.memory.context import build_answer_messages
from app.prompts.answer import ANSWER_REPAIR_USER, ANSWER_SYSTEM, ANSWER_USER
from app.providers.base import LLMProvider
from app.providers.fake import OfflineLLMProvider
from app.providers.json_parser import parse_model
from app.schemas.agent import ToolResult
from app.schemas.common import LatencyBreakdown, TokenUsage
from app.schemas.retrieval import Citation

logger = logging.getLogger(__name__)


class AnswerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str = Field(min_length=1)
    citation_chunk_ids: list[str] = Field(default_factory=list)
    refused: bool = False

    @field_validator("answer")
    @classmethod
    def nonblank_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must not be blank")
        return value


async def generate_node(
    state: AgentState,
    llm: LLMProvider,
    *,
    token_budget: int = 128_000,
    use_history: bool = True,
) -> dict:
    started = time.perf_counter()
    chunks = state.get("retrieved_chunks") or []
    tools = state.get("tool_results") or []
    # tools必须存在，如果等待确认可能也是空
    if tools and all(not tool.success for tool in tools):
        return {
            "final_answer": "暂时无法查到相关业务数据，请核对订单号或稍后重试。",
            "citations": [],
        }
    # 获取确认的状态，如果是待确认就让用户提供订单号
    pending = state.get("pending_action")
    if (state.get("extra") or {}).get("missing_order_id"):
        if (state.get("extra") or {}).get("ambiguous_order_items"):
            return {
                "final_answer": "刚才提到了多笔订单，请告诉我你想查询哪一笔的商品。",
                "citations": [],
            }
        if (state.get("extra") or {}).get("missing_read_order_id"):
            return {"final_answer": "请提供要查询的订单号。", "citations": []}
        return {"final_answer": "请提供要处理的订单号，确认后我才能继续。", "citations": []}
    # 如果已经确认，就返回确认信息
    if pending:
        answer = pending.summary
        latency = state.get("latency") or LatencyBreakdown()
        latency.llm_ms = (time.perf_counter() - started) * 1000
        return {
            "final_answer": answer,
            "citations": [],
            "refused": False,
            "latency": latency,
        }
    # 这个就是吧检索到的chunks拼接起来，方便给llm当做提示词
    evidence = (
        "\n\n".join(f"[{c.chunk_id}] {c.title}\n{c.content}" for c in chunks[:6]) or "(无检索结果)"
    )
    # model_dump是将ToolResult转成字典，json.dumps是转成json字符串
    tool_text = json.dumps([t.model_dump() for t in tools], ensure_ascii=False) if tools else "(无)"
    intent = state.get("intent")
    strategy = state.get("strategy") or RetrievalStrategy.DIRECT
    if isinstance(llm, OfflineLLMProvider):
        answer, citation_ids = _offline_answer(state)
        citations = [
            Citation(
                document_id=chunk.document_id,
                title=chunk.title,
                chunk_id=chunk.chunk_id,
                section=chunk.section,
            )
            for chunk in chunks
            if chunk.chunk_id in citation_ids
        ]
        return {"final_answer": answer, "citations": citations, "refused": False}
    # 记忆改造第1步：历史实际进入模型输入；当前问题只在最后加入一次。
    complexity = state.get("complexity")
    system_prompt = ANSWER_SYSTEM
    history_recall = complexity and complexity.rationale == "history_recall_only"
    if history_recall:
        system_prompt += (
            "\n本轮只复述对话。历史原文和历史摘要可用于回答用户之前说了什么，"
            "包括用户更正后的值与待办。以明确更正后的信息为准，勿推断实时状态。"
            "回顾待办时，写出用户要核对的具体事项和已知细节；"
            "助手此前建议提供的材料或下一步，只是建议，不能替代用户的待办目标。"
        )
    current_prompt = ANSWER_USER.format(
        query=state["user_query"],
        intent=intent.label.value if intent else "unknown",
        strategy=strategy.value,
        evidence=evidence,
        tool_results=tool_text,
    )
    if history_recall:
        current_prompt += (
            "\n如果用户问自己的待办，从历史的‘当前诉求’或用户原话提取要完成的目标，"
            "逐项写出其中明确的对象、动作和待核对的具体属性。"
            "‘待确认操作’里助手索要的材料与核对目标不同，只回顾待办时省略这类建议。"
        )
    messages = build_answer_messages(
        system_prompt=system_prompt,
        current_prompt=current_prompt,
        history=list(state.get("messages") or []) if use_history else [],
        summary=(state.get("summary") or "") if use_history else "",
        token_budget=token_budget,
        max_output_tokens=700,
    )
    usage = TokenUsage()
    payload = None
    # 最多一次初始生成和一次格式修复；API失败由Provider处理，不在这里重复发起。
    for attempt in range(2):
        phase = "generate" if attempt == 0 else "format_repair"
        try:
            response = await llm.generate(
                messages,
                temperature=0.2 if attempt == 0 else 0.0,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
        except LLMProviderError:
            logger.warning(
                "answer_api_failed", extra={"trace_id": state.get("trace_id"), "phase": phase}
            )
            break
        # 请求成功返回就累计usage，之后解析失败不能将这次消耗归零。
        usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens + response.usage.prompt_tokens,
            completion_tokens=usage.completion_tokens + response.usage.completion_tokens,
            total_tokens=usage.total_tokens + response.usage.total_tokens,
        )
        try:
            # 通用parser会从数组中提取对象；Generate明确禁止顶层数组等非对象JSON。
            try:
                text = response.content.strip()
                fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
                root = json.loads(fence.group(1) if fence else text)
            except json.JSONDecodeError:
                root = None  # 仍允许通用parser处理Markdown围栏等兼容格式。
            if root is not None and not isinstance(root, dict):
                raise LLMOutputError(
                    "Final answer JSON root must be an object, not an array or scalar"
                )
            payload = parse_model(response.content, AnswerPayload)
            break
        except LLMOutputError as exc:
            logger.warning(
                "answer_schema_failed", extra={"trace_id": state.get("trace_id"), "phase": phase}
            )
            if attempt == 0:
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": ANSWER_REPAIR_USER.format(
                            schema=json.dumps(
                                AnswerPayload.model_json_schema(), ensure_ascii=False
                            ),
                            error=json.dumps(str(exc), ensure_ascii=False),
                            raw_output=json.dumps(response.content, ensure_ascii=False),
                        ),
                    },
                ]
    if payload is None:
        # 与离线回答共享只读工具格式化，不能从失败输出或写工具中推测执行结果。
        tool_answer = _readonly_tool_answer(tools)
        payload = AnswerPayload(
            answer=tool_answer or _extractive_fallback(state["user_query"], evidence, tool_text),
            citation_chunk_ids=[chunks[0].chunk_id] if chunks and not tool_answer else [],
        )
    citations = [
        Citation(
            document_id=chunk.document_id,
            title=chunk.title,
            chunk_id=chunk.chunk_id,
            section=chunk.section,
        )
        for chunk in chunks
        if chunk.chunk_id in set(payload.citation_chunk_ids)
    ]
    latency = state.get("latency") or LatencyBreakdown()
    latency.llm_ms = (time.perf_counter() - started) * 1000
    prev_usage = state.get("token_usage") or TokenUsage()
    merged = TokenUsage(
        prompt_tokens=prev_usage.prompt_tokens + usage.prompt_tokens,
        completion_tokens=prev_usage.completion_tokens + usage.completion_tokens,
        total_tokens=prev_usage.total_tokens + usage.total_tokens,
    )
    return {
        "final_answer": payload.answer,
        "citations": citations,
        "refused": payload.refused,
        "latency": latency,
        "token_usage": merged,
    }


async def refuse_node(state: AgentState) -> dict:
    return {
        "final_answer": "这个问题超出电商客服范围，我无法协助。如需购物、订单、物流或售后帮助，请换个问题。",
        "refused": True,
        "citations": [],
    }


def _extractive_fallback(query: str, evidence: str, tool_text: str) -> str:
    if evidence and evidence != "(无检索结果)":
        snippet = evidence.split("\n", 1)[-1][:240].strip()
        return f"暂时无法生成完整答复。相关资料节选：\n\n> {snippet}\n\n请以原文为准。"
    if tool_text and tool_text != "(无)":
        return "现有业务数据不足以可靠地整理答复。请核对信息、稍后重试或转人工。"
    return "暂时无法基于现有信息给出可靠答复，请补充相关信息或转人工。"


def _offline_answer(state: AgentState) -> tuple[str, set[str]]:
    """Use only retrieved text or whitelisted tool fields when no LLM is configured."""
    strategy = state.get("strategy")
    if strategy == RetrievalStrategy.DIRECT:
        return "你好，我可以帮你查看商品政策、订单、物流和售后问题。", set()

    tool_answer = _readonly_tool_answer(state.get("tool_results") or [])
    if tool_answer:
        return tool_answer, set()

    for result in reversed(state.get("tool_results") or []):
        if not result.success:
            continue
        data = result.output
        if result.name == "cancel_order":
            return f"订单 {data.get('order_id', '')} 已取消。", set()
        if result.name == "create_refund_request":
            return f"订单 {data.get('order_id', '')} 的退款申请已提交，尚不代表款项到账。", set()
        if result.name == "create_return_request":
            return f"订单 {data.get('order_id', '')} 的退货申请已提交，尚不代表退款完成。", set()
        if result.name == "get_refund_status":
            refunds = data.get("refunds") or []
            if not refunds:
                return "这笔订单暂未查询到退款申请记录。", set()
            first = refunds[0]
            if isinstance(first, dict):
                return (
                    f"查询到退款单 {first.get('refund_id', '')}，当前状态：{first.get('status', '未知')}。",
                    set(),
                )
        if result.name == "search_knowledge":
            found = data.get("chunks") or []
            if found and isinstance(found[0], dict):
                item = found[0]
                title = str(item.get("title") or "相关政策")
                excerpt = str(item.get("content") or "").strip()[:260]
                return (
                    f"当前未配置生成模型。找到《{title}》中的资料节选，请核对原文：\n\n> {excerpt}",
                    set(),
                )

    chunks = state.get("retrieved_chunks") or []
    if chunks:
        chunk = chunks[0]
        excerpt = chunk.content.strip()[:260]
        return (
            f"当前未配置生成模型，以下是《{chunk.title}》的资料节选，请核对原文：\n\n> {excerpt}",
            {chunk.chunk_id},
        )
    return (
        "当前无法可靠回答这个问题。请补充相关信息，或联系人工客服。",
        set(),
    )


def _readonly_tool_answer(tools: list[ToolResult]) -> str | None:
    """Conservative shared formatter: supported successful read results only.

    不推断时效/金额/售后资格。混合失败、写工具、多订单或同工具矛盾结果均不自动选取。
    """
    supported = {"get_order", "track_shipment", "get_order_items"}
    if not tools or any(
        not t.success or t.name not in supported or t.risk != ToolRisk.READ or t.truncated
        for t in tools
    ):
        return None
    order_ids = {
        t.output.get("order_id") for t in tools if isinstance(t.output.get("order_id"), str)
    }
    # 商品工具返回items但无order_id；和其他工具混合时无法可靠关联，保守降级。
    if len(order_ids) > 1 or (len(tools) > 1 and any(t.name == "get_order_items" for t in tools)):
        return None
    by_name: dict[str, dict] = {}
    for tool in tools:
        if tool.name in by_name and by_name[tool.name] != tool.output:
            return None
        by_name[tool.name] = tool.output
    answers = []
    for name, data in by_name.items():
        order_id = data.get("order_id")
        if name in {"get_order", "track_shipment"} and (
            not isinstance(order_id, str) or not order_id.strip()
        ):
            return None
        if name == "get_order":
            labels = {
                "created": "待支付",
                "paid": "已支付",
                "shipped": "已发货",
                "delivered": "已签收",
                "cancelled": "已取消",
                "refunding": "退款中",
                "refunded": "已退款",
            }
            status = data.get("status")
            if not isinstance(status, str) or status not in labels:
                return None
            answers.append(f"订单 {order_id} 当前状态：{labels[status]}。")
        elif name == "track_shipment":
            labels = {"warehouse": "仓库处理中", "in_transit": "运输中", "delivered": "已签收"}
            events = data.get("events")
            if (
                not isinstance(events, list)
                or not events
                or any(
                    not isinstance(e, dict)
                    or not isinstance(e.get("status"), str)
                    or e["status"] not in labels
                    for e in events
                )
            ):
                return None
            answers.append(f"订单 {order_id} 的物流状态：{labels[events[-1]['status']]}。")
        elif name == "get_order_items":
            items = data.get("items")
            if (
                not isinstance(items, list)
                or not items
                or any(
                    not isinstance(item, dict)
                    or not isinstance(item.get("name"), str)
                    or not item["name"].strip()
                    or type(item.get("quantity")) is not int
                    or item["quantity"] <= 0
                    for item in items
                )
            ):
                return None
            details = "；".join(f"{item['name']} × {item['quantity']}" for item in items[:5])
            answers.append(f"查询到的订单商品（共{len(items)}项）：{details}。")
    return "\n".join(answers) or None
