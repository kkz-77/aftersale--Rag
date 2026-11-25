"""Planner that turns intent, skills and complexity into tool plans."""

from __future__ import annotations

import re

from app.agents.state import AgentState
from app.core.exceptions import InvalidRequestError, LLMOutputError, LLMProviderError
from app.core.ids import new_id
from app.memory.context import build_decision_messages
from app.memory.order_reference import order_item_references
from app.prompts.planner import PLANNER_SYSTEM, PLANNER_USER
from app.providers.base import LLMProvider
from app.providers.fake import OfflineLLMProvider
from app.providers.json_parser import parse_model
from app.schemas.agent import AgentPlan, PlanStep
from app.schemas.chat import ActionProposal, ChatMessage
from app.skills.loader import match_skill
from app.tools.permissions import WRITE_RISKS
from app.tools.registry import ToolRegistry

_ORDER_RE = re.compile(r"ORD\d+", re.IGNORECASE)
_ALWAYS_AVAILABLE = ("search_knowledge", "escalate_to_human")


class Planner:
    def __init__(self, llm: LLMProvider, registry: ToolRegistry) -> None:
        self._llm = llm
        self._registry = registry

    def _tool_catalog(self, allowed: set[str] | None = None) -> str:
        lines = []
        for tool in self._registry.all():
            if allowed is not None and tool.name not in allowed:
                continue
            lines.append(f"- {tool.name}: {tool.description} [{tool.risk.value}]")
        return "\n".join(lines) or "- (no tools)"

    # 降级策略：先离线判断，有skill就用skill，没有skill就用知识库
    # 在线的话，llm成功就是plan，不成功，有skill就是skill，没有就是知识库

    async def create_plan(
        self,
        query: str,
        intent_label: str,
        summary: str,
        *,
        history: list[ChatMessage] | None = None,
    ) -> AgentPlan:
        references = order_item_references(query, intent_label, history or [])
        if references and "get_order_items" in self._registry.names():
            if len(references) > 1:
                return AgentPlan(steps=[], rationale="ambiguous_order_items")
            return AgentPlan(
                steps=[
                    PlanStep(
                        step_id=1,
                        goal="查询订单商品明细",
                        tool_name="get_order_items",
                        arguments={"order_id": references[0]},
                    )
                ],
                rationale="order_items_lookup",
            )
        # 先匹配技能，这个skill是写好的步骤规划，匹配的话直接用这个
        skill = match_skill(query, intent_label)
        allowed = _allowed_tools(skill, self._registry)
        # 大模型离线的话，降级策略
        if isinstance(self._llm, OfflineLLMProvider):
            return (
                skill_to_plan(skill, query, self._registry)
                if skill
                else _knowledge_fallback(query, self._registry)
            )
        try:
            response = await self._llm.generate(
                build_decision_messages(
                    system_prompt=PLANNER_SYSTEM.format(
                        tools=self._tool_catalog(allowed), skill_context=_format_skill(skill)
                    ),
                    current_prompt=PLANNER_USER.format(
                        query=query,
                        intent=intent_label,
                        # 摘要由统一组装器按长度选择，不在本轮Prompt中再重复一次。
                        summary="见前面的历史背景" if summary else "无",
                    ),
                    history=history,
                    summary=summary,
                    max_output_tokens=512,
                ),
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            plan = parse_model(response.content, AgentPlan)
            # 清洗llm输出
            plan = _sanitize_plan(plan, self._registry, allowed)
            # 检查清洗之后是不是还有内容
            if plan.steps:
                if skill:
                    plan.rationale = f"skill:{skill['name']};{plan.rationale}"
                return plan
        except (LLMOutputError, LLMProviderError):
            pass
        if skill:
            return skill_to_plan(skill, query, self._registry)
        return _knowledge_fallback(query, self._registry)


def _allowed_tools(skill: dict | None, registry: ToolRegistry) -> set[str] | None:
    if skill is None:
        return None
    names = set(registry.names())
    allowed = {name for name in skill.get("tools", []) if name in names}
    allowed.update(name for name in _ALWAYS_AVAILABLE if name in names)
    return allowed or None


def _format_skill(skill: dict | None) -> str:
    if skill is None:
        return "无（通用工具集）"
    steps = "、".join(skill.get("steps") or [])
    tools = ", ".join(skill.get("tools") or [])
    return f"{skill['name']}：{skill.get('description', '')}。步骤：{steps}。工具：{tools}"


def _sanitize_plan(
    plan: AgentPlan,
    registry: ToolRegistry,
    allowed: set[str] | None,
) -> AgentPlan:
    names = set(registry.names())
    if allowed is not None:
        names = names & allowed
    steps = []
    for step in plan.steps:
        if step.tool_name and step.tool_name.strip().lower() == "none" and not step.needs_retrieval:
            step = step.model_copy(update={"tool_name": None})
        if not step.tool_name or step.tool_name in names:
            steps.append(step)
    return AgentPlan(steps=steps[:6], rationale=plan.rationale)


def skill_to_plan(skill: dict, query: str, registry: ToolRegistry) -> AgentPlan:
    steps: list[PlanStep] = []
    for index, tool_name in enumerate(skill.get("tools", []), start=1):
        if tool_name not in registry.names():
            continue
        steps.append(
            PlanStep(
                step_id=index,
                goal=skill["name"],
                tool_name=tool_name,
                arguments=_tool_arguments(tool_name, query),
                needs_retrieval=tool_name == "search_knowledge",
            )
        )
    if steps:
        return AgentPlan(steps=steps, rationale=f"skill:{skill['name']}")
    return _knowledge_fallback(query, registry)


def _tool_arguments(tool_name: str, query: str) -> dict:
    if tool_name in {"search_knowledge"}:
        return {"query": query[:200]}
    if tool_name in {"search_products"}:
        return {"query": query[:80]}
    if tool_name == "recommend_products":
        return {"query": query[:80]}
    if tool_name in {
        "get_order",
        "get_order_items",
        "track_shipment",
        "cancel_order",
        "create_return_request",
        "create_refund_request",
    }:
        match = _ORDER_RE.search(query)
        if match:
            arguments = {"order_id": match.group(0).upper()}
            if tool_name in {"create_return_request", "create_refund_request"}:
                arguments["reason"] = query[:200]
            return arguments
    return {}


def _knowledge_fallback(query: str, registry: ToolRegistry) -> AgentPlan:
    if "search_knowledge" in registry.names():
        return AgentPlan(
            steps=[
                PlanStep(
                    step_id=1,
                    goal="policy lookup",
                    tool_name="search_knowledge",
                    arguments={"query": query},
                    needs_retrieval=True,
                )
            ],
            rationale="fallback_knowledge",
        )
    return AgentPlan(steps=[], rationale="empty")


async def planner_node(state: AgentState, planner: Planner, *, use_history: bool = True) -> dict:
    confirmed = state.get("confirmed_action")
    if confirmed is not None:
        return {
            "plan": AgentPlan(
                steps=[
                    PlanStep(
                        step_id=1,
                        goal="confirmed action",
                        tool_name=confirmed.tool_name,
                        arguments=confirmed.arguments,
                    )
                ],
                rationale="confirmed_action",
            ),
            "pending_action": None,
        }
    intent = state.get("intent")
    # 怎么生成计划
    # 记忆改造第2步：历史允许模型补工具参数，写授权仍只检查本轮user_query。
    if use_history and (state.get("messages") or state.get("summary")):
        plan = await planner.create_plan(
            state["user_query"],
            intent.label.value if intent else "faq_policy",
            state.get("summary") or "",
            history=list(state.get("messages") or []),
        )
    else:
        plan = await planner.create_plan(
            state["user_query"], intent.label.value if intent else "faq_policy", ""
        )
    if plan.rationale == "ambiguous_order_items" and not plan.steps:
        return {
            "plan": plan,
            "pending_action": None,
            "extra": {
                **(state.get("extra") or {}),
                "missing_order_id": True,
                "ambiguous_order_items": True,
            },
        }
    write_steps = [
        step
        for step in plan.steps
        if step.tool_name and planner._registry.get(step.tool_name).risk in WRITE_RISKS
    ]
    # 这个地方是判断llm规划的每个工具调用是不是必要的，读操作都保留了，写操作的话需要判断用户是否真的要求执行
    if any(
        # 这个函数判断用户是真的在下命令，还是只是在提问
        # 这个外层就是如果任何一个write_step是用户显示要求，为True
        _explicit_write_request(state["user_query"], step.tool_name or "")
        for step in write_steps
    ):
        # 如果用户提问中没有订单号，返回True
        if any(not step.arguments.get("order_id") for step in write_steps):
            return {
                "plan": plan.model_copy(update={"steps": []}),
                # 打上缺失订单号的标记，后续系统会追问订单号
                "extra": {"missing_order_id": True},
            }

    plan = plan.model_copy(
        update={
            "steps": [
                step
                for step in plan.steps
                if step not in write_steps
                or _explicit_write_request(state["user_query"], step.tool_name or "")
            ]
        }
    )
    pending = None
    valid_steps = []
    missing_read_order_id = False
    for step in plan.steps:
        if step.tool_name:
            tool = planner._registry.get(step.tool_name)
            try:
                tool.parse(step.arguments)
            except InvalidRequestError:
                if tool.risk not in WRITE_RISKS and "order_id" in tool.input_model.model_fields:
                    missing_read_order_id = True
                continue
        valid_steps.append(step)
    plan = plan.model_copy(update={"steps": valid_steps})
    if missing_read_order_id and not any(step.tool_name for step in plan.steps):
        return {
            "plan": plan,
            "pending_action": None,
            "extra": {
                **(state.get("extra") or {}),
                "missing_order_id": True,
                "missing_read_order_id": True,
            },
        }
    for step in plan.steps:
        if not step.tool_name:
            continue
        tool = planner._registry.get(step.tool_name)
        if tool.risk in WRITE_RISKS and not state.get("confirm_action_id"):
            pending = ActionProposal(
                action_id=new_id("act_"),
                tool_name=tool.name,
                arguments=step.arguments,
                risk=tool.risk,
                summary=_action_summary(tool.name, step.arguments),
            )
            break
    return {"plan": plan, "pending_action": pending}


# 判断用户是否真的要求执行
def _explicit_write_request(query: str, tool_name: str) -> bool:
    # 用户的问题中有这些关键词，直接就是判定为提问
    if any(
        cue in query
        for cue in (
            "规则",
            "政策",
            "条件",
            "怎么",
            "能不能",
            "是否",
            "可以吗",
            "能退吗",
            "能取消吗",
        )
    ):
        return False
    if tool_name == "cancel_order":
        return any(cue in query for cue in ("取消", "不想要了"))
    if tool_name == "create_refund_request":
        return any(cue in query for cue in ("申请退款", "帮我退款", "我要退款"))
    if tool_name == "create_return_request":
        return any(cue in query for cue in ("申请退货", "帮我退货", "我要退货", "提交退货"))
    return False


# 确认卡片上的摘要文案
def _action_summary(tool_name: str, arguments: dict) -> str:
    order_id = arguments.get("order_id", "该订单")
    if tool_name == "cancel_order":
        return f"确认后将取消订单 {order_id}。请核对订单号。"
    if tool_name == "create_refund_request":
        return f"确认后将为订单 {order_id} 提交退款申请；提交不代表退款已到账。"
    if tool_name == "create_return_request":
        return f"确认后将为订单 {order_id} 提交退货申请；提交不代表退款已到账。"
    return "确认后将提交该操作。"
