"""Resolve read-only questions about the items in an identified order."""

import re
from collections.abc import Sequence

from app.schemas.chat import ChatMessage

_ORDER_ID = re.compile(r"(?<![a-z0-9_])ORD[-_]?\d{4,}(?![a-z0-9_])", re.I | re.ASCII)
_ITEM_REFERENCE = re.compile(
    r"(?:这|那)(?:个|件|款)(?:商品|产品|东西)?(?:到底)?(?:是什么|是哪款|是什么型号)"
)
_ORDER_ITEMS = re.compile(
    r"(?:请|帮我|能帮我)?(?:查询|查一下|查查|查|看看|看一下)?"
    r"(?:刚才|刚刚|上面)?(?:这笔订单|那笔订单|这单|那单|订单<id>|<id>)"
    r"(?:里|里面|中|的)?(?:有(?:哪[些几]|什么)(?:个|件)?(?:商品|产品|东西)|"
    r"(?:商品|产品)明细|(?:买了|都买了)什么|(?:商品|产品)是(?:什么|哪款))"
)


def order_item_references(
    query: str,
    intent_label: str,
    history: Sequence[ChatMessage] = (),
) -> tuple[str, ...] | None:
    """Return candidates for a scoped item lookup, or None for other requests."""
    if intent_label not in {"product_consult", "order_inquiry"}:
        return None
    text = re.sub(r"\s+", "", query).strip("？?。！!")
    has_explicit_id = bool(_ORDER_ID.search(query))
    normalized = _ORDER_ID.sub("<id>", text)
    if not (_ITEM_REFERENCE.fullmatch(text) or _ORDER_ITEMS.fullmatch(normalized)):
        return None
    if has_explicit_id:
        return tuple(dict.fromkeys(match.group().upper() for match in _ORDER_ID.finditer(query)))
    # User-authored IDs establish the reference. Assistant output alone can expose an unrelated ID.
    return tuple(
        dict.fromkeys(
            match.group().upper()
            for message in history
            if message.role == "user"
            for match in _ORDER_ID.finditer(message.content)
        )
    )
