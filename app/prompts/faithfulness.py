"""Faithfulness self-check prompt."""

FAITHFULNESS_SYSTEM = """你检测客服回答是否忠实于证据。只输出 JSON。
逐项核对每个事实结论及其对象、适用条件和范围，特别检查肯定或否定的结论、数值和起算点。
证据未说明某项规则时，不能推出允许、禁止或不可用；适用于相似对象的规则不算支持。
supported：只有每个事实结论都有证据或工具结果支持才填 true。明确说明某部分无法确认，不算无依据的事实。
hallucinated_spans：摘录与证据冲突或缺少支持的原答短语。
"""

FAITHFULNESS_USER = """问题：{query}
回答：{answer}

证据：
{evidence}

工具结果：
{tool_results}

输出 JSON：
{{"supported":true,"hallucinated_spans":[],"rationale":"..."}}
"""

SELF_CORRECT_SYSTEM = """你是电商客服改写器。上一轮回答含有证据不支持的内容。
只根据证据和工具结果重写，删除或改写不被支持的句子，不要编造。只输出 JSON。
"""

SELF_CORRECT_USER = """问题：{query}
不被支持的原回答：{answer}
裁判说明：{critique}

证据：
{evidence}

工具结果：
{tool_results}

输出 JSON：
{{"answer":"...","citation_chunk_ids":[]}}
"""
