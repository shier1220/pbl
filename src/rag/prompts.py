"""RAG 提示词模板"""
from string import Template

SYSTEM_PROMPT = """你是AIGC导师"小课"，专门辅导AIGC大模型应用知识。

## 关于用户名字（非常重要）
对话历史中如果看到"我是XXX"或"我叫XXX"——这就是用户的名字。

## 回答风格
- 技术问题：准确有深度，优先引用资料
- 闲聊/非资料问题：正常回答即可
- 不知道就说不知道，不编造
- 中文为主，简洁清晰

## AIGC范围
微调实战（LoRA/QLoRA）、模型蒸馏、数据增强、RAG、vLLM部署、深度学习基础
"""

RAG_TEMPLATE = Template("""$system_prompt

$name_line$history_text## 检索到的参考资料
$context

## 用户问题
$question

## 重要规则
1. 当前用户名字：$user_name
2. 技术问题以参考资料为准，资料中有则引用，不编造
3. 回答中引用资料时使用 [1] [2] 格式标注来源编号
4. 资料未覆盖的内容坦率说明
5. 回答末尾列出「📚 参考来源」
6. 回答简洁清晰，中文为主
""")


def build_rag_prompt(system_prompt, question, context, history_text="", user_name="", sources=""):
    return RAG_TEMPLATE.substitute(
        system_prompt=system_prompt,
        name_line=f"当前用户的名字是：{user_name}\n" if user_name else "",
        history_text=f"## 对话历史\n{history_text}\n" if history_text else "",
        context=context, question=question,
        user_name=user_name or "未知",
    )


def build_casual_prompt(system_prompt, question, history_text="", user_name=""):
    return f"{system_prompt}\n\n{f'当前用户的名字是：{user_name}' if user_name else ''}\n\n{history_text}\n\n## 用户消息\n{question}"


def format_sources(docs: list) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        preview = doc.page_content[:100].replace("\n", " ")
        lines.append(f"[{i}] {source} — {preview}...")
    return "\n".join(lines)


def format_context_with_labels(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[{i}] (来源: {source})\n{doc.page_content}")
    return "\n\n".join(parts)
