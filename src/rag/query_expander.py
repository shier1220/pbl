"""查询扩展与压缩"""
import logging
from typing import List
logger = logging.getLogger("course_assistant.query")

EXPAND_PROMPT = """生成 {n_variants} 个与原问题语义相同但表述不同的变体，每个一行，直接输出。
原问题：{query}
变体："""

CONDENSE_PROMPT = """基于对话历史，将最新问题重写为独立的文档检索查询。
## 历史\n{history}
## 最新问题\n{query}
## 独立查询："""

class QueryExpander:
    def __init__(self, llm): self.llm = llm
    async def expand(self, query, n_variants=3):
        import asyncio
        try:
            resp = await asyncio.to_thread(self.llm.invoke, EXPAND_PROMPT.format(query=query, n_variants=n_variants))
            variants = [l.strip("- 1234567890.、) ") for l in resp.content.split("\n") if l.strip() and len(l.strip())>1]
            seen = {query}
            result = [query]
            for v in variants:
                if v not in seen: seen.add(v); result.append(v)
            return result[:n_variants+1]
        except Exception: return [query]

class QueryCondenser:
    def __init__(self, llm): self.llm = llm
    async def condense(self, query, history, max_hist=6):
        if not history: return query
        import asyncio
        hist_text = ""
        for m in history[-max_hist:]: hist_text += f"{'用户' if m['role']=='user' else '助手'}：{m['content']}\n"
        if not hist_text.strip(): return query
        try:
            resp = await asyncio.to_thread(self.llm.invoke, CONDENSE_PROMPT.format(history=hist_text, query=query))
            condensed = resp.content.strip()
            return condensed if condensed and len(condensed) >= 3 else query
        except Exception: return query
