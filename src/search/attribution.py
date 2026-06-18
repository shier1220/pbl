"""搜索来源格式化"""
from typing import List
from src.search.engine import SearchResult

def format_search_context(results: List[SearchResult]) -> str:
    if not results: return "（无搜索结果）"
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.title}\n    URL: {r.url}\n    {r.snippet}\n")
    return "\n".join(parts)

def format_search_sources(results: List[SearchResult]) -> str:
    if not results: return ""
    lines = ["\n## 🌐 搜索来源"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] [{r.title}]({r.url})")
    return "\n".join(lines)

def format_search_sources_short(results: List[SearchResult]) -> List[str]:
    return [f"[{i}] {r.title} — {r.url}" for i, r in enumerate(results, 1)] if results else []
