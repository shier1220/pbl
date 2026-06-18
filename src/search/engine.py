"""搜索引擎 — DuckDuckGo + Bing 备用"""
import asyncio, logging
from dataclasses import dataclass
from typing import List, Optional
from src.config import SEARCH_MAX_RESULTS, BING_API_KEY
logger = logging.getLogger("course_assistant.search")

@dataclass
class SearchResult:
    title: str; url: str; snippet: str; source: str = "duckduckgo"

class DuckDuckGoEngine:
    def __init__(self): self._ddgs = None
    @property
    def ddgs(self):
        if self._ddgs is None:
            from duckduckgo_search import DDGS
            self._ddgs = DDGS()
        return self._ddgs

    async def search(self, query, max_results=5):
        if self.ddgs is None: return []
        try:
            results = await asyncio.to_thread(lambda: list(self.ddgs.text(query, max_results=max_results)))
            return [SearchResult(title=r.get("title",""), url=r.get("href",""), snippet=r.get("body","")) for r in results]
        except Exception as e: logger.warning("DDG 失败: %s", e); return []

class BingSearchEngine:
    def __init__(self, api_key=""): self.api_key = api_key
    async def search(self, query, max_results=5):
        if not self.api_key: return []
        try:
            import httpx
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            async with httpx.AsyncClient(timeout=10) as cl:
                resp = await cl.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params={"q":query,"count":max_results,"mkt":"zh-CN"})
                resp.raise_for_status()
                return [SearchResult(title=r.get("name",""), url=r.get("url",""), snippet=r.get("snippet",""), source="bing") for r in resp.json().get("webPages",{}).get("value",[])]
        except Exception as e: logger.warning("Bing 失败: %s", e); return []

class WebSearchEngine:
    def __init__(self, bing_api_key=None):
        self.primary = DuckDuckGoEngine()
        self.secondary = BingSearchEngine(bing_api_key or BING_API_KEY)

    async def search(self, query, max_results=None):
        if max_results is None: max_results = SEARCH_MAX_RESULTS
        results = await self.primary.search(query, max_results)
        if results: return results
        results = await self.secondary.search(query, max_results)
        return results
