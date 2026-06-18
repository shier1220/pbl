"""网页内容提取"""
import logging
logger = logging.getLogger("course_assistant.search.parser")

class SearchResultParser:
    def __init__(self, timeout=10): self.timeout = timeout
    async def extract_content(self, url, max_chars=3000):
        try:
            import httpx
            from bs4 import BeautifulSoup
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as cl:
                resp = await cl.get(url); resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for t in soup(["script","style","nav","footer","header","aside","noscript"]): t.decompose()
                text = "\n".join(l.strip() for l in soup.get_text(separator="\n", strip=True).split("\n") if l.strip())
                return text[:max_chars] + ("..." if len(text) > max_chars else "")
        except Exception as e: logger.debug("提取失败 [%s]: %s", url[:60], e); return ""
