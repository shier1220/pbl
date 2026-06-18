"""HTML 解析器"""
import logging
from typing import List
from bs4 import BeautifulSoup
from src.parser.base import BaseParser, ParseResult
logger = logging.getLogger("course_assistant.parser.html")

class HTMLParser(BaseParser):
    @property
    def supported_extensions(self) -> List[str]: return [".html", ".HTML", ".htm"]

    def parse(self, file_path):
        result = ParseResult()
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            for t in soup(["script","style","nav","footer","header","aside","noscript"]): t.decompose()
            headings = []
            for lv in range(1, 7):
                for h in soup.find_all(f"h{lv}"):
                    txt = h.get_text(strip=True)
                    if txt: headings.append({"level": lv, "text": txt})
            title = soup.find("title")
            result.metadata = {"title": title.get_text(strip=True) if title else "", "headings": headings}
            result.text = soup.get_text(separator="\n", strip=True)
        except Exception as e: result.errors.append(f"HTML error: {e}")
        return result
