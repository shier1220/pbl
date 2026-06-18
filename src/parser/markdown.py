"""Markdown 解析器"""
import re, logging
from typing import List
from src.parser.base import BaseParser, ParseResult
logger = logging.getLogger("course_assistant.parser.markdown")

class MarkdownParser(BaseParser):
    @property
    def supported_extensions(self) -> List[str]: return [".md", ".MD", ".markdown"]

    def parse(self, file_path):
        result = ParseResult()
        try:
            with open(file_path, "r", encoding="utf-8") as f: content = f.read()
            headings = []
            for m in re.finditer(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE):
                headings.append({"level": len(m.group(1)), "text": m.group(2).strip()})
            result.metadata = {"headings": headings, "title": headings[0]["text"] if headings else ""}
            clean = re.sub(r"```(?:\w+)?\n.*?```", "", content, flags=re.DOTALL)
            clean = re.sub(r"`([^`]+)`", r"\1", clean)
            clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
            clean = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", clean)
            clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
            clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
            clean = re.sub(r"~~([^~]+)~~", r"\1", clean)
            clean = re.sub(r"^[-*_]{3,}\s*$", "", clean, flags=re.MULTILINE)
            result.text = clean.strip()
        except Exception as e: result.errors.append(f"MD error: {e}")
        return result
