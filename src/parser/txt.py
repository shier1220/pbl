"""TXT 解析器"""
import logging
from typing import List
from src.parser.base import BaseParser, ParseResult
logger = logging.getLogger("course_assistant.parser.txt")

class TXTParser(BaseParser):
    @property
    def supported_extensions(self) -> List[str]: return [".txt", ".TXT", ".text"]

    def parse(self, file_path):
        result = ParseResult()
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            result.text = text.strip()
            result.metadata = {"size": len(text), "lines": text.count("\n") + 1}
        except Exception as e: result.errors.append(f"TXT error: {e}")
        return result
