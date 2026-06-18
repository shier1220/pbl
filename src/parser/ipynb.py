"""IPYNB 解析器"""
import json, logging
from typing import List
from src.parser.base import BaseParser, ParseResult
logger = logging.getLogger("course_assistant.parser.ipynb")

class IPYNBParser(BaseParser):
    @property
    def supported_extensions(self) -> List[str]: return [".ipynb", ".IPYNB"]

    def parse(self, file_path):
        result = ParseResult()
        try:
            with open(file_path, "r", encoding="utf-8") as f: nb = json.load(f)
            texts = []; counts = {"markdown": 0, "code": 0}
            for cell in nb.get("cells", []):
                ct = cell.get("cell_type",""); src = cell.get("source",[])
                st = "".join(src).strip() if isinstance(src, list) else str(src).strip()
                if not st: continue
                if ct == "markdown": counts["markdown"] += 1; texts.append(st)
                elif ct == "code": counts["code"] += 1; texts.append(f"```python\n{st}\n```")
            result.text = "\n\n".join(texts)
            result.metadata = {"cells": counts}
        except Exception as e: result.errors.append(f"IPYNB error: {e}")
        return result
