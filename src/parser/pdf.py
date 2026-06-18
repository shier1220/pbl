"""PDF 解析器 — 文本 + 表格提取"""
import logging, fitz
from typing import List
from src.parser.base import BaseParser, ParseResult
logger = logging.getLogger("course_assistant.parser.pdf")

class PDFParser(BaseParser):
    @property
    def supported_extensions(self) -> List[str]: return [".pdf", ".PDF"]

    def parse(self, file_path):
        result = ParseResult(); doc = fitz.open(file_path)
        try:
            meta = doc.metadata or {}
            result.metadata = {"title": meta.get("title",""), "author": meta.get("author",""), "pages": doc.page_count}
            all_text, all_tables, pages = [], [], []
            for pn, page in enumerate(doc, 1):
                pt = page.get_text(); all_text.append(pt)
                pages.append({"page": pn, "text": pt[:200]})
                try:
                    found = page.find_tables()
                    if found:
                        for t in found.tables:
                            data = t.extract()
                            if data: all_tables.append(self._table_to_markdown(data))
                except Exception: pass
            result.text = "\n".join(all_text); result.tables = all_tables
            result.metadata["pages"] = pages
            if all_tables: result.text += "\n\n## 表格\n\n" + "\n\n".join(all_tables)
        except Exception as e: result.errors.append(f"PDF error: {e}")
        finally: doc.close()
        return result
