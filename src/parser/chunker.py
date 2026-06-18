"""智能分块器 — 按文档类型采用不同策略"""
import logging
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS
logger = logging.getLogger("course_assistant.parser.chunker")

class DocumentChunker:
    def __init__(self):
        self.generic = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=CHUNK_SEPARATORS)
        self.large = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n","\n","。","！","？","；","，"," ",""])

    def chunk(self, result, file_type, source_name):
        if not result.text.strip(): return []
        if file_type in ('.pdf',): chunks = self._chunk_pdf(result, source_name)
        elif file_type in ('.md','.markdown'): chunks = self._chunk_md(result, source_name)
        elif file_type in ('.csv',): chunks = self._chunk_csv(result, source_name)
        elif file_type in ('.docx',): chunks = self._chunk_docx(result, source_name)
        else: chunks = self._chunk_generic(result, source_name)
        hp = self._heading_path(result.metadata.get("headings", []))
        for i, c in enumerate(chunks):
            c["metadata"].update({"chunk_index": i, "heading_path": hp, "document_title": result.metadata.get("title", source_name)})
            c["metadata"].setdefault("page_number", None); c["metadata"].setdefault("section", None)
        return chunks

    def _chunk_pdf(self, r, src):
        pages = r.metadata.get("pages", [])
        if not pages: return self._chunk_generic(r, src)
        chunks = []
        for p in pages:
            pt = p["text"]
            if len(pt) <= CHUNK_SIZE * 1.5: chunks.append({"text": pt, "metadata": {"source": src, "page_number": p["page"]}})
            else:
                for sub in self.large.split_text(pt): chunks.append({"text": sub, "metadata": {"source": src, "page_number": p["page"]}})
        return chunks

    def _chunk_md(self, r, src):
        headings = r.metadata.get("headings", [])
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n# ","\n## ","\n### ","\n\n","\n","。"])
        return [{"text": t, "metadata": {"source": src, "section": self._find_heading(t, headings)}} for t in splitter.split_text(r.text)]

    def _chunk_csv(self, r, src):
        cols = r.metadata.get("columns", [])
        desc = f"CSV: {src}\n列名: {', '.join(cols)}\n行数: {r.metadata.get('num_rows', 0)}"
        chunks = [{"text": desc, "metadata": {"source": src, "section": "概览"}}]
        if r.tables:
            t = r.tables[0]
            if len(t) > CHUNK_SIZE:
                for sub in self.generic.split_text(t): chunks.append({"text": sub, "metadata": {"source": src, "section": "数据"}})
            else: chunks.append({"text": t, "metadata": {"source": src, "section": "数据"}})
        return chunks

    def _chunk_docx(self, r, src):
        headings = r.metadata.get("headings", [])
        return [{"text": t, "metadata": {"source": src, "section": self._find_heading(t, headings)}} for t in self.generic.split_text(r.text)]

    def _chunk_generic(self, r, src):
        return [{"text": t, "metadata": {"source": src}} for t in self.generic.split_text(r.text)]

    @staticmethod
    def _find_heading(text, headings): return next((h["text"] for h in (headings or []) if h["text"] in text), "")

    @staticmethod
    def _heading_path(headings): return " > ".join(h["text"] for h in (headings or [])[:3])
