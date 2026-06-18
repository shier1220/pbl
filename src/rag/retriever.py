"""混合检索器 — 稠密 + BM25 + RRF 融合"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import jieba
from rank_bm25 import BM25Okapi

from src.config import RAG_TOP_K, RAG_OVERFETCH, RRF_K

logger = logging.getLogger("course_assistant.retriever")


@dataclass
class RetrievalResult:
    content: str; metadata: dict = field(default_factory=dict)
    dense_score: Optional[float] = None; sparse_score: Optional[float] = None
    fused_score: Optional[float] = None; chroma_id: Optional[str] = None


class BM25Index:
    """BM25 稀疏检索 + jieba 分词"""
    def __init__(self):
        self.corpus = []; self.metadatas = []; self.chroma_ids = []; self._index = None

    @staticmethod
    def tokenize(text): return [t.strip() for t in jieba.lcut(text) if t.strip()]

    def build(self, documents, metadatas=None, ids=None):
        self.corpus = list(documents)
        self.metadatas = metadatas or [{}] * len(documents)
        self.chroma_ids = ids or [str(i) for i in range(len(documents))]
        tokenized = [self.tokenize(d) for d in self.corpus]
        if tokenized and any(tokenized): self._index = BM25Okapi(tokenized)

    def search(self, query, top_k=10):
        if self._index is None: return []
        scores = self._index.get_scores(self.tokenize(query))
        if len(scores) == 0: return []
        results = []
        for idx in np.argsort(scores)[::-1][:top_k]:
            if scores[idx] <= 0: continue
            results.append(RetrievalResult(content=self.corpus[idx], metadata=self.metadatas[idx], sparse_score=float(scores[idx]), chroma_id=self.chroma_ids[idx]))
        return results

    @property
    def is_empty(self): return self._index is None or len(self.corpus) == 0

    @property
    def doc_count(self): return len(self.corpus)


class HybridRetriever:
    """稠密 + 稀疏 + RRF 融合"""
    def __init__(self, vectorstore, embedding_fn):
        self.vectorstore = vectorstore; self.embedding_fn = embedding_fn
        self.bm25 = BM25Index(); self.rrf_k = RRF_K

    def build_bm25_index(self, force=False):
        if not force and not self.bm25.is_empty: return
        try:
            data = self.vectorstore._collection.get(include=["documents","metadatas"])
            docs = data.get("documents", [])
            if docs: self.bm25.build(docs, data.get("metadatas", []), data.get("ids", []))
        except Exception as e: logger.warning("BM25 构建失败: %s", e)

    def rebuild_bm25_index(self):
        self.bm25 = BM25Index(); self.build_bm25_index(force=True)

    def _dense_search(self, query, top_k):
        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
            return [RetrievalResult(content=doc.page_content, metadata=doc.metadata, dense_score=float(s), chroma_id=doc.metadata.get("id","")) for doc, s in results]
        except Exception: return []

    def _rrf_fuse(self, dense, sparse, top_k):
        doc_map = {}; rrf_scores = {}
        for rank, r in enumerate(dense, 1):
            key = r.chroma_id or r.content[:100]; doc_map[key] = r
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (self.rrf_k + rank)
        for rank, r in enumerate(sparse, 1):
            key = r.chroma_id or r.content[:100]
            if key in doc_map: doc_map[key].sparse_score = r.sparse_score
            else: doc_map[key] = r
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (self.rrf_k + rank)
        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        fused = []
        for key in sorted_keys:
            r = doc_map[key]; r.fused_score = rrf_scores[key]; fused.append(r)
        return fused

    def retrieve(self, query, top_k=None):
        top_k = top_k or RAG_TOP_K; overfetch = top_k * 2
        dense = self._dense_search(query, overfetch)
        sparse = self.bm25.search(query, overfetch) if not self.bm25.is_empty else []
        if sparse: return self._rrf_fuse(dense, sparse, top_k)
        return dense[:top_k]

    @property
    def stats(self): return {"bm25_docs": self.bm25.doc_count, "fusion": f"RRF(k={self.rrf_k})"}
