"""混合检索器 — 稠密 + BM25 + RRF 融合"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import jieba
from rank_bm25 import BM25Okapi

from src.config import RAG_TOP_K, RAG_OVERFETCH, RRF_K, MAX_BM25_DOCS

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
        self.metadatas = metadatas or [{} for _ in range(len(documents))]
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
            if not docs: return
            if len(docs) > MAX_BM25_DOCS:
                logger.warning("BM25 文档数 %d 超出上限 %d，将截断前 %d 条索引", len(docs), MAX_BM25_DOCS, MAX_BM25_DOCS)
                docs = docs[:MAX_BM25_DOCS]
                metas = (data.get("metadatas", []) or [])[:MAX_BM25_DOCS]
                ids = (data.get("ids", []) or [])[:MAX_BM25_DOCS]
            else:
                metas = data.get("metadatas", []) or []
                ids = data.get("ids", []) or []
            self.bm25.build(docs, metas, ids)
        except Exception as e: logger.warning("BM25 构建失败: %s", e)

    def rebuild_bm25_index(self):
        self.bm25 = BM25Index(); self.build_bm25_index(force=True)

    def _dense_search(self, query, top_k):
        """稠密检索 — 直接查 ChromaDB collection 以获取真实 ID"""
        try:
            q_embedding = self.embedding_fn.embed_query(query)
            raw = self.vectorstore._collection.query(
                query_embeddings=[q_embedding], n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            ids = raw.get("ids", [[]])[0]
            docs = raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]
            dists = raw.get("distances", [[]])[0]
            results = []
            for i in range(len(docs)):
                # cosine 空间: distance = 1 - cos_sim → 还原 similarity
                similarity = 1.0 - dists[i] if dists[i] is not None else 0.0
                results.append(RetrievalResult(
                    content=docs[i],
                    metadata=metas[i] or {},
                    dense_score=float(similarity),
                    chroma_id=ids[i] if ids else ""
                ))
            return results
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
