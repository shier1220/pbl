"""检索质量评估 — Recall@K, Precision@K, MRR"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RetrievalMetrics:
    recall_at_k: Optional[float] = None
    precision_at_k: Optional[float] = None
    mrr: Optional[float] = None
    hit_rate: Optional[float] = None
    num_retrieved: int = 0
    num_relevant: int = 0
    avg_relevance_score: Optional[float] = None

class RetrievalEvaluator:
    def evaluate(self, query, retrieved_ids, ground_truth_ids, k=5):
        retrieved_k = retrieved_ids[:k]; gt_set = set(ground_truth_ids)
        rel = set(retrieved_k) & gt_set; n_rel = len(rel)
        recall = n_rel / len(gt_set) if gt_set else 0.0
        precision = n_rel / k if k else 0.0
        mrr = 0.0
        for i, did in enumerate(retrieved_k, 1):
            if did in gt_set: mrr = 1.0/i; break
        return RetrievalMetrics(recall_at_k=recall, precision_at_k=precision, mrr=mrr, hit_rate=1.0 if n_rel>0 else 0.0, num_retrieved=len(retrieved_ids), num_relevant=len(gt_set))

    def evaluate_batch(self, queries, retrieved_batch, gt_batch, k=5):
        metrics = [self.evaluate(q, r, g, k) for q, r, g in zip(queries, retrieved_batch, gt_batch)]
        n = len(metrics)
        return {"avg_recall@k": np.mean([m.recall_at_k for m in metrics]) if n else 0, "avg_precision@k": np.mean([m.precision_at_k for m in metrics]) if n else 0, "avg_mrr": np.mean([m.mrr for m in metrics]) if n else 0, "avg_hit_rate": np.mean([m.hit_rate for m in metrics]) if n else 0, "num_queries": n}
