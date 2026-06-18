"""嵌入模型 — 重导出 shim（实现已迁移至 src/embedding.py）"""
from src.embedding import InstructorEmbedding, MODEL_PATH
__all__ = ["InstructorEmbedding", "MODEL_PATH"]
