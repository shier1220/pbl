"""
本地嵌入模型封装 — instructor-xl
兼容 LangChain Embeddings 接口 + ChromaDB 嵌入函数
"""

import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer


# 优先使用项目目录下的本地模型，避免 HuggingFace 下载
import os
_LOCAL_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instructor-xl")
MODEL_PATH = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else "hkunlp/instructor-xl"

class InstructorEmbedding:
    """LangChain 兼容的嵌入函数包装器"""

    def __init__(self, model_path: str = MODEL_PATH, device: str = "cpu"):
        self.model = SentenceTransformer(model_path, device=device)
        self.model_path = model_path

    def embed_query(self, text: str) -> List[float]:
        """单行查询嵌入（LangChain 标准接口）"""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档嵌入（LangChain 标准接口）"""
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB 嵌入函数调用接口（多文本）"""
        return self.embed_documents(input)
