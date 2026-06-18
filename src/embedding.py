"""本地嵌入模型 — instructor-xl"""
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from src.config import MODEL_PATH, EMBEDDING_DEVICE

class InstructorEmbedding:
    def __init__(self, model_path=MODEL_PATH, device=EMBEDDING_DEVICE):
        self.model = SentenceTransformer(model_path, device=device)
        self.model_path = model_path

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)
