"""意图路由器 — 统一入口"""
import logging
from typing import Tuple
from src.intent.classifier import Intent, IntentResult, EmbeddingIntentClassifier
from src.intent.fallback import IntentFallbackChain
from src.intent.prompts import get_prompt_for_intent

logger = logging.getLogger("course_assistant.intent.router")


class IntentRouter:
    def __init__(self, vectorstore=None, llm=None, embedding_fn=None):
        self.classifier = EmbeddingIntentClassifier(embedding_fn, llm)
        self.fallback_chain = IntentFallbackChain(self.classifier, llm)
        self.vectorstore = vectorstore

    async def route(self, message, history=None, session_id="") -> IntentResult:
        return await self.fallback_chain.resolve(message, history or [], session_id)

    def route_sync(self, message, history=None) -> IntentResult:
        return self.classifier.classify_with_context(message, history or [])

    @staticmethod
    def get_prompt(intent):
        return get_prompt_for_intent(intent)
