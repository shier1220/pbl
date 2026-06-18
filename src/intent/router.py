"""意图路由器 — 统一入口 + 向后兼容"""
import logging
from typing import Tuple
from src.intent.classifier import Intent, IntentResult, MultiClassIntentClassifier
from src.intent.fallback import IntentFallbackChain
from src.intent.prompts import get_prompt_for_intent
logger = logging.getLogger("course_assistant.intent.router")

class IntentRouter:
    def __init__(self, vectorstore=None, llm=None):
        self.classifier = MultiClassIntentClassifier(vectorstore)
        self.fallback_chain = IntentFallbackChain(self.classifier, llm)
        self.vectorstore = vectorstore

    async def route(self, message, history=None, session_id=""):
        return await self.fallback_chain.resolve(message, history or [], session_id)

    def route_sync(self, message, history=None):
        return self.classifier.classify_with_context(message, history or [])

    # 向后兼容旧版二分类接口
    def route_legacy(self, message, k=3) -> Tuple[str, list]:
        result = self.classifier.classify(message)
        if result.intent == Intent.COURSE_QUESTION:
            if self.vectorstore:
                try: return "rag", self.vectorstore.similarity_search(message, k=k)
                except Exception: pass
            return "rag", []
        return "casual", []

    @staticmethod
    def get_prompt(intent): return get_prompt_for_intent(intent)

    def needs_web_search(self, result, rag_score=0.0):
        return IntentFallbackChain.is_web_search_needed(result, rag_score)
