"""意图回退链 — 低置信度时逐层回退"""
import logging
from src.intent.classifier import Intent, IntentResult, EmbeddingIntentClassifier

logger = logging.getLogger("course_assistant.intent.fallback")


class IntentFallbackChain:
    def __init__(self, classifier: EmbeddingIntentClassifier, llm=None):
        self.classifier = classifier
        self.llm = llm

    async def resolve(self, message, history=None, session_id="") -> IntentResult:
        # LLM 主分类（async）
        result = await self.classifier.classify(message, history or [])

        if result.is_high_confidence:
            return result

        # 中/低置信度 → 回退
        return self._fallback(message, result)

    def _fallback(self, message, result) -> IntentResult:
        if self.classifier.has_technical_terms(message):
            return IntentResult(
                intent=Intent.COURSE_QUESTION, confidence=0.4,
                all_scores=result.all_scores,
                threshold_level="low", method="fallback_tech",
            )
        return IntentResult(
            intent=Intent.CASUAL_CHAT, confidence=0.3,
            all_scores=result.all_scores,
            threshold_level="low", method="fallback_default",
        )
