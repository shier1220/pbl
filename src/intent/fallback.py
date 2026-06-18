"""意图回退链"""
import logging
from src.intent.classifier import Intent, IntentResult
logger = logging.getLogger("course_assistant.intent.fallback")

VERIFY_PROMPT = """判断用户消息属于哪个意图：course_question / casual_chat / file_operation / web_search / system_command
消息："{message}"
当前预测：{predicted}
只回复一个类别名称。"""

class IntentFallbackChain:
    def __init__(self, classifier, llm=None): self.classifier = classifier; self.llm = llm

    async def resolve(self, message, history=None, session_id=""):
        result = self.classifier.classify_with_context(message, history or [])
        if result.is_high_confidence: return result
        if result.is_medium_confidence and self.llm:
            verified = await self._llm_verify(message, result)
            if verified: return verified
        return self._fallback(message, result)

    async def _llm_verify(self, message, result):
        import asyncio
        try:
            resp = await asyncio.to_thread(self.llm.invoke, VERIFY_PROMPT.format(message=message, predicted=result.intent.value))
            answer = resp.content.strip().lower()
            for intent in Intent:
                if intent.value in answer:
                    conf = 0.75 if intent == result.intent else 0.65
                    return IntentResult(intent=intent, confidence=conf, all_scores=result.all_scores, threshold_level="high", method="llm_verify")
        except Exception as e: logger.warning("LLM 验证失败: %s", e)
        return None

    def _fallback(self, message, result):
        if self.classifier.has_technical_terms(message):
            return IntentResult(intent=Intent.COURSE_QUESTION, confidence=0.4, all_scores=result.all_scores, threshold_level="low", method="fallback_tech")
        return IntentResult(intent=Intent.CASUAL_CHAT, confidence=0.3, all_scores=result.all_scores, threshold_level="low", method="fallback_default")

    @staticmethod
    def is_web_search_needed(result, rag_score=0.0):
        return result.intent == Intent.WEB_SEARCH or (result.intent == Intent.COURSE_QUESTION and rag_score < 0.35)
