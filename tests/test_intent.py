"""意图分类器测试"""
import pytest
from src.intent.classifier import Intent, IntentResult, EmbeddingIntentClassifier


class TestIntentEnum:
    def test_all_intents(self):
        assert len(Intent) == 5
        assert Intent.COURSE_QUESTION.value == "course_question"
        assert Intent.CASUAL_CHAT.value == "casual_chat"
        assert Intent.WEB_SEARCH.value == "web_search"
        assert Intent.FILE_OPERATION.value == "file_operation"
        assert Intent.SYSTEM_COMMAND.value == "system_command"


class TestIntentResult:
    def test_high_confidence(self):
        r = IntentResult(intent=Intent.CASUAL_CHAT, confidence=0.9, method="knn")
        assert r.is_high_confidence
        assert not r.is_medium_confidence
        assert not r.is_low_confidence

    def test_medium_confidence(self):
        r = IntentResult(intent=Intent.WEB_SEARCH, confidence=0.5, method="knn")
        assert not r.is_high_confidence
        assert r.is_medium_confidence

    def test_low_confidence(self):
        r = IntentResult(intent=Intent.COURSE_QUESTION, confidence=0.2, method="knn")
        assert r.is_low_confidence


class TestEmbeddingIntentClassifier:
    @pytest.mark.asyncio
    async def test_fallback_when_no_collection(self):
        """无嵌入/LLM 时回退到 casual_chat"""
        c = EmbeddingIntentClassifier()  # 无 embedding_fn, 无 llm
        result = await c.classify("你好")
        assert result.intent == Intent.CASUAL_CHAT
        assert result.confidence == 0.5
        assert result.method == "fallback"

    def test_sync_classify(self):
        """同步版本"""
        c = EmbeddingIntentClassifier()
        result = c.classify_sync("你好")
        assert result.intent == Intent.CASUAL_CHAT

    def test_has_technical_terms(self):
        assert EmbeddingIntentClassifier.has_technical_terms("什么是LoRA微调")
        assert EmbeddingIntentClassifier.has_technical_terms("大模型部署方案")
        assert not EmbeddingIntentClassifier.has_technical_terms("你好")
        assert not EmbeddingIntentClassifier.has_technical_terms("今天天气怎么样")
