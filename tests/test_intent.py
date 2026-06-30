"""意图分类器测试"""
import pytest
from src.intent.classifier import Intent, IntentResult, EmbeddingIntentClassifier
from src.intent.complexity import (
    ComplexityAnalyzer, TaskPlanner, TaskPlan, TaskStep,
    ComplexityLevel, ComplexityResult,
)
from src.intent.followup import FollowUpRecommender


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


class TestComplexityAnalyzer:
    """复杂度分析器 — 规则 + LLM 兜底"""

    def test_simple_greeting(self):
        """问候 → 简单"""
        ca = ComplexityAnalyzer(llm=None)
        result = ca.analyze("你好")
        assert result.level == ComplexityLevel.SIMPLE
        assert result.method == "rule"

    def test_simple_thanks(self):
        """感谢 → 简单"""
        ca = ComplexityAnalyzer(llm=None)
        result = ca.analyze("谢谢你的帮助")
        assert result.level == ComplexityLevel.SIMPLE

    def test_simple_confirmation(self):
        """确认 → 简单"""
        ca = ComplexityAnalyzer(llm=None)
        result = ca.analyze("好的，明白了")
        assert result.level == ComplexityLevel.SIMPLE

    def test_complex_multi_step(self):
        """多步骤指令 → 复杂"""
        ca = ComplexityAnalyzer(llm=None)
        result = ca.analyze("先查一下 RAG 的原理，然后对比一下 LangChain 和 LlamaIndex")
        assert result.level == ComplexityLevel.COMPLEX
        assert result.method == "rule"

    def test_complex_comparison(self):
        """对比类 → 复杂"""
        ca = ComplexityAnalyzer(llm=None)
        result = ca.analyze("LoRA 和 QLoRA 有什么区别？哪个更好？")
        assert result.level == ComplexityLevel.COMPLEX

    def test_complex_analysis(self):
        """分析+建议 → 复杂"""
        ca = ComplexityAnalyzer(llm=None)
        result = ca.analyze("帮我分析一下 vLLM 部署方案，并给出建议")
        assert result.level == ComplexityLevel.COMPLEX

    def test_short_message_defaults_simple(self):
        """短消息无技术术语 → 简单"""
        ca = ComplexityAnalyzer(llm=None)
        result = ca.analyze("好的")
        assert result.level == ComplexityLevel.SIMPLE

    def test_long_message_tends_complex(self):
        """长消息（>80字） → 倾向复杂"""
        ca = ComplexityAnalyzer(llm=None)
        long_msg = "我想了解一下关于大模型微调的各种方法" * 5  # >80 chars
        result = ca.analyze(long_msg)
        assert result.level == ComplexityLevel.COMPLEX

    def test_medium_technical_falls_back(self):
        """中等长度技术问题 → 规则无法判断，fallback 到简单（安全兜底）"""
        ca = ComplexityAnalyzer(llm=None)
        # 这条消息不算简单（不是问候），不算复杂（无多步骤信号），不算很长
        result = ca.analyze("LoRA 的参数设置有什么讲究？")
        assert result.method == "fallback"
        assert result.level == ComplexityLevel.SIMPLE  # 安全兜底


class TestTaskPlanner:
    """任务规划器测试"""

    def test_plan_comparison_question(self):
        """对比问题 → 3 步计划（检索→对比→建议）"""
        plan = TaskPlanner.plan("LoRA 和 QLoRA 有什么区别？", Intent.COURSE_QUESTION)
        assert len(plan.steps) == 3
        assert plan.steps[0].action == "rag"
        assert "检索" in plan.steps[0].description
        assert "对比" in plan.steps[1].description
        assert "建议" in plan.steps[2].description

    def test_plan_how_to_question(self):
        """"怎么做"问题 → 2 步计划（检索→整理）"""
        plan = TaskPlanner.plan("怎么用 vLLM 部署大模型？", Intent.COURSE_QUESTION)
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "rag"

    def test_plan_simple_course_question(self):
        """简单课程问题 → 1 步（直接检索）"""
        plan = TaskPlanner.plan("什么是 RAG？", Intent.COURSE_QUESTION)
        assert len(plan.steps) == 1
        assert plan.steps[0].action == "rag"

    def test_plan_web_search(self):
        """网络搜索 → 2 步（搜索→总结）"""
        plan = TaskPlanner.plan("今天天气怎么样", Intent.WEB_SEARCH)
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "search"

    def test_plan_casual_chat(self):
        """闲聊 → 1 步直接回答"""
        plan = TaskPlanner.plan("你好呀", Intent.CASUAL_CHAT)
        assert len(plan.steps) == 1


class TestTaskPlan:
    """执行计划状态跟踪测试"""

    def test_plan_progress_tracking(self):
        """步骤状态跟踪"""
        plan = TaskPlan(original_question="测试问题")
        plan.steps = [
            TaskStep(order=1, description="第一步"),
            TaskStep(order=2, description="第二步"),
            TaskStep(order=3, description="第三步"),
        ]

        assert not plan.is_complete

        # 执行第一步
        assert plan.next_step.order == 1
        plan.mark_done(1)
        assert plan.steps[0].status == "done"

        # 下一步
        assert plan.next_step.order == 2
        plan.mark_done(2)
        plan.mark_done(3)

        assert plan.is_complete
        assert plan.next_step is None

    def test_format_progress(self):
        """进度格式化输出"""
        plan = TaskPlan(original_question="测试")
        plan.steps = [
            TaskStep(order=1, description="检索资料"),
            TaskStep(order=2, description="分析对比"),
        ]
        plan.mark_done(1)

        output = plan.format_progress()
        assert "执行计划" in output
        assert "✅" in output  # 第一步完成
        assert "⏳" in output  # 第二步待执行
        assert "检索资料" in output
        assert "分析对比" in output


class TestFollowUpRecommender:
    """后续问题推荐器测试"""

    def test_recommend_returns_list(self):
        """推荐返回列表，不超过 3 个"""
        fr = FollowUpRecommender(llm=None)
        result = fr.recommend(
            "什么是 LoRA？",
            "LoRA 是一种低秩适应微调方法，通过插入低秩矩阵实现参数高效微调。",
            "course_question",
        )
        assert isinstance(result, list)
        assert 1 <= len(result) <= 3

    def test_recommend_no_duplicates(self):
        """推荐问题不重复"""
        fr = FollowUpRecommender(llm=None)
        result = fr.recommend(
            "对比 LoRA 和 QLoRA",
            "LoRA 和 QLoRA 都是微调方法。LoRA 使用低秩分解，QLoRA 加入量化。对比来看...",
            "course_question",
        )
        assert len(result) == len(set(result))

    def test_recommend_with_rag_topic(self):
        """RAG 话题应触发相关模板"""
        fr = FollowUpRecommender(llm=None)
        result = fr.recommend(
            "RAG 怎么实现？",
            "RAG 结合了检索和生成，先用向量数据库检索相关文档，再让 LLM 基于检索结果生成回答。",
            "course_question",
        )
        assert len(result) >= 1

    def test_recommend_short_answer_skips(self):
        """短回答（≤20字）可以被推荐（由调用方控制）"""
        fr = FollowUpRecommender(llm=None)
        # 即使回答短，推荐器自身不会崩溃
        result = fr.recommend("你好", "你好！有什么可以帮助你的？", "casual_chat")
        assert isinstance(result, list)

    def test_recommend_empty_answer(self):
        """空回答不崩溃"""
        fr = FollowUpRecommender(llm=None)
        result = fr.recommend("测试", "", "course_question")
        assert isinstance(result, list)
        assert len(result) >= 0

    def test_recommend_web_search_intent(self):
        """搜索意图 + 技术内容也能生成推荐"""
        fr = FollowUpRecommender(llm=None)
        result = fr.recommend(
            "最新的 LLM 推理框架有哪些",
            "目前主流的 LLM 推理框架包括 vLLM、TensorRT-LLM、LMDeploy 等，vLLM 部署简单高效。",
            "web_search",
        )
        assert len(result) >= 1

    def test_recommend_non_tech_returns_empty(self):
        """非技术内容返回空推荐（合理行为）"""
        fr = FollowUpRecommender(llm=None)
        result = fr.recommend(
            "今天天气怎么样",
            "今天晴朗，适合出行。",
            "web_search",
        )
        # 非技术内容无模板匹配，返回空列表
        assert isinstance(result, list)
        assert len(result) == 0
