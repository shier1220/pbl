"""问题复杂度分析 + 多轮任务规划

策略：
- 规则初筛：问候/确认/简单问句 → simple，多步骤/对比/分析 → complex
- LLM 轻量判断兜底（仅对模糊问题）
- 复杂问题生成执行计划，跟踪步骤进度
"""

import re
import logging
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("course_assistant.complexity")

# ─── 快速规则：简单信号 ───
SIMPLE_PATTERNS = [
    # 纯问候/感谢/告别
    r"^(你好|hi|hello|嗨|早上好|下午好|晚上好|谢谢|多谢|感谢|再见|拜拜|好的|ok|嗯|哦)[\s!！。.,，]*$",
    # 确认/否定
    r"^(是的|对|没错|不是|不对|不用|算了|可以|行|好|明白了|知道了)[\s!！。.,，]*$",
    # 简单系统指令
    r"^(帮助|功能|怎么用|能做什么|你是谁|你叫什么|新建会话|切换会话)[\s!！。.,，？?]*$",
]

# ─── 快速规则：复杂信号 ───
COMPLEX_PATTERNS = [
    # 多步骤指令信号
    r"(先|首先|第一步|然后|接着|再|之后|最后|第二步|第三步)",
    # 对比/比较类
    r"(对比|比较|区别|差异|优劣|哪个更好|优缺点|有什么不同)",
    # 分析+建议组合
    r"(分析|评估).{2,10}(建议|推荐|方案|选择)",
    # 条件判断类
    r"(如果|假如|假设|根据.{2,10}(情况|场景|条件))",
]

# 复杂问题 LLM 判断 prompt（轻量，max_tokens=10）
COMPLEXITY_CHECK_PROMPT = """判断以下用户消息是简单问题还是复杂问题：
- 简单：一句话能回答、无需多步推理、问候/确认/基础知识
- 复杂：需要多步分析、对比多种方案、包含条件判断、先X再Y

{history}
用户消息："{message}"

只输出一个词：simple 或 complex"""


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


@dataclass
class ComplexityResult:
    level: ComplexityLevel
    reason: str = ""
    method: str = "rule"  # rule | llm | fallback


@dataclass
class TaskStep:
    """单个执行步骤"""
    order: int
    description: str         # 步骤描述（给用户看的）
    action: str = "rag"      # rag | search | llm | compose
    query_hint: str = ""     # 搜索/检索的查询提示
    status: str = "pending"  # pending | running | done


@dataclass
class TaskPlan:
    """复杂问题执行计划"""
    original_question: str
    steps: List[TaskStep] = field(default_factory=list)
    current_step: int = 0

    @property
    def is_complete(self) -> bool:
        return all(s.status == "done" for s in self.steps)

    @property
    def next_step(self) -> Optional[TaskStep]:
        """获取下一个待执行步骤"""
        pending = [s for s in self.steps if s.status == "pending"]
        return pending[0] if pending else None

    def mark_done(self, step_order: int):
        """标记步骤完成"""
        for s in self.steps:
            if s.order == step_order:
                s.status = "done"
                break

    def format_progress(self) -> str:
        """格式化进度给用户看"""
        lines = ["📋 **执行计划**："]
        for s in self.steps:
            icon = "✅" if s.status == "done" else "🔄" if s.status == "running" else "⏳"
            lines.append(f"  {icon} 步骤{s.order}：{s.description}")
        return "\n".join(lines)


class ComplexityAnalyzer:
    """复杂度分析器 — 规则优先 + LLM 兜底"""

    def __init__(self, llm=None):
        self.llm = llm

    def analyze(self, message: str, history: list = None) -> ComplexityResult:
        """同步分析（规则部分，不需要 LLM）"""
        # 规则优先
        result = self._rule_check(message)
        if result:
            return result
        # 规则无法判断 → 标记为需要 LLM（由调用方异步处理）
        return ComplexityResult(
            level=ComplexityLevel.SIMPLE,  # 默认简单，安全兜底
            reason="规则无法判断，默认简单",
            method="fallback",
        )

    async def analyze_async(self, message: str, history: list = None) -> ComplexityResult:
        """异步分析（规则 + LLM 兜底）"""
        # 规则优先
        result = self._rule_check(message)
        if result:
            return result

        # LLM 轻量判断
        if self.llm:
            try:
                llm_result = await self._llm_check(message, history)
                if llm_result:
                    return llm_result
            except Exception as e:
                logger.debug("LLM 复杂度判断失败: %s", e)

        # 默认简单
        return ComplexityResult(
            level=ComplexityLevel.SIMPLE,
            reason="默认简单（安全兜底）",
            method="fallback",
        )

    def _rule_check(self, message: str) -> Optional[ComplexityResult]:
        """规则快速判断"""
        msg = message.strip()

        # 检查简单信号
        for pat in SIMPLE_PATTERNS:
            if re.match(pat, msg, re.IGNORECASE):
                return ComplexityResult(
                    level=ComplexityLevel.SIMPLE,
                    reason="问候/确认/简单指令",
                    method="rule",
                )

        # 检查复杂信号
        for pat in COMPLEX_PATTERNS:
            if re.search(pat, msg):
                return ComplexityResult(
                    level=ComplexityLevel.COMPLEX,
                    reason="多步骤/对比/分析",
                    method="rule",
                )

        # 消息很短（≤10字）且无技术术语 → 默认简单
        if len(msg) <= 10:
            from src.intent.classifier import EmbeddingIntentClassifier
            if not EmbeddingIntentClassifier.has_technical_terms(msg):
                return ComplexityResult(
                    level=ComplexityLevel.SIMPLE,
                    reason="短消息、无技术术语",
                    method="rule",
                )

        # 消息很长（>80字）→ 倾向复杂
        if len(msg) > 80:
            return ComplexityResult(
                level=ComplexityLevel.COMPLEX,
                reason="长消息，可能包含多重要求",
                method="rule",
            )

        return None  # 规则无法判断

    async def _llm_check(self, message: str, history: list = None) -> Optional[ComplexityResult]:
        """LLM 轻量判断"""
        hist_text = ""
        if history:
            recent = history[-4:]
            for m in recent:
                role = "用户" if m["role"] == "user" else "助手"
                hist_text += f"{role}：{m['content'][:60]}\n"
            if hist_text:
                hist_text = f"对话历史：\n{hist_text}\n"

        prompt = COMPLEXITY_CHECK_PROMPT.format(history=hist_text, message=message)
        try:
            resp = await asyncio.to_thread(self.llm.invoke, prompt, {"max_tokens": 10})
            answer = resp.content.strip().lower()
            if "complex" in answer:
                return ComplexityResult(level=ComplexityLevel.COMPLEX, reason="LLM判断", method="llm")
            elif "simple" in answer:
                return ComplexityResult(level=ComplexityLevel.SIMPLE, reason="LLM判断", method="llm")
        except Exception as e:
            logger.debug("LLM复杂度判断异常: %s", e)
        return None


class TaskPlanner:
    """复杂问题任务规划器"""

    @staticmethod
    def plan(message: str, intent) -> TaskPlan:
        """根据意图和消息内容生成执行计划"""
        plan = TaskPlan(original_question=message)

        if intent.value == "course_question":
            plan.steps = TaskPlanner._plan_course_question(message)
        elif intent.value == "web_search":
            plan.steps = TaskPlanner._plan_web_search(message)
        elif intent.value == "file_operation":
            plan.steps = TaskPlanner._plan_file_operation(message)
        else:
            # 默认：直接回答
            plan.steps = [TaskStep(order=1, description="分析并回答用户问题", action="llm")]

        return plan

    @staticmethod
    def _plan_course_question(message: str) -> List[TaskStep]:
        """课程问题的执行计划"""
        steps = []

        # 检查是否涉及对比
        if any(kw in message for kw in ["对比", "比较", "区别", "差异", "哪个更好", "优劣"]):
            steps = [
                TaskStep(order=1, description="检索相关资料", action="rag",
                         query_hint=message),
                TaskStep(order=2, description="对比分析各方案", action="compose",
                         query_hint="对比优缺点"),
                TaskStep(order=3, description="给出选择建议", action="compose",
                         query_hint="总结并给出推荐"),
            ]
        # 检查是否涉及"怎么做"/"如何"
        elif any(kw in message for kw in ["怎么做", "如何", "怎么", "怎样", "步骤", "教程", "方法"]):
            steps = [
                TaskStep(order=1, description="检索相关教程/方法", action="rag",
                         query_hint=message),
                TaskStep(order=2, description="整理步骤和实践建议", action="compose",
                         query_hint="整理为可操作的步骤"),
            ]
        # 检查是否涉及"分析"
        elif any(kw in message for kw in ["分析", "评估", "总结"]):
            steps = [
                TaskStep(order=1, description="收集相关背景资料", action="rag",
                         query_hint=message),
                TaskStep(order=2, description="多维度分析", action="compose",
                         query_hint="从多个角度分析"),
            ]
        else:
            # 默认：检索 + 回答
            steps = [
                TaskStep(order=1, description="检索知识库资料", action="rag",
                         query_hint=message),
            ]

        return steps

    @staticmethod
    def _plan_web_search(message: str) -> List[TaskStep]:
        """网络搜索的执行计划"""
        steps = [
            TaskStep(order=1, description="搜索最新信息", action="search",
                     query_hint=message),
            TaskStep(order=2, description="整理并总结搜索结果", action="compose",
                     query_hint="总结关键信息"),
        ]
        return steps

    @staticmethod
    def _plan_file_operation(message: str) -> List[TaskStep]:
        """文件操作的执行计划"""
        steps = [
            TaskStep(order=1, description="解析文件内容", action="llm",
                     query_hint="等待用户上传文件"),
            TaskStep(order=2, description="根据文件内容回答", action="compose",
                     query_hint="根据用户问题分析文件"),
        ]
        return steps
