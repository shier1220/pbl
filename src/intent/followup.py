"""后续问题推荐系统 — 规则模板 + LLM 兜底

策略：
- 从回答中提取技术关键词，用模板生成 2-3 个相关问题
- LLM 模式（可选）：基于对话上下文生成更精准的后续问题
- 去重 + 截断，确保推荐简洁有用
"""

import re
import random
import logging
import asyncio
from typing import List, Optional

logger = logging.getLogger("course_assistant.followup")

# 技术关键词 → 模板问题（至少匹配一个关键词才触发对应模板）
TOPIC_TEMPLATES = [
    # (关键词列表, [模板问题列表])
    (
        ["LoRA", "QLoRA", "微调", "fine-tune"],
        [
            "LoRA 和 QLoRA 的主要区别是什么？",
            "微调需要多少显存？",
            "全量微调和 LoRA 微调怎么选？",
        ],
    ),
    (
        ["RAG", "检索增强", "知识库", "向量数据库"],
        [
            "RAG 和传统搜索有什么区别？",
            "如何提升 RAG 的检索准确率？",
            "有哪些好用的向量数据库推荐？",
        ],
    ),
    (
        ["vLLM", "部署", "推理", "GPU"],
        [
            "vLLM 部署需要什么配置？",
            "如何优化推理速度？",
            "除了 vLLM 还有哪些推理框架？",
        ],
    ),
    (
        ["Agent", "智能体", "LangChain", "Dify"],
        [
            "Agent 和 RAG 怎么配合使用？",
            "LangChain 和 Dify 各有什么优缺点？",
            "多 Agent 协作怎么实现？",
        ],
    ),
    (
        ["Transformer", "注意力", "大模型", "LLM"],
        [
            "Transformer 的核心原理是什么？",
            "大模型选型有什么建议？",
            "开源模型和闭源模型怎么选？",
        ],
    ),
    (
        ["量化", "剪枝", "蒸馏", "压缩"],
        [
            "模型量化对精度影响大吗？",
            "知识蒸馏的效果怎么样？",
            "有哪些模型压缩的最佳实践？",
        ],
    ),
    (
        ["Prompt", "提示词", "上下文"],
        [
            "怎么写好 Prompt？",
            "Few-shot 和 Zero-shot 怎么选？",
            "上下文窗口不够用怎么办？",
        ],
    ),
]

# 通用兜底模板（至少匹配一个技术关键词就触发）
GENERIC_TECH_TEMPLATES = [
    "能再详细解释一下吗？",
    "有没有相关的实战案例？",
    "这个技术有什么局限性？",
    "学习这个需要什么前置知识？",
]

# 对比/分析类回复的专用模板
COMPARISON_TEMPLATES = [
    "在实际项目中应该怎么选择？",
    "还有哪些类似的方案可以对比？",
    "不同场景下哪个更合适？",
]

# 教程/方法类回复的专用模板
HOWTO_TEMPLATES = [
    "有没有更简单的替代方法？",
    "常见的坑有哪些？怎么避免？",
    "能推荐一些学习资源吗？",
]

# LLM 生成推荐问题的 prompt（轻量）
FOLLOWUP_LLM_PROMPT = """根据以下对话，生成 3 个用户可能感兴趣的后续问题。
问题要简短（≤20字），用中文，每行一个。

用户问题：{user_query}
助手回答摘要：{answer_summary}

只输出 3 行问题，不要编号："""


class FollowUpRecommender:
    """后续问题推荐器"""

    def __init__(self, llm=None):
        self.llm = llm

    def recommend(self, user_query: str, answer: str, intent: str = "") -> List[str]:
        """
        规则模板生成推荐问题。

        返回 2-3 个推荐问题，最多 3 个。
        """
        suggestions = []

        # 1. 从回答中提取技术关键词 → 匹配模板
        suggestions.extend(self._match_topic_templates(answer))

        # 2. 根据意图/内容类型补充
        suggestions.extend(self._match_content_type(user_query, answer, intent))

        # 3. 补充通用模板
        suggestions.extend(self._add_generic(user_query, answer, suggestions))

        # 去重 + 截断至 3 个
        seen = set()
        unique = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique[:3]

    async def recommend_async(
        self, user_query: str, answer: str, intent: str = ""
    ) -> List[str]:
        """异步推荐（规则 + LLM 兜底）"""
        # 先尝试规则
        suggestions = self.recommend(user_query, answer, intent)

        # 如果规则产出不足 3 个，且 LLM 可用，用 LLM 补充
        if len(suggestions) < 3 and self.llm:
            try:
                llm_suggestions = await self._llm_recommend(user_query, answer)
                if llm_suggestions:
                    # 合并并去重
                    seen = set(suggestions)
                    for s in llm_suggestions:
                        if s not in seen:
                            seen.add(s)
                            suggestions.append(s)
                    suggestions = suggestions[:3]
            except Exception as e:
                logger.debug("LLM 后续问题生成失败: %s", e)

        return suggestions[:3]

    def _match_topic_templates(self, answer: str) -> List[str]:
        """从回答中提取关键词，匹配话题模板"""
        result = []
        answer_lower = answer.lower()

        for keywords, templates in TOPIC_TEMPLATES:
            if any(kw.lower() in answer_lower for kw in keywords):
                # 随机取 1 个该话题的模板（避免每次都一样）
                picked = random.choice(templates)
                result.append(picked)

        return result[:2]  # 最多 2 个话题模板

    def _match_content_type(
        self, user_query: str, answer: str, intent: str
    ) -> List[str]:
        """根据内容类型补充专用模板"""
        combined = (user_query + " " + answer).lower()

        # 检测对比/比较类
        if any(kw in combined for kw in ["对比", "比较", "区别", "差异", "哪个更"]):
            return random.sample(
                COMPARISON_TEMPLATES, min(2, len(COMPARISON_TEMPLATES))
            )

        # 检测教程/方法类
        if any(kw in combined for kw in ["怎么做", "如何", "步骤", "教程", "方法", "实现"]):
            return random.sample(
                HOWTO_TEMPLATES, min(2, len(HOWTO_TEMPLATES))
            )

        return []

    def _add_generic(
        self, user_query: str, answer: str, existing: List[str]
    ) -> List[str]:
        """补充通用兜底模板"""
        # 如果已有足够建议，不补
        if len(existing) >= 2:
            return []

        has_tech = any(
            kw.lower() in (user_query + " " + answer).lower()
            for kw in [
                "LoRA", "RAG", "vLLM", "Agent", "微调", "部署", "模型",
                "Transformer", "LLM", "大模型", "训练", "推理",
            ]
        )

        if has_tech:
            # 从通用技术模板中随机选（避免与已有重复）
            available = [t for t in GENERIC_TECH_TEMPLATES if t not in existing]
            if available:
                return random.sample(available, min(1, len(available)))

        return []

    async def _llm_recommend(self, user_query: str, answer: str) -> List[str]:
        """LLM 生成后续问题"""
        if not self.llm:
            return []

        # 截断回答（LLM 不需要完整回答）
        answer_summary = answer[:300] + "…" if len(answer) > 300 else answer

        prompt = FOLLOWUP_LLM_PROMPT.format(
            user_query=user_query, answer_summary=answer_summary
        )

        try:
            resp = await asyncio.to_thread(
                self.llm.invoke, prompt, {"max_tokens": 80}
            )
            lines = [
                line.strip()
                for line in resp.content.strip().split("\n")
                if line.strip() and len(line.strip()) <= 30
            ]
            return lines[:3]
        except Exception as e:
            logger.debug("LLM 推荐失败: %s", e)
            return []
