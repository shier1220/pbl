"""上下文记忆管理 — 滑动窗口 + 关键事实提取

策略：
- 最近 N 条消息保留原文（精确上下文），默认 6 条
- 更早的消息提取关键事实，拼成一句「记忆摘要」
- 摘要插入 system prompt，无需额外 LLM 调用
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("course_assistant.memory")

# 技术关键词（AIGC 领域），用于识别讨论主题
TECH_KEYWORDS = [
    "LoRA", "QLoRA", "RAG", "vLLM", "微调", "fine-tune", "fine-tuning",
    "Transformer", "大模型", "LLM", "Agent", "AIGC", "模型蒸馏", "数据增强",
    "深度学习", "机器学习", "部署", "推理", "量化", "剪枝", "RLHF",
    "Dify", "LangChain", "Ollama", "GPU", "CUDA", "embedding",
    "知识库", "向量数据库", "Prompt", "提示词", "上下文", "多模态",
]

# "未解决"信号的句子模式
PENDING_PATTERNS = [
    re.compile(p) for p in [
        r"还(?:有一个|有个)问题",
        r"(?:另外|还有|此外).*想问",
        r"下一个问题",
        r"再问(?:一个)?",
        r"还没(?:解决|弄|搞)清楚",
        r"(?:还是|依然|一直)不[懂明白理解]",
    ]
]

# 多步骤任务信号（提取计划描述）
PLAN_PATTERNS = [
    re.compile(p) for p in [
        r"(?:先|首先|第一步)[^，。]*[再然后接着第二步]",
        r"(?:需要|想|帮我|给我)[^，。]{5,}(?:然后|再|接着|之后)",
        r"(?:总结|概述|介绍)[^，。]*(?:对比|比较|区别|优劣)",
    ]
]


class ContextMemory:
    """滑动窗口 + 事实提取的上下文记忆管理器"""

    def __init__(self, recent_window: int = 6):
        self.recent_window = recent_window

    def build_context(self, history: List[dict]) -> Tuple[List[dict], str]:
        """
        处理对话历史，返回 (近期历史, 记忆摘要)。

        - 历史 ≤ recent_window：全部保留为近期历史，无摘要
        - 历史 > recent_window：近 N 条保留，旧消息提取摘要
        """
        if not history:
            return [], ""

        if len(history) <= self.recent_window:
            return list(history), ""

        recent = history[-self.recent_window:]
        older = history[:-self.recent_window]

        memory = self._extract_memory(older)

        return list(recent), self._format_memory(memory)

    def _extract_memory(self, messages: List[dict]) -> dict:
        """从旧消息中提取关键记忆"""
        memory = {
            "user_name": None,
            "topics": set(),
            "pending": [],        # 未解决的问题列表
            "important_msgs": [], # 看起来重要的消息
        }

        for msg in messages:
            content = msg.get("content", "").strip()
            if not content:
                continue

            role = msg.get("role", "user")

            # 提取用户名
            if role == "user":
                name = self._extract_name(content)
                if name and not memory["user_name"]:
                    memory["user_name"] = name

            # 提取技术关键词
            for kw in TECH_KEYWORDS:
                if kw.lower() in content.lower():
                    memory["topics"].add(kw)

            # 检测未解决问题信号（仅用户消息）
            if role == "user":
                for pat in PENDING_PATTERNS:
                    if pat.search(content):
                        # 截取问句部分
                        snippet = content[-100:] if len(content) > 100 else content
                        memory["pending"].append(snippet)
                        break

        return memory

    def _format_memory(self, memory: dict) -> str:
        """将提取的记忆格式化为自然语言摘要"""
        parts = []

        if memory["user_name"]:
            parts.append(f"用户名叫{memory['user_name']}")

        # 主题去重后列出（最多5个）
        topics = sorted(memory["topics"])
        if topics:
            topics_str = "、".join(topics[:5])
            if len(topics) > 5:
                topics_str += "等"
            parts.append(f"之前讨论过：{topics_str}")

        if memory["pending"]:
            # 最多保留 2 条未解决问题
            for p in memory["pending"][:2]:
                short = p[:80] + "…" if len(p) > 80 else p
                parts.append(f"用户还问了但未解决：「{short}」")

        if not parts:
            return ""  # 没有可提取的记忆

        return "【之前的对话记忆】" + "。".join(parts) + "。"

    @staticmethod
    def _extract_name(text: str) -> Optional[str]:
        """从消息中提取用户名"""
        patterns = [
            r"我是(.+?)[，。！？\s]", r"我叫(.+?)[，。！？\s]",
            r"^我是(.+)$", r"^我叫(.+)$", r"我叫(.+)", r"我是(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, text.strip())
            if m:
                name = m.group(1).strip()
                if len(name) <= 10 and name:
                    return name
        return None

    @staticmethod
    def has_pending_questions(history: List[dict]) -> bool:
        """快速检查历史最后一条用户消息是否包含未解决信号"""
        if not history:
            return False
        # 找最后一条用户消息
        for msg in reversed(history):
            if msg.get("role") == "user":
                for pat in PENDING_PATTERNS:
                    if pat.search(msg.get("content", "")):
                        return True
                return False
        return False
