"""
多类意图分类器 — 混合策略

策略:
  1. 关键词/正则 → 细分非RAG意图（web_search, file_operation, system_command）
  2. ChromaDB 检索分数 → 判断课程问题
  3. 技术术语检测 → 辅助判断
  4. 默认 → casual_chat
"""
import re, logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional
from src.config import INTENT_CONFIDENCE_HIGH, INTENT_CONFIDENCE_MEDIUM
logger = logging.getLogger("course_assistant.intent")

class Intent(str, Enum):
    COURSE_QUESTION = "course_question"
    CASUAL_CHAT = "casual_chat"
    FILE_OPERATION = "file_operation"
    WEB_SEARCH = "web_search"
    SYSTEM_COMMAND = "system_command"

@dataclass
class IntentResult:
    intent: Intent; confidence: float
    all_scores: Dict[Intent, float] = field(default_factory=dict)
    threshold_level: str = "high"; method: str = "default"
    @property
    def is_high_confidence(self): return self.confidence >= INTENT_CONFIDENCE_HIGH
    @property
    def is_medium_confidence(self): return INTENT_CONFIDENCE_MEDIUM <= self.confidence < INTENT_CONFIDENCE_HIGH
    @property
    def is_low_confidence(self): return self.confidence < INTENT_CONFIDENCE_MEDIUM

# 关键词预编译
_WEB_PATTERNS = [re.compile(p) for p in [
    r"(搜索|查一下|帮我查|上网|联网).*(最新|新闻|天气|股价|股票|进展|动态)",
    r"(今天|现在|当前|实时|最新).*(天气|新闻|股价|进展|动态)",
    r"(几点了|现在几点|今天几号)", r"(帮我搜|搜索一下|查一查|查查)",
]]
_FILE_PATTERNS = [re.compile(p) for p in [
    r"(上传|解析|分析|查看).*(文件|文档|PDF|DOCX|PPT|资料)",
    r"(文件|文档|资料).*(上传|解析|处理|分析|格式)",
]]
_SYS_PATTERNS = [re.compile(p) for p in [
    r"(列出|查看|显示).*(会话|对话|聊天)", r"(新建|创建|删除|切换).*(会话|对话)",
    r"(帮助|使用说明|怎么用|能做什么|功能)", r"^(帮助|help)$",
]]
_TECH_TERMS = ["RAG","LoRA","QLoRA","vLLM","PagedAttention","微调","fine-tune","向量数据库","embedding","Transformer","注意力机制","attention","大模型","LLM","推理","inference","Agent","智能体","Prompt","提示词","蒸馏","distillation","部署","deploy","GPU","显存","深度学习","AIGC","生成式","RLHF","DPO"]

class MultiClassIntentClassifier:
    def __init__(self, vectorstore=None): self.vectorstore = vectorstore

    def classify(self, message, history=None):
        msg = message.strip(); ml = msg.lower(); scores = {i: 0.0 for i in Intent}

        # ChromaDB 分数
        rag_score = 0.0
        if self.vectorstore:
            try:
                r = self.vectorstore.similarity_search_with_relevance_scores(msg, k=1)
                if r: rag_score = r[0][1]
            except Exception: pass
        scores[Intent.COURSE_QUESTION] = rag_score

        # 关键词匹配
        web_score = 0.8 if any(p.search(msg) for p in _WEB_PATTERNS) else (0.6 if any(k in ml for k in ["搜索","上网查","查一下","搜一下"]) else 0.0)
        file_score = 0.8 if any(p.search(msg) for p in _FILE_PATTERNS) else 0.0
        sys_score = 0.8 if any(p.search(msg) for p in _SYS_PATTERNS) else 0.0
        scores[Intent.WEB_SEARCH] = web_score
        scores[Intent.FILE_OPERATION] = file_score
        scores[Intent.SYSTEM_COMMAND] = sys_score

        has_tech = any(t.lower() in ml for t in _TECH_TERMS)

        # 决策
        if web_score >= 0.6: intent, conf, method = Intent.WEB_SEARCH, web_score, "keyword"
        elif file_score >= 0.6: intent, conf, method = Intent.FILE_OPERATION, file_score, "keyword"
        elif sys_score >= 0.6: intent, conf, method = Intent.SYSTEM_COMMAND, sys_score, "keyword"
        elif rag_score >= 0.60: intent, conf, method = Intent.COURSE_QUESTION, rag_score, "chromadb"
        elif has_tech: intent, conf, method = Intent.COURSE_QUESTION, 0.55, "tech_keyword"
        else: intent, conf, method = Intent.CASUAL_CHAT, 0.5, "default"

        level = "high" if conf >= INTENT_CONFIDENCE_HIGH else ("medium" if conf >= INTENT_CONFIDENCE_MEDIUM else "low")
        scores[intent] = max(scores[intent], conf)
        logger.info("[Intent] '%s...' → %s (conf=%.3f, %s, rag=%.3f)", msg[:30], intent.value, conf, method, rag_score)
        return IntentResult(intent=intent, confidence=conf, all_scores=scores, threshold_level=level, method=method)

    def classify_with_context(self, message, history): return self.classify(message, history)
    @staticmethod
    def has_technical_terms(msg): return any(t.lower() in msg.lower() for t in _TECH_TERMS)
