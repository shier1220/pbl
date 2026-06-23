"""
意图分类器 — LLM 分类 + 嵌入 k-NN 回退

数据层（LLM标注）→ 编码层（instructor-xl嵌入）→
分类层（LLM 主分类 + k-NN 回退）→ 输出层（置信度路由）
"""
import asyncio, logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.config import CHROMA_PATH, INTENT_CONFIDENCE_HIGH, INTENT_CONFIDENCE_MEDIUM

logger = logging.getLogger("course_assistant.intent")

INTENT_CLASSIFY_PROMPT = """判断用户消息的意图，5个类别：
- course_question：AIGC/大模型技术问题（LoRA、RAG、vLLM等）
- casual_chat：日常闲聊问候（你好、谢谢、天气真好）
- web_search：需要实时信息或联网搜索（天气、赛事、新闻、股价、城市名）
- file_operation：上传/解析/分析文件（PDF、文档、帮我看看这个文件）
- system_command：系统操作（帮助、新建会话、功能列表）

{history}
用户消息："{message}"

只输出一个类别名和一个0-1的置信度，如：web_search,0.9"""


class Intent(str, Enum):
    COURSE_QUESTION = "course_question"
    CASUAL_CHAT = "casual_chat"
    FILE_OPERATION = "file_operation"
    WEB_SEARCH = "web_search"
    SYSTEM_COMMAND = "system_command"


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    all_scores: Dict[Intent, float] = field(default_factory=dict)
    threshold_level: str = "high"
    method: str = "llm"

    @property
    def is_high_confidence(self):
        return self.confidence >= INTENT_CONFIDENCE_HIGH

    @property
    def is_medium_confidence(self):
        return INTENT_CONFIDENCE_MEDIUM <= self.confidence < INTENT_CONFIDENCE_HIGH

    @property
    def is_low_confidence(self):
        return self.confidence < INTENT_CONFIDENCE_MEDIUM


# 快速规则拦截（城市名、赛事追问等明确场景）
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "武汉",
          "南京", "西安", "天津", "苏州", "长沙", "郑州", "东莞", "青岛"]
SPORTS_FOLLOWUPS = ["小组赛", "淘汰赛", "半决赛", "决赛", "八强", "四强",
                   "今晚", "比分", "谁赢了", "几比几", "哪个队", "几点开始"]


def _quick_check(msg: str) -> Optional[IntentResult]:
    """快速规则拦截——明确场景直接返回，无需 LLM"""
    # 城市查询 → web_search
    for c in CITIES:
        if msg == c or msg.startswith(c + "的") or msg.startswith(c + "天气") or \
           msg.startswith(c + "热") or msg.startswith(c + "冷"):
            return IntentResult(intent=Intent.WEB_SEARCH, confidence=0.9,
                               threshold_level="high", method="quick_city")
    # 赛事追问 → web_search
    for s in SPORTS_FOLLOWUPS:
        if msg == s or msg.startswith(s):
            return IntentResult(intent=Intent.WEB_SEARCH, confidence=0.85,
                               threshold_level="high", method="quick_sports")
    return None


class EmbeddingIntentClassifier:
    """LLM 主分类 + k-NN 回退"""

    def __init__(self, embedding_fn=None, llm=None):
        self.embedding_fn = embedding_fn
        self.llm = llm
        self._collection = None

    async def classify(self, message: str, history=None) -> IntentResult:
        msg = message.strip()
        scores = {i: 0.0 for i in Intent}

        # 快速规则拦截
        quick = _quick_check(msg)
        if quick:
            return quick

        # LLM 分类
        if self.llm:
            result = await self._llm_classify(msg, history)
            if result:
                return result

        # k-NN 回退
        return self._knn_classify(msg, scores)

    def classify_sync(self, message: str, history=None) -> IntentResult:
        """同步版本（无 LLM 时使用）"""
        msg = message.strip()
        scores = {i: 0.0 for i in Intent}
        quick = _quick_check(msg)
        if quick:
            return quick
        return self._knn_classify(msg, scores)

    classify_with_context = classify_sync

    async def _llm_classify(self, msg: str, history=None) -> Optional[IntentResult]:
        """LLM 直接分类"""
        hist_text = ""
        if history:
            recent = history[-4:]
            for m in recent:
                role = "用户" if m["role"] == "user" else "助手"
                hist_text += f"{role}：{m['content'][:80]}\n"
            if hist_text:
                hist_text = f"对话历史：\n{hist_text}\n"

        prompt = INTENT_CLASSIFY_PROMPT.format(history=hist_text, message=msg)
        try:
            resp = await asyncio.to_thread(self.llm.invoke, prompt, {"max_tokens": 20})
            answer = resp.content.strip().lower()
            # 解析 "intent,confidence"
            parts = answer.replace(" ", "").split(",")
            intent_str = parts[0].strip()
            conf = float(parts[1]) if len(parts) > 1 else 0.7

            for intent in Intent:
                if intent.value in intent_str:
                    level = "high" if conf >= INTENT_CONFIDENCE_HIGH else (
                        "medium" if conf >= INTENT_CONFIDENCE_MEDIUM else "low")
                    logger.info("[Intent LLM] '%s' → %s (%.2f)", msg[:30], intent.value, conf)
                    return IntentResult(intent=intent, confidence=conf,
                                       threshold_level=level, method="llm")
        except Exception as e:
            logger.debug("LLM 分类失败: %s", e)
        return None

    def _knn_classify(self, msg: str, scores: dict) -> IntentResult:
        """k-NN 嵌入分类（回退）"""
        if self.embedding_fn is None or self._get_collection() is None:
            return IntentResult(intent=Intent.CASUAL_CHAT, confidence=0.5,
                               all_scores=scores, threshold_level="low", method="fallback")

        q_embedding = self.embedding_fn.embed_query(msg)
        raw = self._collection.query(
            query_embeddings=[q_embedding], n_results=10,
            include=["documents", "metadatas", "distances"],
        )

        if not raw.get("ids") or not raw["ids"][0]:
            return IntentResult(intent=Intent.CASUAL_CHAT, confidence=0.5,
                               all_scores=scores, threshold_level="low", method="no_match")

        metas = raw["metadatas"][0]
        distances = raw["distances"][0]
        for i in range(len(raw["documents"][0])):
            intent_str = metas[i].get("intent", "")
            similarity = 1.0 - (distances[i] or 0)
            if intent_str in Intent.__members__.values():
                scores[Intent(intent_str)] += similarity

        total = sum(scores.values()) or 1.0
        for k in scores:
            scores[k] /= total

        best = max(scores, key=scores.get)
        best_sim = 1.0 - (distances[0] or 0) if distances else 0
        conf = scores[best] * (0.9 if best_sim >= 0.75 else 0.7)
        level = "high" if conf >= INTENT_CONFIDENCE_HIGH else "medium" if conf >= INTENT_CONFIDENCE_MEDIUM else "low"

        logger.info("[Intent k-NN] '%s' → %s (%.2f, sim=%.2f)", msg[:30], best.value, conf, best_sim)
        return IntentResult(intent=best, confidence=conf, all_scores=scores,
                           threshold_level=level, method="knn")

    def _get_collection(self):
        if self._collection is None:
            from chromadb import PersistentClient
            client = PersistentClient(path=CHROMA_PATH)
            try:
                self._collection = client.get_collection("intent_samples")
            except Exception:
                logger.warning("intent_samples collection 不存在")
                return None
        return self._collection

    @staticmethod
    def has_technical_terms(msg: str) -> bool:
        terms = ["RAG", "LoRA", "QLoRA", "vLLM", "微调", "fine-tune",
                 "Transformer", "大模型", "LLM", "Agent", "AIGC"]
        return any(t.lower() in msg.lower() for t in terms)
