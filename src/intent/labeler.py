"""
意图标注数据生成器 — 用 DeepSeek 为每类意图生成多样化样本

运行: python -m src.intent.labeler
输出: src/intent/intent_samples.json + ChromaDB collection "intent_samples"
"""
import json, os, sys, logging
from pathlib import Path
from typing import List, Dict

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langchain_openai import ChatOpenAI
from chromadb import PersistentClient

from src.config import (
    CHROMA_PATH, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    MODEL_PATH, EMBEDDING_DEVICE,
)
from src.embedding import InstructorEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("intent_labeler")

INTENT_COLLECTION = "intent_samples"

# 每类意图的生成提示
INTENT_SPECS = {
    "course_question": {
        "desc": "AIGC/大模型技术问题，用户想了解或学习某个技术概念",
        "examples": [
            "什么是LoRA微调？", "RAG和微调有什么区别？", "vLLM怎么部署？",
            "PagedAttention原理是什么？", "QLoRA和LoRA的区别", "向量数据库选型建议",
            "如何做模型蒸馏？", "大模型显存不够怎么办？", "Agent怎么搭建？",
            "Prompt工程最佳实践", "RLHF训练流程", "embedding模型怎么选？",
        ],
    },
    "casual_chat": {
        "desc": "日常闲聊问候，与技术无关",
        "examples": [
            "你好", "嗨", "早上好", "你是谁？", "今天心情不好",
            "谢谢你的回答", "再见", "哈哈", "你叫什么名字？", "在吗",
            "晚安", "下午好", "hello", "hi", "你能做什么？",
        ],
    },
    "web_search": {
        "desc": "需要实时信息或联网搜索的问题",
        "examples": [
            "今天天气怎么样？", "帮我搜一下最新的AI新闻", "现在几点了？",
            "搜一下DeepSeek最新版本", "世界杯今晚谁打谁", "查一下明天上海天气",
            "最近有什么AI大会？", "帮我查一下股票行情", "今天星期几？",
            "搜索Transformer最新论文", "今年双十一什么时候？", "帮我找一下租房信息",
        ],
    },
    "file_operation": {
        "desc": "用户想上传/解析/分析文件",
        "examples": [
            "帮我分析这个PDF", "上传一个文档", "能解析Word文件吗？",
            "我把论文传给你看看", "帮忙总结这个PPT", "支持哪些文件格式？",
            "上传文件后怎么用？", "帮我看一下这份报告", "能读txt吗？",
            "解析一下这个IPYNB文件", "怎么上传资料？", "文件太大能传吗？",
        ],
    },
    "system_command": {
        "desc": "系统操作、帮助、会话管理",
        "examples": [
            "帮助", "怎么用？", "能做什么？", "功能列表",
            "新建会话", "切换对话", "列出我的会话", "删掉这个对话",
            "有哪些命令？", "使用说明", "清空对话", "导出聊天记录",
        ],
    },
}


def generate_samples(llm, intent: str, spec: dict, n: int = 100) -> List[str]:
    """用 LLM 生成指定意图的多样化样本"""
    prompt = f"""你是一个数据标注专家。请为意图类别「{intent}」生成 {n} 条可能的用户消息。

意图描述：{spec['desc']}

参考示例：{', '.join(spec['examples'][:8])}

要求：
1. 每条消息独立一行，不要编号
2. 覆盖不同表达方式：简短/冗长、口语/书面、直接/间接
3. 包含一些不太典型但仍属于该类意图的边界案例
4. 不要和参考示例完全相同
5. 只输出消息内容，不要任何解释

现在开始生成 {n} 条「{intent}」类别的用户消息："""

    resp = llm.invoke(prompt)
    lines = []
    for line in resp.content.strip().split("\n"):
        line = line.strip()
        # 去掉可能的编号前缀
        if line and len(line) > 1:
            # 去掉 "1. " "1、" 等前缀
            while line and (line[0].isdigit() or line[0] in ".、)）"):
                line = line[1:].strip()
            if len(line) > 1:
                lines.append(line)
    # 去重 + 加上参考示例
    seen = set()
    samples = []
    for s in spec["examples"] + lines:
        if s not in seen:
            seen.add(s)
            samples.append(s)
    return samples[:n]


def main():
    # 1. 初始化 LLM
    logger.info("初始化 DeepSeek...")
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL, base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY,
        temperature=0.9, timeout=120, max_retries=3,
    )

    # 2. 生成样本
    all_samples: Dict[str, List[str]] = {}
    for intent, spec in INTENT_SPECS.items():
        logger.info("生成 %s 样本...", intent)
        samples = generate_samples(llm, intent, spec, n=100)
        all_samples[intent] = samples
        logger.info("  → %d 条", len(samples))

    # 3. 保存 JSON
    json_path = os.path.join(os.path.dirname(__file__), "intent_samples.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    logger.info("样本已保存: %s", json_path)

    # 4. 嵌入 + 存入 ChromaDB
    logger.info("加载嵌入模型...")
    embedding = InstructorEmbedding(MODEL_PATH, device=EMBEDDING_DEVICE)

    client = PersistentClient(path=CHROMA_PATH)
    # 删除旧 collection
    try:
        client.delete_collection(INTENT_COLLECTION)
        logger.info("已删除旧 intent_samples collection")
    except Exception:
        pass

    col = client.create_collection(name=INTENT_COLLECTION, metadata={"description": "意图分类标注样本"})
    logger.info("创建新 collection: %s", INTENT_COLLECTION)

    total = 0
    for intent, samples in all_samples.items():
        if not samples:
            continue
        embeddings = embedding.embed_documents(samples)
        ids = [f"{intent}_{i}" for i in range(len(samples))]
        metas = [{"intent": intent, "idx": i} for i in range(len(samples))]
        col.add(documents=samples, embeddings=embeddings, ids=ids, metadatas=metas)
        total += len(samples)
        logger.info("  写入 %s: %d 条", intent, len(samples))

    logger.info("完成！共 %d 条样本入 ChromaDB", total)


if __name__ == "__main__":
    main()
