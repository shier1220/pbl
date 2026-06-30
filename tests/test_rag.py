"""RAG 检索与生成测试"""
import pytest
from src.rag.retriever import BM25Index, RetrievalResult
from src.rag.prompts import (
    build_rag_prompt,
    format_context_with_labels,
    format_sources,
    SYSTEM_PROMPT,
)


# ─── 辅助：构造假的检索结果 ───────────────────────────────────────
def _make_doc(content, source="test.pdf", chroma_id="id_0"):
    """快速构造一个 RetrievalResult"""
    return RetrievalResult(
        content=content,
        metadata={"source": source},
        chroma_id=chroma_id,
        dense_score=0.9,
    )


class TestBM25Index:
    """BM25 稀疏检索单元测试 — 无外部依赖"""

    def test_tokenize_chinese(self):
        """jieba 中文分词"""
        tokens = BM25Index.tokenize("Python机器学习教程")               # 中文 分词
        assert "Python" in tokens                   # 英文 保留
        assert "机器学习" in tokens or "机器" in tokens         # 中文分词结果可能是 "机器学习" 或 "机器" + "学习"
        assert len(tokens) >= 2             # 分词结果至少包含两个词

    def test_tokenize_english(self):
        """英文按空格分词"""
        tokens = BM25Index.tokenize("hello world")
        assert tokens == ["hello", "world"]

    def test_build_and_search(self):
        """建索引 + 搜索返回非空结果"""
        docs = [
            "Python 是一门流行的编程语言",
            "机器学习是人工智能的一个分支",
            "深度学习使用神经网络进行训练",
        ]
        bm25 = BM25Index()
        bm25.build(docs)

        assert not bm25.is_empty
        assert bm25.doc_count == 3

        results = bm25.search("Python 编程", top_k=2)
        assert len(results) >= 1
        # 第一个结果应该包含 "Python"
        assert "Python" in results[0].content

    def test_search_returns_retrieval_result_type(self):
        """搜索结果类型正确"""
        docs = [
            "Python 机器学习深度学习教程",
            "Java 企业级开发框架",
            "前端 React 组件设计模式",
        ]
        bm25 = BM25Index()
        bm25.build(docs)

        results = bm25.search("机器学习", top_k=3)
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r, RetrievalResult)
        assert r.sparse_score is not None
        assert r.sparse_score > 0

    def test_empty_index_search_returns_empty(self):
        """空索引搜索返回空列表"""
        bm25 = BM25Index()
        assert bm25.is_empty
        assert bm25.search("任意查询") == []

    def test_build_with_metadata_and_ids(self):
        """带元数据和 ID 建索引"""
        docs = ["神经网络深度学习入门", "Python 数据分析教程", "Linux 系统管理指南"]
        metas = [{"source": "dl.pdf"}, {"source": "py.pdf"}, {"source": "linux.pdf"}]
        ids = ["id_1", "id_2", "id_3"]

        bm25 = BM25Index()
        bm25.build(docs, metadatas=metas, ids=ids)

        results = bm25.search("深度学习", top_k=1)
        assert len(results) == 1
        assert results[0].metadata["source"] == "dl.pdf"
        assert results[0].chroma_id == "id_1"


class TestRAGPrompts:
    """RAG 提示词构建单元测试 — 无外部依赖"""

    def test_format_sources_numbered(self):
        """format_sources 产出编号来源行"""
        docs = [
            _make_doc("LoRA 是一种高效微调方法", "lora.pdf"),
            _make_doc("QLoRA 引入了量化技术", "qlora.pdf"),
        ]
        result = format_sources(docs)
        assert "[1]" in result
        assert "[2]" in result
        assert "lora.pdf" in result
        assert "qlora.pdf" in result

    def test_format_sources_empty(self):
        """空列表返回空字符串"""
        assert format_sources([]) == ""

    def test_format_context_with_labels(self):
        """format_context_with_labels 带来源标注"""
        docs = [_make_doc("内容片段", "notes.txt")]
        result = format_context_with_labels(docs)
        assert "[1]" in result
        assert "来源: notes.txt" in result
        assert "内容片段" in result

    def test_build_rag_prompt_contains_key_parts(self):
        """build_rag_prompt 拼装完整 prompt"""
        prompt = build_rag_prompt(
            system_prompt="你是小课",
            question="什么是 LoRA？",
            context="LoRA = Low-Rank Adaptation",
            history_text="用户：你好\n小课：你好！",
            user_name="小明",
        )
        assert "你是小课" in prompt
        assert "什么是 LoRA？" in prompt
        assert "LoRA = Low-Rank Adaptation" in prompt
        assert "小明" in prompt
        assert "对话历史" in prompt

    def test_build_rag_prompt_no_history_no_name(self):
        """无历史无用户名时不报错"""
        prompt = build_rag_prompt(
            system_prompt="你是小课",
            question="测试？",
            context="一些资料",
            history_text="",
            user_name="",
        )
        assert "测试？" in prompt
        assert "一些资料" in prompt
        # 无历史时不应出现"对话历史"
        assert "对话历史" not in prompt


# ─── HybridRetriever 集成测试 fixture（类级别，只加载一次）─────────

@pytest.fixture(scope="class")
def hybrid_retriever():
    """加载 ChromaDB + InstructorEmbedding，整个测试类复用"""
    import chromadb
    from langchain_community.vectorstores import Chroma
    from src.embedding import InstructorEmbedding
    from src.rag.retriever import HybridRetriever
    from src.config import CHROMA_PATH, COLLECTION_NAME

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = InstructorEmbedding()
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    ret = HybridRetriever(vectorstore, embedding_fn)
    ret.build_bm25_index()
    return ret


class TestHybridRetriever:
    """混合检索器集成测试 — 需要 ChromaDB + embedding 模型"""

    def test_retrieve_rag_returns_results(self, hybrid_retriever):
        """检索 RAG 相关关键词应返回非空结果"""
        results = hybrid_retriever.retrieve("RAG 检索增强生成", top_k=3)
        assert len(results) >= 1, "RAG 关键词应命中至少 1 条结果"

    def test_retrieve_agent_returns_results(self, hybrid_retriever):
        """检索 Agent 框架应返回非空结果"""
        results = hybrid_retriever.retrieve("Agent 智能体框架", top_k=3)
        assert len(results) >= 1, "Agent 关键词应命中至少 1 条结果"

    def test_retrieve_result_has_required_fields(self, hybrid_retriever):
        """每条结果都包含 content 字段"""
        results = hybrid_retriever.retrieve("微调 LoRA", top_k=5)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r.content, str) and len(r.content) > 0, \
                "每条结果的 content 应为非空字符串"

    def test_retrieve_dense_search_works_alone(self, hybrid_retriever):
        """纯稠密检索也能返回结果（不依赖 BM25）"""
        results = hybrid_retriever._dense_search("深度学习 Transformer", top_k=3)
        assert len(results) >= 1
        for r in results:
            assert r.dense_score is not None

    def test_bm25_index_is_built(self, hybrid_retriever):
        """BM25 索引已构建"""
        assert not hybrid_retriever.bm25.is_empty
        assert hybrid_retriever.bm25.doc_count > 0

    def test_retrieve_stats(self, hybrid_retriever):
        """统计信息可读"""
        stats = hybrid_retriever.stats
        assert "bm25_docs" in stats
        assert stats["bm25_docs"] > 0
