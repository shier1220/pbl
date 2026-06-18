"""
AIGC 课程助手 API v0.6.0 — FastAPI 入口
"""
import os, sys, json, logging, asyncio
from contextlib import asynccontextmanager
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.config import *
from src.embedding import InstructorEmbedding
from src.session.manager import SessionManager
from src.rag.retriever import HybridRetriever
from src.rag.cache import QueryCache
from src.rag.prompts import SYSTEM_PROMPT, build_rag_prompt, format_context_with_labels, format_sources
from src.parser.registry import ParserRegistry
from src.parser.chunker import DocumentChunker
from src.parser.pdf import PDFParser
from src.parser.docx import DocxParser
from src.parser.pptx import PPTXParser
from src.parser.html import HTMLParser
from src.parser.ipynb import IPYNBParser
from src.parser.txt import TXTParser
from src.parser.markdown import MarkdownParser
from src.parser.csv_parser import CSVParser

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("course_assistant")

# ========== 全局组件 ==========
session_mgr = embedding_fn = chroma_client = vectorstore = collection = _llm = None
router = hybrid_retriever = query_cache = parser_registry = document_chunker = None


# ========== 兼容旧版意图路由 ==========
class EmbeddingIntentRouter:
    def __init__(self, vs): self.vs = vs
    def route(self, msg, k=3):
        results = self.vs.similarity_search_with_relevance_scores(msg, k=k)
        if not results: return "casual", []
        if results[0][1] >= RAG_THRESHOLD: return "rag", [d for d, _ in results]
        return "casual", []


# ========== 请求/响应模型 ==========
class ChatRequest(BaseModel): message: str; session_id: str = "default"
class ChatResponse(BaseModel): answer: str; source: str = "rag"; docs: list[str] = []
class UploadResponse(BaseModel): status: str; filename: str; chunks: int
class SearchRequest(BaseModel): query: str; max_results: int = 5


# ========== LLM 辅助 ==========
async def _llm_invoke(msgs): return await asyncio.to_thread(_llm.invoke, msgs)


# ========== 生命周期 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_mgr, embedding_fn, chroma_client, vectorstore, collection, _llm, router
    global hybrid_retriever, query_cache, parser_registry, document_chunker

    logger.info("=" * 50)
    logger.info("AIGC 课程助手 v0.6.0 启动中...")
    logger.info("=" * 50)

    session_mgr = SessionManager()
    logger.info("[1/7] 会话管理器就绪")

    embedding_fn = InstructorEmbedding(MODEL_PATH, device=EMBEDDING_DEVICE)
    logger.info("[2/7] 嵌入模型就绪 (device=%s)", EMBEDDING_DEVICE)

    chroma_client = PersistentClient(path=CHROMA_PATH)
    vectorstore = Chroma(client=chroma_client, collection_name=COLLECTION_NAME,
                         embedding_function=embedding_fn, persist_directory=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata=COLLECTION_METADATA)
    logger.info("[3/7] ChromaDB 就绪 (%d 文档)", collection.count())

    hybrid_retriever = HybridRetriever(vectorstore, embedding_fn)
    hybrid_retriever.build_bm25_index()
    logger.info("[4/7] 混合检索器就绪 (BM25=%d, RRF k=%d)", hybrid_retriever.bm25.doc_count, hybrid_retriever.rrf_k)

    parser_registry = ParserRegistry()
    parser_registry.register(PDFParser()); parser_registry.register(DocxParser())
    parser_registry.register(PPTXParser()); parser_registry.register(HTMLParser())
    parser_registry.register(IPYNBParser()); parser_registry.register(TXTParser())
    parser_registry.register(MarkdownParser()); parser_registry.register(CSVParser())
    document_chunker = DocumentChunker()
    logger.info("[5/7] 解析器就绪 (%d 种格式)", parser_registry.parser_count)

    _llm = ChatOpenAI(model=DEEPSEEK_MODEL, base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY,
                      temperature=0.7, timeout=60, max_retries=2, streaming=True)
    query_cache = QueryCache()
    logger.info("[6/7] LLM 就绪: %s", DEEPSEEK_MODEL)

    router = EmbeddingIntentRouter(vectorstore)
    logger.info("[7/7] 意图路由就绪 (阈值=%.2f)", RAG_THRESHOLD)
    logger.info("=" * 50)

    yield
    logger.info("正在关闭...")


app = FastAPI(title="AIGC助手 API", version="0.6.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ========== 端点 ==========
@app.get("/health")
async def health():
    dc = collection.count() if collection else 0
    bd = hybrid_retriever.bm25.doc_count if hybrid_retriever else 0
    cs = query_cache.stats if query_cache else {}
    return {"status":"ok","service":"AIGC助手 API","version":"0.6.0","llm":DEEPSEEK_MODEL,"docs_in_kb":dc,"bm25_docs":bd,"cache":cs}


@app.get("/sessions")
async def list_sessions(user_name=""): return {"sessions": session_mgr.list_sessions(user_name)}


@app.post("/sessions")
async def create_session(data: dict):
    sid = session_mgr.register_session(name=data.get("name",""), user_name=data.get("user_name",""))
    return {"session_id": sid}


@app.get("/sessions/{sid}/history")
async def session_history(sid: str): return session_mgr.get_session_history(sid)


# ========== 流式聊天 ==========
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    msg = req.message.strip(); sid = req.session_id or "default"
    intent, docs = router.route(msg)

    history = session_mgr.get_history(sid)
    user_name = session_mgr.get_user_name(sid)
    extracted = session_mgr.extract_user_name(msg)
    if extracted: user_name = extracted; session_mgr.set_user_name(sid, user_name)

    async def gen():
        full = ""
        try:
            if intent == "rag":
                rr = hybrid_retriever.retrieve(msg, top_k=5)
                if rr:
                    ctx = format_context_with_labels(rr); st = format_sources(rr); sl = st.split("\n")
                else:
                    ctx = "\n\n".join(d.page_content for d in docs)
                    sl = list({d.metadata.get("source","未知") for d in docs})
                ht = ""
                for m in history[-6:]: ht += f"{'用户' if m['role']=='user' else '小课'}：{m['content']}\n"
                prompt = build_rag_prompt(SYSTEM_PROMPT, msg, ctx, ht, user_name or "未知",
                                          format_sources(rr) if rr else "\n".join(f"[{i+1}] {s}" for i,s in enumerate(sl)))
                async for chunk in _llm.astream([HumanMessage(content=prompt)]):
                    t = chunk.content
                    if t: full += t; yield f"data: {json.dumps({'token':t,'done':False},ensure_ascii=False)}\n\n"
                session_mgr.append_history(sid, "user", msg)
                session_mgr.append_history(sid, "assistant", full)
                yield f"data: {json.dumps({'token':'','done':True,'source':'rag','docs':sl},ensure_ascii=False)}\n\n"
            else:
                msgs = session_mgr.build_messages(history, msg, SYSTEM_PROMPT, user_name)
                async for chunk in _llm.astream(msgs):
                    t = chunk.content
                    if t: full += t; yield f"data: {json.dumps({'token':t,'done':False},ensure_ascii=False)}\n\n"
                session_mgr.append_history(sid, "user", msg)
                session_mgr.append_history(sid, "assistant", full)
                yield f"data: {json.dumps({'token':'','done':True,'source':'casual','docs':[]},ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("流式异常")
            yield f"data: {json.dumps({'token':'⚠️ 服务暂不可用','done':True,'source':'error'},ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})


# ========== 非流式聊天 ==========
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    msg = req.message.strip(); sid = req.session_id or "default"
    intent, docs = router.route(msg)
    history = session_mgr.get_history(sid)
    user_name = session_mgr.get_user_name(sid)
    extracted = session_mgr.extract_user_name(msg)
    if extracted: user_name = extracted; session_mgr.set_user_name(sid, user_name)

    if intent == "rag":
        try:
            rr = hybrid_retriever.retrieve(msg, top_k=5)
            if rr:
                ctx = format_context_with_labels(rr); sl = format_sources(rr).split("\n")
            else:
                ctx = "\n\n".join(d.page_content for d in docs)
                sl = list({d.metadata.get("source","未知") for d in docs})
            ht = ""
            for m in history[-6:]: ht += f"{'用户' if m['role']=='user' else '小课'}：{m['content']}\n"
            prompt = build_rag_prompt(SYSTEM_PROMPT, msg, ctx, ht, user_name or "未知",
                                      format_sources(rr) if rr else "")
            resp = await _llm_invoke([HumanMessage(content=prompt)])
            answer = resp.content
            session_mgr.append_history(sid, "user", msg)
            session_mgr.append_history(sid, "assistant", answer)
            return ChatResponse(answer=answer, source="rag", docs=sl)
        except Exception as e:
            logger.exception("RAG 异常")
            return ChatResponse(answer="抱歉，知识库检索暂不可用。", source="fallback")
    else:
        try:
            msgs = session_mgr.build_messages(history, msg, SYSTEM_PROMPT, user_name)
            resp = await _llm_invoke(msgs)
            answer = resp.content
            session_mgr.append_history(sid, "user", msg)
            session_mgr.append_history(sid, "assistant", answer)
            return ChatResponse(answer=answer, source="casual")
        except Exception as e:
            logger.exception("闲聊异常")
            return ChatResponse(answer="抱歉，AI 服务暂不可用。", source="fallback")


# ========== 文件上传 ==========
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not parser_registry.supports(file.filename):
        raise HTTPException(400, f"不支持格式，支持: {', '.join(parser_registry.supported_extensions)}")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"文件不能超过 {MAX_FILE_SIZE//(1024*1024)}MB")
    tmp = os.path.join("/tmp", file.filename)
    try:
        with open(tmp, "wb") as f: f.write(content)
        result = parser_registry.get_parser(tmp).parse(tmp)
        if result.errors:
            for e in result.errors: logger.warning("解析警告 [%s]: %s", file.filename, e)
        if not result.success: raise HTTPException(400, "文件内容为空")
        chunks = document_chunker.chunk(result, os.path.splitext(file.filename)[1].lower(), file.filename)
        if not chunks: raise HTTPException(400, "分块结果为空")
    finally:
        if os.path.exists(tmp): os.remove(tmp)

    vectorstore.add_texts(texts=[c["text"] for c in chunks], metadatas=[c["metadata"] for c in chunks])
    logger.info("已写入 %d 片段 (%s), %d 表格", len(chunks), file.filename, len(result.tables))
    try:
        hybrid_retriever.rebuild_bm25_index()
        logger.info("BM25 已重建: %d 文档", hybrid_retriever.bm25.doc_count)
    except Exception as e: logger.warning("BM25 重建失败: %s", e)
    return UploadResponse(status="success", filename=file.filename, chunks=len(chunks))


# ========== 网络搜索 ==========
_search_engine = _search_cache = _rate_limiter = None

def _get_search():
    global _search_engine, _search_cache, _rate_limiter
    if _search_engine is None:
        from src.search.engine import WebSearchEngine
        from src.search.cache import SearchCache, RateLimiter
        _search_engine = WebSearchEngine()
        _search_cache = SearchCache()
        _rate_limiter = RateLimiter()
    return _search_engine, _search_cache, _rate_limiter


@app.post("/search")
async def web_search(req: SearchRequest):
    engine, cache, limiter = _get_search()
    q = req.query.strip()
    if not q: raise HTTPException(400, "查询不能为空")
    cached = cache.get(q)
    if cached:
        return {"query": q, "source": "cache", "results": [{"title":r.title,"url":r.url,"snippet":r.snippet,"source":r.source} for r in cached]}
    try: await limiter.acquire()
    except RuntimeError as e: raise HTTPException(429, str(e))
    results = await engine.search(q, req.max_results)
    if results: cache.put(q, results)
    return {"query": q, "source": "live", "results": [{"title":r.title,"url":r.url,"snippet":r.snippet,"source":r.source} for r in results], "count": len(results)}


# ========== 启动 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
