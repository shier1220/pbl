"""
AIGC 课程助手 API v0.6.0 — FastAPI 入口
"""
import os, sys, json, logging, asyncio
from contextlib import asynccontextmanager
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
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
from src.rag.prompts import SYSTEM_PROMPT
from src.rag.generator import RAGGenerator
from src.rag.query_expander import QueryCondenser
from src.intent import Intent, IntentResult, IntentRouter, get_prompt_for_intent
from src.search.engine import WebSearchEngine
from src.search.cache import RateLimiter
from src.search.attribution import format_search_context, format_search_sources
from src.auth import UserService, get_current_user, UserRegister, UserLogin, TokenResponse, UserInfo
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

from datetime import datetime

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("course_assistant")

# 当前日期（注入 prompt，解决 LLM 不知道时间的问题）
TODAY_STR = datetime.now().strftime("%Y年%m月%d日 %A")

# ========== 全局组件 ==========
session_mgr = embedding_fn = chroma_client = vectorstore = collection = _llm = None
router = hybrid_retriever = query_cache = parser_registry = document_chunker = None
intent_router = rag_generator = query_condenser = None
search_engine = rate_limiter = None
user_svc = None


# ========== 请求/响应模型 ==========
class ChatRequest(BaseModel): message: str; session_id: str = "default"
class ChatResponse(BaseModel): answer: str; source: str = "rag"; docs: list = []
class UploadResponse(BaseModel): status: str; filename: str; chunks: int
class SearchRequest(BaseModel): query: str; max_results: int = 5


# ========== LLM 辅助 ==========
async def _llm_invoke(msgs): return await asyncio.to_thread(_llm.invoke, msgs)


# ========== 生命周期 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_mgr, embedding_fn, chroma_client, vectorstore, collection, _llm, router
    global hybrid_retriever, query_cache, parser_registry, document_chunker
    global intent_router, rag_generator, query_condenser
    global search_engine, rate_limiter, user_svc

    logger.info("=" * 50)
    logger.info("AIGC 课程助手 v0.7.0 启动中...")
    logger.info("=" * 50)

    user_svc = UserService()
    logger.info("[1/9] 用户服务就绪")

    session_mgr = SessionManager()
    logger.info("[2/9] 会话管理器就绪")

    embedding_fn = InstructorEmbedding(MODEL_PATH, device=EMBEDDING_DEVICE)
    logger.info("[3/9] 嵌入模型就绪 (device=%s)", EMBEDDING_DEVICE)

    chroma_client = PersistentClient(path=CHROMA_PATH)
    vectorstore = Chroma(client=chroma_client, collection_name=COLLECTION_NAME,
                         embedding_function=embedding_fn, persist_directory=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata=COLLECTION_METADATA)
    logger.info("[4/9] ChromaDB 就绪 (%d 文档)", collection.count())

    hybrid_retriever = HybridRetriever(vectorstore, embedding_fn)
    hybrid_retriever.build_bm25_index()
    logger.info("[5/9] 混合检索器就绪 (BM25=%d, RRF k=%d)", hybrid_retriever.bm25.doc_count, hybrid_retriever.rrf_k)

    parser_registry = ParserRegistry()
    parser_registry.register(PDFParser()); parser_registry.register(DocxParser())
    parser_registry.register(PPTXParser()); parser_registry.register(HTMLParser())
    parser_registry.register(IPYNBParser()); parser_registry.register(TXTParser())
    parser_registry.register(MarkdownParser()); parser_registry.register(CSVParser())
    document_chunker = DocumentChunker()
    logger.info("[6/9] 解析器就绪 (%d 种格式)", parser_registry.parser_count)

    _llm = ChatOpenAI(model=DEEPSEEK_MODEL, base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY,
                      temperature=0.7, timeout=60, max_retries=2, streaming=True)
    query_cache = QueryCache()
    logger.info("[7/9] LLM 就绪: %s", DEEPSEEK_MODEL)

    # 新：5 类意图路由 + LLM 回退验证
    intent_router = IntentRouter(vectorstore, _llm, embedding_fn)
    rag_generator = RAGGenerator(_llm, session_mgr)
    query_condenser = QueryCondenser(_llm)
    router = intent_router  # 向后兼容
    logger.info("[8/9] 意图路由就绪 (嵌入k-NN + LLM few-shot)")

    # 新：网络搜索引擎 + 缓存 + 限流
    search_engine = WebSearchEngine(llm=_llm)
    rate_limiter = RateLimiter()
    logger.info("[9/9] 网络搜索就绪 (DDG + Bing备用)")

    logger.info("=" * 50)

    yield
    logger.info("正在关闭...")


app = FastAPI(title="AIGC助手 API", version="0.7.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ========== 端点 ==========
@app.get("/health")
async def health():
    dc = collection.count() if collection else 0
    bd = hybrid_retriever.bm25.doc_count if hybrid_retriever else 0
    cs = query_cache.stats if query_cache else {}
    ir = intent_router.classifier.__class__.__name__ if intent_router else "N/A"
    return {"status":"ok","service":"AIGC助手 API","version":"0.7.0","llm":DEEPSEEK_MODEL,
            "docs_in_kb":dc,"bm25_docs":bd,"cache":cs,
            "intent_classifier":ir,"search_engine":"DDG+Bing"}


# ========== 认证端点 ==========
@app.post("/register", status_code=201)
async def register(data: UserRegister):
    try:
        info = user_svc.register(data.username, data.password)
        return info
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/login")
async def login(data: UserLogin):
    user = user_svc.authenticate(data.username, data.password)
    if not user:
        raise HTTPException(401, "用户名或密码错误")
    token = UserService.create_token(user["user_id"], user["username"])
    return TokenResponse(access_token=token, username=user["username"])


@app.get("/me")
async def me(user: dict = Depends(get_current_user)):
    info = user_svc.get_user(user["user_id"])
    return info or {"error": "用户不存在"}


# ========== 会话端点 ==========
@app.get("/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    return {"sessions": session_mgr.list_sessions(user_id=user["user_id"])}


@app.post("/sessions")
async def create_session(data: dict, user: dict = Depends(get_current_user)):
    sid = session_mgr.register_session(
        name=data.get("name", ""),
        user_name=user["username"],
        user_id=user["user_id"],
    )
    return {"session_id": sid}


@app.get("/sessions/{sid}/history")
async def session_history(sid: str, user: dict = Depends(get_current_user)):
    owner = session_mgr.get_session_owner(sid)
    if owner is not None and owner != user["user_id"]:
        raise HTTPException(403, "无权访问此会话")
    return session_mgr.get_session_history(sid)


# ========== 流式聊天 ==========
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, user: dict = Depends(get_current_user)):
    msg = req.message.strip()
    # 解析 session_id：default → 用户的默认会话；指定 sid → 校验归属
    if not req.session_id or req.session_id == "default":
        sid = session_mgr.get_or_create_default_session(user["user_id"], user["username"])
    else:
        sid = req.session_id
        owner = session_mgr.get_session_owner(sid)
        if owner is not None and owner != user["user_id"]:
            raise HTTPException(403, "无权访问此会话")
        # 未知会话自动归属当前用户
        if owner is None:
            session_mgr.claim_session(sid, user["user_id"])

    history = session_mgr.get_history(sid)
    user_name = session_mgr.get_user_name(sid)
    extracted = session_mgr.extract_user_name(msg)
    if extracted: user_name = extracted; session_mgr.set_user_name(sid, user_name)

    # 快速规则先拦截（0延迟），未命中再 LLM + 搜索并行
    quick = intent_router.classifier.classify_sync(msg)
    if quick.method.startswith("quick_"):
        intent_result = quick
        search_task = None  # 城市名/赛事名不需要搜索预取
    else:
        intent_task = asyncio.create_task(intent_router.route(msg, history, sid))
        search_task = asyncio.create_task(search_engine.search_with_answer(msg))
        try:
            intent_result = await asyncio.wait_for(intent_task, timeout=1.5)
        except asyncio.TimeoutError:
            intent_result = quick  # LLM 超时 → 快速规则结果
    logger.info("[stream] intent=%s conf=%.2f method=%s", intent_result.intent.value, intent_result.confidence, intent_result.method)

    cache_key = f"{intent_result.intent.value}:{msg}"
    cached = query_cache.get(cache_key)

    async def gen():
        try:
            if cached:
                full_cached = "".join(cached["tokens"])
                if not history or history[-1][0] != "user" or history[-1][1] != msg:
                    session_mgr.append_history(sid, "user", msg, user["user_id"])
                session_mgr.append_history(sid, "assistant", full_cached, user["user_id"])
                for token_chunk in cached["tokens"]:
                    yield f"data: {json.dumps({'token':token_chunk,'done':False},ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'token':'','done':True,'source':cached.get('source','rag'),'docs':cached.get('docs',[])},ensure_ascii=False)}\n\n"
                return
        except Exception:
            pass

        full = ""
        # 等待搜索结果（如果有预取）
        wr, bocha_answer = await search_task if search_task else (await search_engine.search_with_answer(msg))

        try:
            # ── COURSE_QUESTION：混合检索 + RAG 生成 ──
            if intent_result.intent == Intent.COURSE_QUESTION:
                condensed = await query_condenser.condense(msg, history) if history else msg
                rr = hybrid_retriever.retrieve(condensed, top_k=5)

                # 低置信度时补充网络搜索
                web_ctx = ""
                if intent_result.confidence < 0.6:
                    try:
                        wr = await search_engine.search(msg, max_results=3)
                        if wr:
                            web_ctx = "## 🌐 网络搜索结果\n" + format_search_context(wr)
                    except Exception: pass

                async for event in rag_generator.stream_rag(msg, rr, sid, history, user_name, web_context=web_ctx, user_id=user["user_id"]):
                    yield event

            # ── WEB_SEARCH：Bocha 流式 → 非流式 → Bing+DeepSeek ──
            elif intent_result.intent == Intent.WEB_SEARCH:
                tokens = []

                # 1. Bocha 流式（边搜边推，最快）
                async for token in search_engine.bocha.search_stream(msg):
                    full += token; tokens.append(token)
                    yield f"data: {json.dumps({'token':token,'done':False},ensure_ascii=False)}\n\n"
                if tokens:
                    session_mgr.append_history(sid, "user", msg, user["user_id"])
                    session_mgr.append_history(sid, "assistant", full, user["user_id"])
                    yield f"data: {json.dumps({'token':'','done':True,'source':'web_search','docs':[]},ensure_ascii=False)}\n\n"
                    return

                # 2. Bocha 非流式 / Bing + DeepSeek
                wr, bocha_answer = await search_task if search_task else (await search_engine.search_with_answer(msg))
                search_docs = [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in wr] if wr else []
                answer_text = bocha_answer
                if not answer_text:
                    ctx = format_search_context(wr) if wr else "（无搜索结果）"
                    prompt = f"{SYSTEM_PROMPT}\n\n## 网络搜索结果\n{ctx}\n\n## 用户问题\n{msg}\n\n## ⚠️ 严格规则\n1. 仅根据搜索结果回答，**禁止编造**\n2. 搜索结果中没有的信息，直接说「搜索结果未包含该信息」\n3. 标注来源编号 [1] [2]"
                    async for chunk in _llm.astream([HumanMessage(content=prompt)]):
                        t = chunk.content
                        if t: full += t; tokens.append(t); yield f"data: {json.dumps({'token':t,'done':False},ensure_ascii=False)}\n\n"
                    answer_text = full[len(full)-sum(len(t) for t in tokens):] if tokens else full
                    query_cache.put(cache_key, {"tokens": tokens, "source": "web_search", "docs": search_docs})
                else:
                    for ch in bocha_answer:
                        full += ch; tokens.append(ch)
                        yield f"data: {json.dumps({'token':ch,'done':False},ensure_ascii=False)}\n\n"
                session_mgr.append_history(sid, "user", msg, user["user_id"])
                session_mgr.append_history(sid, "assistant", full, user["user_id"])
                yield f"data: {json.dumps({'token':'','done':True,'source':'web_search','docs':search_docs},ensure_ascii=False)}\n\n"

            # ── CASUAL_CHAT / FILE_OPERATION / SYSTEM_COMMAND ──
            else:
                prompt = get_prompt_for_intent(intent_result.intent)
                msgs = session_mgr.build_messages(history, msg, prompt, user_name)
                tokens = []
                async for chunk in _llm.astream(msgs):
                    t = chunk.content
                    if t: full += t; tokens.append(t); yield f"data: {json.dumps({'token':t,'done':False},ensure_ascii=False)}\n\n"
                session_mgr.append_history(sid, "user", msg, user["user_id"])
                session_mgr.append_history(sid, "assistant", full, user["user_id"])
                query_cache.put(cache_key, {"tokens": tokens, "source": intent_result.intent.value, "docs": []})
                yield f"data: {json.dumps({'token':'','done':True,'source':intent_result.intent.value,'docs':[]},ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.exception("流式异常: %s", e)
            yield f"data: {json.dumps({'token':'⚠️ 服务暂不可用','done':True,'source':'error'},ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})


# ========== 非流式聊天 ==========
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    msg = req.message.strip()
    # 解析 session_id：default → 用户的默认会话；指定 sid → 校验归属
    if not req.session_id or req.session_id == "default":
        sid = session_mgr.get_or_create_default_session(user["user_id"], user["username"])
    else:
        sid = req.session_id
        owner = session_mgr.get_session_owner(sid)
        if owner is not None and owner != user["user_id"]:
            raise HTTPException(403, "无权访问此会话")
        # 未知会话自动归属当前用户
        if owner is None:
            session_mgr.claim_session(sid, user["user_id"])

    history = session_mgr.get_history(sid)
    user_name = session_mgr.get_user_name(sid)
    extracted = session_mgr.extract_user_name(msg)
    if extracted: user_name = extracted; session_mgr.set_user_name(sid, user_name)

    # 并行：意图分类 + 搜索
    intent_task = asyncio.create_task(intent_router.route(msg, history, sid))
    search_prefetch = asyncio.create_task(search_engine.search_with_answer(msg))
    intent_result = await intent_task
    logger.info("[chat] intent=%s conf=%.2f method=%s", intent_result.intent.value, intent_result.confidence, intent_result.method)

    # 缓存检查
    cache_key = f"{intent_result.intent.value}:{msg}"
    cached = query_cache.get(cache_key)
    if cached:
        full_answer = "".join(cached["tokens"])
        return ChatResponse(answer=full_answer, source=cached.get("source", "rag"),
                          docs=cached.get("docs", []))

    try:
        # ── COURSE_QUESTION：混合检索 + RAG ──
        if intent_result.intent == Intent.COURSE_QUESTION:
            condensed = await query_condenser.condense(msg, history) if history else msg
            rr = hybrid_retriever.retrieve(condensed, top_k=5)
            # 低置信度时补充网络搜索
            web_ctx = ""
            if intent_result.confidence < 0.6:
                try:
                    wr = await search_engine.search(msg, max_results=3)
                    if wr:
                        web_ctx = "## 🌐 网络搜索结果\n" + format_search_context(wr)
                except Exception: pass
            response = await rag_generator.generate_rag(msg, rr, sid, history, user_name, web_context=web_ctx, user_id=user["user_id"])
            query_cache.put(cache_key, {"tokens": [response["answer"]],
                            "source": response.get("source", "rag"),
                            "docs": response.get("docs", [])})
            return ChatResponse(answer=response["answer"], source=response.get("source", "rag"),
                              docs=response.get("docs", []))

        # ── WEB_SEARCH：Bocha 直接回答 / Bing + DeepSeek 总结 ──
        elif intent_result.intent == Intent.WEB_SEARCH:
            wr, bocha_answer = await search_prefetch
            search_docs = [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in wr] if wr else []

            if bocha_answer:
                answer = bocha_answer
            else:
                ctx = format_search_context(wr) if wr else "（无搜索结果）"
                prompt = f"{SYSTEM_PROMPT}\n\n## 网络搜索结果\n{ctx}\n\n## 用户问题\n{msg}\n\n## ⚠️ 严格规则\n1. 仅根据搜索结果回答，**禁止编造**\n2. 搜索结果中没有的信息，直接说「搜索结果未包含该信息」\n3. 标注来源编号 [1] [2]"
                resp = await _llm_invoke([HumanMessage(content=prompt)])
                answer = resp.content
                query_cache.put(cache_key, {"tokens": [answer], "source": "web_search", "docs": search_docs})

            session_mgr.append_history(sid, "user", msg, user["user_id"])
            session_mgr.append_history(sid, "assistant", answer, user["user_id"])
            return ChatResponse(answer=answer, source="web_search", docs=search_docs)

        # ── CASUAL_CHAT / FILE_OPERATION / SYSTEM_COMMAND ──
        else:
            prompt = get_prompt_for_intent(intent_result.intent)
            msgs = session_mgr.build_messages(history, msg, prompt, user_name)
            resp = await _llm_invoke(msgs)
            answer = resp.content
            session_mgr.append_history(sid, "user", msg, user["user_id"])
            session_mgr.append_history(sid, "assistant", answer, user["user_id"])
            query_cache.put(cache_key, {"tokens": [answer], "source": intent_result.intent.value, "docs": []})
            return ChatResponse(answer=answer, source=intent_result.intent.value)

    except Exception as e:
        logger.exception("[chat] 异常: %s", e)
        return ChatResponse(answer="抱歉，服务暂不可用。", source="fallback")


# ========== 文件上传 ==========
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
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

    # 过滤 metadata 中的 None 值（ChromaDB 不接受）
    clean_metadatas = [{k: v for k, v in c["metadata"].items() if v is not None} for c in chunks]
    vectorstore.add_texts(texts=[c["text"] for c in chunks], metadatas=clean_metadatas)
    logger.info("已写入 %d 片段 (%s), %d 表格", len(chunks), file.filename, len(result.tables))
    try:
        hybrid_retriever.rebuild_bm25_index()
        logger.info("BM25 已重建: %d 文档", hybrid_retriever.bm25.doc_count)
    except Exception as e: logger.warning("BM25 重建失败: %s", e)
    return UploadResponse(status="success", filename=file.filename, chunks=len(chunks))


# ========== 网络搜索 ==========
@app.post("/search")
async def web_search(req: SearchRequest, user: dict = Depends(get_current_user)):
    q = req.query.strip()
    if not q: raise HTTPException(400, "查询不能为空")
    # 使用 lifespan 中初始化的全局组件
    cache_key = f"search:{q}"
    cached = query_cache.get(cache_key)
    if cached:
        return {"query": q, "source": "cache", "results": cached}
    try: await rate_limiter.acquire()
    except RuntimeError as e: raise HTTPException(429, str(e))
    results = await search_engine.search(q, req.max_results)
    if results:
        serialized = [{"title":r.title,"url":r.url,"snippet":r.snippet,"source":r.source} for r in results]
        query_cache.put(cache_key, serialized)
        return {"query": q, "source": "live", "results": serialized, "count": len(results)}
    return {"query": q, "source": "live", "results": [], "count": 0}


# ========== 启动 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
