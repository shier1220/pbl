"""
课程助手 API 后端
FastAPI + DeepSeek 云端 API + ChromaDB RAG 问答链路
"""

import os
import re
import json
import logging
import asyncio
import sqlite3
from pathlib import Path
from typing import List, Optional, AsyncGenerator

import fitz  # PyMuPDF
from docx import Document
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from embedding import InstructorEmbedding, MODEL_PATH

# ========== 0. 环境 & 日志配置 ==========
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("course_assistant")

# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

# 路径
BASE_DIR = Path(__file__).resolve().parent      # ChromaDB 和 SQLite 数据库文件都放在这里
CHROMA_PATH = str(BASE_DIR / "chroma_db")
DB_PATH = str(BASE_DIR / "sessions.db")

# ========== 1. 创建 FastAPI 应用 ==========
app = FastAPI(
    title="课程助手 API",
    description="AI Engineer Mentor — 课程问答后端服务 (DeepSeek 云端)",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 2. 请求/响应模型 ==========
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    source: str = "rag"
    docs: list[str] = []


class UploadResponse(BaseModel):
    status: str
    filename: str
    chunks: int


# ========== 3. 助手人设 ==========
SYSTEM_PROMPT = """你是课程导师"小课"，专门辅导AIGC大模型应用工程师课程。

## 关于用户名字（非常重要）
对话历史中如果看到"我是XXX"或"我叫XXX"——这就是用户的名字。
当用户问"我叫什么名字"时，从历史中提取这个名字回答。
如果历史中没有名字信息，说："你还没有告诉我你的名字"。
绝对不要用"小课"回答用户的名字问题——那是你的名字。

## 回答风格
- 技术问题：准确有深度，优先引用课程资料
- 闲聊/非课程问题：正常回答即可。你背后是通用大模型，历史、科普、编程答疑都可以聊。
  只有确实做不到的事（查实时天气、搜新闻、股票价格）才坦率说明
  你的定位是AIGC课程导师，回答完闲聊后如果合适，可以顺带引导回课程话题
- 不知道就说不知道，不编造
- 中文为主，简洁清晰

## 课程范围
微调实战（LoRA/QLoRA）、模型蒸馏、数据增强、RAG、vLLM部署、深度学习基础
"""

# ========== 4. 会话持久化（SQLite）==========
MAX_HISTORY = 10


def _init_db():
    """建表（幂等）"""
    with sqlite3.connect(DB_PATH) as conn:
        # 消息表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT,
                role       TEXT,
                content    TEXT,
                seq        INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now','localtime'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_id ON sessions(session_id)"
        )
        # 会话元信息（用户名）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_meta (
                session_id TEXT PRIMARY KEY,
                user_name  TEXT
            )
        """)
        # 会话注册表（名称 + 创建时间）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_registry (
                session_id TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                user_name  TEXT DEFAULT '',
                created_at REAL DEFAULT (unixepoch())
            )
        """)
    logger.info("会话数据库就绪")


def get_history(session_id: str) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content FROM sessions WHERE session_id=? ORDER BY seq",
            (session_id,),
        ).fetchall()
    return [{"role": r, "content": c} for r, c in rows]


def _ensure_session_registered(session_id: str):
    """确保会话在 registry 中有记录（首次发消息时自动注册）"""
    with sqlite3.connect(DB_PATH) as conn:
        exists = conn.execute(
            "SELECT 1 FROM session_registry WHERE session_id=?", (session_id,)
        ).fetchone()
        if not exists:
            conn.execute(
                """INSERT INTO session_registry (session_id, name, user_name, created_at)
                   VALUES (?, ?, '', unixepoch())""",
                (session_id, f"会话{session_id[:6]}"),
            )


def append_history(session_id: str, role: str, content: str):
    _ensure_session_registered(session_id)
    with sqlite3.connect(DB_PATH) as conn:
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO sessions (session_id, role, content, seq) VALUES (?,?,?,?)",
            (session_id, role, content, seq),
        )
        conn.execute(
            """
            DELETE FROM sessions WHERE session_id=? AND seq NOT IN (
                SELECT seq FROM sessions WHERE session_id=? ORDER BY seq DESC LIMIT ?
            )
            """,
            (session_id, session_id, MAX_HISTORY),
        )


def build_messages(history: list, current_msg: str, user_name: str = "") -> list:
    system_text = SYSTEM_PROMPT
    if user_name:
        system_text += f"\n\n当前用户的名字是：{user_name}"
    messages = [SystemMessage(content=system_text)]
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=current_msg))
    return messages


# ── 用户名提取（预编译正则）──
_NAME_PATTERNS = [
    re.compile(pat)
    for pat in [
        r"我是(.+?)[，。！？\s]",
        r"我叫(.+?)[，。！？\s]",
        r"^我是(.+)$",
        r"^我叫(.+)$",
        r"我叫(.+)",
        r"我是(.+)",
        r"称呼我(.+)",
    ]
]


def extract_user_name(msg: str) -> Optional[str]:
    for pat in _NAME_PATTERNS:
        m = pat.search(msg.strip())
        if m:
            name = m.group(1).strip()
            if len(name) <= 10 and name:
                return name
    return None


def get_user_name(session_id: str) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT user_name FROM session_meta WHERE session_id=?", (session_id,)
        ).fetchone()
    return row[0] if row else ""


def set_user_name(session_id: str, user_name: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO session_meta (session_id, user_name) VALUES (?,?)",
            (session_id, user_name),
        )


# ========== 5. 意图路由（复用 RAG 检索嵌入，一次调用同时完成路由+检索）==========

RAG_THRESHOLD = 0.69  # instructor-xl 基线~0.65，设 0.69 区分相关(≥0.70)与无关(≤0.66)


class EmbeddingIntentRouter:
    """用 ChromaDB 检索的嵌入分数判断意图，避免重复嵌入计算"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def route(self, message: str, k: int = 3):
        """
        一次嵌入调用，同时完成：
        1. 意图判断（Top-1 分数 vs 阈值）
        2. 知识库检索（Top-k 文档）

        返回: (intent: "rag"|"casual", docs: list)
        """
        results = self.vectorstore.similarity_search_with_relevance_scores(
            message, k=k
        )
        if not results:
            logger.debug("[Router] 知识库为空 → casual")
            return "casual", []

        top_doc, top_score = results[0]
        if top_score >= RAG_THRESHOLD:
            docs = [d for d, _ in results]
            logger.info(
                "[Router] '%s...' → RAG (score=%.4f, threshold=%.2f)",
                message[:30], top_score, RAG_THRESHOLD,
            )
            return "rag", docs

        logger.info(
            "[Router] '%s...' → casual (score=%.4f, threshold=%.2f)",
            message[:30], top_score, RAG_THRESHOLD,
        )
        return "casual", []


# ========== 6. 文本分块（使用 LangChain 的标准分块器）==========
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)


def split_text(text: str) -> List[str]:
    return text_splitter.split_text(text)


# ========== 7. 文件解析 ==========
def parse_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    try:
        text = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()
    return text


def parse_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ========== 8. 初始化全局组件 ==========
logger.info("正在初始化服务组件...")

# 嵌入模型
embedding_fn = InstructorEmbedding(MODEL_PATH)
logger.info("嵌入模型 instructor-xl 加载完成")

# ChromaDB
chroma_client = PersistentClient(path=CHROMA_PATH)
vectorstore = Chroma(
    client=chroma_client,
    collection_name="course_knowledge",
    embedding_function=embedding_fn,
    persist_directory=CHROMA_PATH,
)
collection = chroma_client.get_or_create_collection(
    name="course_knowledge",
    metadata={
        "description": "AIGC课程助手知识库",
        "hnsw:space": "cosine",   # 余弦距离 → 只看语义方向，不受向量模长影响
    },
)
logger.info("ChromaDB 就绪，当前文档数: %d", collection.count())

# LLM — DeepSeek 云端 API
_llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
    temperature=0.7,
    timeout=60,
    max_retries=2,
    streaming=True,   # ← 开启流式
)
logger.info("LLM 就绪: %s @ %s", DEEPSEEK_MODEL, DEEPSEEK_BASE_URL)

# 意图路由器（复用 ChromaDB 检索嵌入）
router = EmbeddingIntentRouter(vectorstore)
logger.info("嵌入意图路由器就绪（阈值=%.2f）", RAG_THRESHOLD)

# 初始化数据库
_init_db()


# ── 封装 LLM 异步调用 ──
async def _llm_invoke(messages: list):
    """在线程池中执行同步 LLM 调用，避免阻塞 FastAPI 事件循环"""
    return await asyncio.to_thread(_llm.invoke, messages)


# ========== 9. 接口 ==========
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "课程助手 API",
        "llm": DEEPSEEK_MODEL,
        "docs_in_kb": collection.count(),
    }


# ── 会话管理接口 ──

@app.get("/sessions")
async def list_sessions(user_name: str = ""):
    """获取会话列表（支持按用户名过滤）"""
    with sqlite3.connect(DB_PATH) as conn:
        if user_name:
            rows = conn.execute(
                """SELECT session_id, name, user_name, created_at
                   FROM session_registry
                   WHERE user_name = ?
                   ORDER BY created_at DESC""",
                (user_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT session_id, name, user_name, created_at
                   FROM session_registry
                   ORDER BY created_at DESC"""
            ).fetchall()
    return {
        "sessions": [
            {
                "id": r[0],
                "name": r[1],
                "user_name": r[2] or "",
                "created_at": r[3],
            }
            for r in rows
        ]
    }


@app.post("/sessions")
async def create_session(data: dict):
    """创建新会话"""
    import uuid

    session_id = str(uuid.uuid4())
    name = data.get("name", f"会话{session_id[:6]}")
    user_name = data.get("user_name", "")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO session_registry (session_id, name, user_name, created_at)
               VALUES (?, ?, ?, unixepoch())""",
            (session_id, name, user_name),
        )
    logger.info("新会话: %s (name=%s)", session_id, name)
    return {"session_id": session_id}


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """获取会话历史消息"""
    history = get_history(session_id)
    return {"session_id": session_id, "history": history}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口 — SSE 格式逐 token 返回"""
    import asyncio

    msg = request.message.strip()
    session_id = request.session_id or "default"
    intent, docs = router.route(msg)  # 一次嵌入：路由 + 检索同时完成

    # 加载历史 + 用户名
    history = get_history(session_id)
    user_name = get_user_name(session_id)

    # 尝试从当前消息提取用户名
    extracted = extract_user_name(msg)
    if extracted:
        user_name = extracted
        set_user_name(session_id, user_name)

    async def event_generator() -> AsyncGenerator[str, None]:
        """SSE event generator"""
        full_answer = ""

        try:
            # ── RAG 路径 ──
            if intent == "rag":
                context = "\n\n".join(d.page_content for d in docs)
                sources = list({d.metadata.get("source", "未知") for d in docs})

                history_text = ""
                for m in history[-6:]:
                    role = "用户" if m["role"] == "user" else "小课"
                    history_text += f"{role}：{m['content']}\n"
                if history_text:
                    history_text = f"## 对话历史\n{history_text}\n"

                name_line = f"当前用户的名字是：{user_name}\n" if user_name else ""

                full_prompt = f"""{SYSTEM_PROMPT}

{name_line}{history_text}## 检索到的参考资料
{context}

## 用户问题
{msg}

## 重要规则
1. 当前用户名字：{user_name or '未知'}
2. 技术问题以参考资料为准，资料中有则引用，不编造
3. 资料未覆盖的内容坦率说明
4. 回答简洁清晰，中文为主
"""

                # 流式调用 LLM
                async for chunk in _llm.astream([HumanMessage(content=full_prompt)]):
                    token = chunk.content
                    if token:
                        full_answer += token
                        # SSE 格式：data: {json}\n\n
                        yield f"data: {json.dumps({'token': token, 'done': False}, ensure_ascii=False)}\n\n"

                # 流式结束，发送 sources 和 done 信号
                append_history(session_id, "user", msg)
                append_history(session_id, "assistant", full_answer)
                yield f"data: {json.dumps({'token': '', 'done': True, 'source': 'rag', 'docs': sources}, ensure_ascii=False)}\n\n"

            # ── 闲聊路径 ──
            else:
                messages = build_messages(history, msg, user_name)

                async for chunk in _llm.astream(messages):
                    token = chunk.content
                    if token:
                        full_answer += token
                        yield f"data: {json.dumps({'token': token, 'done': False}, ensure_ascii=False)}\n\n"

                append_history(session_id, "user", msg)
                append_history(session_id, "assistant", full_answer)
                yield f"data: {json.dumps({'token': '', 'done': True, 'source': 'casual', 'docs': []}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.exception("流式聊天异常, session=%s", session_id)
            yield f"data: {json.dumps({'token': '⚠️ 服务暂时不可用，请稍后重试。', 'done': True, 'source': 'error'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """非流式聊天接口（保留，兼容旧版）"""
    msg = request.message.strip()
    session_id = request.session_id or "default"
    intent, docs = router.route(msg)  # 一次嵌入：路由 + 检索同时完成
    # 加载历史 + 用户名
    history = get_history(session_id)
    user_name = get_user_name(session_id)

    # 尝试从当前消息提取用户名
    extracted = extract_user_name(msg)
    if extracted:
        user_name = extracted
        set_user_name(session_id, user_name)

    # ── RAG 路径 ──
    if intent == "rag":
        try:
            context = "\n\n".join(d.page_content for d in docs)
            sources = list(
                {d.metadata.get("source", "未知") for d in docs}
            )

            # 拼接对话历史
            history_text = ""
            for m in history[-6:]:
                role = "用户" if m["role"] == "user" else "小课"
                history_text += f"{role}：{m['content']}\n"
            if history_text:
                history_text = f"## 对话历史\n{history_text}\n"

            name_line = f"当前用户的名字是：{user_name}\n" if user_name else ""

            full_prompt = f"""{SYSTEM_PROMPT}

{name_line}{history_text}## 检索到的参考资料
{context}

## 用户问题
{msg}

## 重要规则
1. 当前用户名字：{user_name or '未知'}
2. 技术问题以参考资料为准，资料中有则引用，不编造
3. 资料未覆盖的内容坦率说明
4. 回答简洁清晰，中文为主
"""

            resp = await _llm_invoke([HumanMessage(content=full_prompt)])
            answer = resp.content

            append_history(session_id, "user", msg)
            append_history(session_id, "assistant", answer)

            return ChatResponse(answer=answer, source="rag", docs=sources)

        except Exception as e:
            logger.exception("RAG 链路异常, session=%s", session_id)
            return ChatResponse(
                answer=f"抱歉，知识库检索暂时不可用，请稍后重试。",
                source="fallback",
            )

    # ── 闲聊路径 ──
    else:
        try:
            messages = build_messages(history, msg, user_name)
            resp = await _llm_invoke(messages)
            answer = resp.content

            append_history(session_id, "user", msg)
            append_history(session_id, "assistant", answer)

            return ChatResponse(answer=answer, source="casual")

        except Exception as e:
            logger.exception("闲聊链路异常, session=%s", session_id)
            return ChatResponse(
                answer=f"抱歉，AI 服务暂时不可用，请稍后重试。",
                source="fallback",
            )


#===== 9. 文件上传 =====
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".docx"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 和 DOCX 格式")

    # 检查文件大小（限制 50MB）
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小不能超过 50MB")

    temp_path = os.path.join("/tmp", file.filename)
    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        if ext == ".pdf":
            text = parse_pdf(temp_path)
        else:
            text = parse_docx(temp_path)
    except Exception as e:
        logger.exception("文件解析失败: %s", file.filename)
        raise HTTPException(status_code=500, detail=f"文件解析失败")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空或无法解析")

    chunks = split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="分块结果为空")

    metas = [{"source": file.filename, "chunk": i} for i in range(len(chunks))]
    vectorstore.add_texts(texts=chunks, metadatas=metas)
    logger.info("已写入 %d 个片段，来源: %s", len(chunks), file.filename)

    return UploadResponse(
        status="success",
        filename=file.filename,
        chunks=len(chunks),
    )


# ========== 10. 启动入口 ==========
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
