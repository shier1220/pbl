"""会话管理 — SQLite 持久化"""
import re, sqlite3, logging
from typing import Optional, List, Dict
from src.config import DB_PATH, MAX_HISTORY

logger = logging.getLogger("course_assistant.session")

_NAME_PATTERNS = [re.compile(p) for p in [
    r"我是(.+?)[，。！？\s]", r"我叫(.+?)[，。！？\s]",
    r"^我是(.+)$", r"^我叫(.+)$", r"我叫(.+)", r"我是(.+)", r"称呼我(.+)",
]]

class SessionManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT, role TEXT, content TEXT, seq INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now','localtime')))""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_id ON sessions(session_id)")
            conn.execute("CREATE TABLE IF NOT EXISTS session_meta (session_id TEXT PRIMARY KEY, user_name TEXT)")
            conn.execute("""CREATE TABLE IF NOT EXISTS session_registry (
                session_id TEXT PRIMARY KEY, name TEXT NOT NULL, user_name TEXT DEFAULT '',
                created_at REAL DEFAULT (unixepoch()))""")
        logger.info("会话数据库就绪: %s", self.db_path)

    def get_history(self, session_id: str) -> list:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT role, content FROM sessions WHERE session_id=? ORDER BY seq", (session_id,)).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    def append_history(self, session_id: str, role: str, content: str):
        self._ensure_registered(session_id)
        with sqlite3.connect(self.db_path) as conn:
            seq = conn.execute("SELECT COALESCE(MAX(seq),-1)+1 FROM sessions WHERE session_id=?", (session_id,)).fetchone()[0]
            conn.execute("INSERT INTO sessions (session_id,role,content,seq) VALUES (?,?,?,?)", (session_id, role, content, seq))
            conn.execute("DELETE FROM sessions WHERE session_id=? AND seq NOT IN (SELECT seq FROM sessions WHERE session_id=? ORDER BY seq DESC LIMIT ?)", (session_id, session_id, MAX_HISTORY))

    def _ensure_registered(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            if not conn.execute("SELECT 1 FROM session_registry WHERE session_id=?", (session_id,)).fetchone():
                conn.execute("INSERT INTO session_registry (session_id,name,user_name,created_at) VALUES (?,?,'',unixepoch())", (session_id, f"会话{session_id[:6]}"))

    def register_session(self, session_id="", name="", user_name=""):
        import uuid; sid = session_id or str(uuid.uuid4())
        display = name or f"会话{sid[:6]}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO session_registry (session_id,name,user_name,created_at) VALUES (?,?,?,unixepoch())", (sid, display, user_name))
        return sid

    def list_sessions(self, user_name="") -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT session_id,name,user_name,created_at FROM session_registry ORDER BY created_at DESC").fetchall() if not user_name else \
                   conn.execute("SELECT session_id,name,user_name,created_at FROM session_registry WHERE user_name=? ORDER BY created_at DESC", (user_name,)).fetchall()
        return [{"id": r[0], "name": r[1], "user_name": r[2] or "", "created_at": r[3]} for r in rows]

    def get_session_history(self, session_id: str) -> Dict:
        return {"session_id": session_id, "history": self.get_history(session_id)}

    def get_user_name(self, session_id: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT user_name FROM session_meta WHERE session_id=?", (session_id,)).fetchone()
        return row[0] if row else ""

    def set_user_name(self, session_id: str, user_name: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO session_meta (session_id,user_name) VALUES (?,?)", (session_id, user_name))

    @staticmethod
    def extract_user_name(msg: str) -> Optional[str]:
        for pat in _NAME_PATTERNS:
            m = pat.search(msg.strip())
            if m:
                name = m.group(1).strip()
                if len(name) <= 10 and name: return name
        return None

    @staticmethod
    def build_messages(history: list, current_msg: str, system_prompt: str, user_name=""):
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        st = system_prompt + (f"\n\n当前用户的名字是：{user_name}" if user_name else "")
        msgs = [SystemMessage(content=st)]
        for m in history:
            msgs.append(HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]))
        msgs.append(HumanMessage(content=current_msg))
        return msgs
