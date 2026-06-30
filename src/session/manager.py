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
            conn.execute("""CREATE TABLE IF NOT EXISTS session_meta (
                session_id TEXT PRIMARY KEY, user_name TEXT)""")
            # 新：添加 user_id 列（支持迁移）
            conn.execute("""CREATE TABLE IF NOT EXISTS session_registry (
                session_id TEXT PRIMARY KEY, name TEXT NOT NULL, user_name TEXT DEFAULT '',
                created_at REAL DEFAULT (unixepoch()), user_id INTEGER)""")
            # 迁移：如果表已存在但没有 user_id 列，添加它
            cursor = conn.execute("PRAGMA table_info(session_registry)")
            columns = [row[1] for row in cursor.fetchall()]
            if "user_id" not in columns:
                conn.execute("ALTER TABLE session_registry ADD COLUMN user_id INTEGER")
                logger.info("已迁移 session_registry 表：添加 user_id 列")
        logger.info("会话数据库就绪: %s", self.db_path)

    def get_history(self, session_id: str) -> list:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT role, content FROM sessions WHERE session_id=? ORDER BY seq", (session_id,)).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    def append_history(self, session_id: str, role: str, content: str, user_id: int = None):
        self._ensure_registered(session_id, user_id)
        with sqlite3.connect(self.db_path) as conn:
            seq = conn.execute("SELECT COALESCE(MAX(seq),-1)+1 FROM sessions WHERE session_id=?", (session_id,)).fetchone()[0]
            conn.execute("INSERT INTO sessions (session_id,role,content,seq) VALUES (?,?,?,?)", (session_id, role, content, seq))
            conn.execute("DELETE FROM sessions WHERE session_id=? AND seq NOT IN (SELECT seq FROM sessions WHERE session_id=? ORDER BY seq DESC LIMIT ?)", (session_id, session_id, MAX_HISTORY))

    def _ensure_registered(self, session_id: str, user_id: int = None):
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT user_id FROM session_registry WHERE session_id=?", (session_id,)).fetchone()
            if not row:
                # 会话不存在，注册它
                conn.execute("INSERT INTO session_registry (session_id,name,user_name,created_at,user_id) VALUES (?,?,'',unixepoch(),?)", (session_id, f"会话{session_id[:6]}", user_id))
            elif row[0] is None and user_id is not None:
                # 会话存在但没有 user_id，更新它
                conn.execute("UPDATE session_registry SET user_id=? WHERE session_id=?", (user_id, session_id))

    def register_session(self, session_id="", name="", user_name="", user_id=None):
        import uuid; sid = session_id or str(uuid.uuid4())
        display = name or f"会话{sid[:6]}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO session_registry (session_id,name,user_name,created_at,user_id) VALUES (?,?,?,unixepoch(),?)", (sid, display, user_name, user_id))
        return sid

    def list_sessions(self, user_name="", user_id=None) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            if user_id is not None:
                # 只返回属于该用户的会话（不包含 user_id IS NULL 的孤儿会话）
                rows = conn.execute(
                    "SELECT session_id,name,user_name,created_at FROM session_registry WHERE user_id=? ORDER BY created_at DESC",
                    (user_id,)).fetchall()
            elif user_name:
                rows = conn.execute(
                    "SELECT session_id,name,user_name,created_at FROM session_registry WHERE user_name=? AND user_id IS NOT NULL ORDER BY created_at DESC",
                    (user_name,)).fetchall()
            else:
                # 管理员查询：只返回有关联的会话
                rows = conn.execute(
                    "SELECT session_id,name,user_name,created_at FROM session_registry WHERE user_id IS NOT NULL ORDER BY created_at DESC").fetchall()
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

    def get_or_create_default_session(self, user_id: int, user_name: str = "") -> str:
        """获取或创建用户的默认会话"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT session_id FROM session_registry WHERE user_id=? ORDER BY created_at ASC LIMIT 1",
                (user_id,)
            ).fetchone()
        if row:
            return row[0]
        return self.register_session(name="默认会话", user_name=user_name, user_id=user_id)

    def get_session_owner(self, session_id: str) -> Optional[int]:
        """返回会话的 user_id，无记录返回 None"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT user_id FROM session_registry WHERE session_id=?", (session_id,)
            ).fetchone()
        return row[0] if row else None

    def claim_session(self, session_id: str, user_id: int):
        """将已有会话归属到用户（UPDATE user_id）"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE session_registry SET user_id=? WHERE session_id=?",
                (user_id, session_id),
            )

    def _delete_session(self, session_id: str):
        """删除会话及其所有消息（硬删除）"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM session_meta WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM session_registry WHERE session_id=?", (session_id,))
        logger.info("已删除会话: %s", session_id)

    def ensure_session_owner(self, session_id: str, user_id: int) -> bool:
        """检查会话是否属于该用户（None 表示孤儿会话，拒绝访问）"""
        owner = self.get_session_owner(session_id)
        if owner is None:
            return False  # 孤儿会话不允许访问
        return owner == user_id

    @staticmethod
    def build_messages(history: list, current_msg: str, system_prompt: str, user_name="", memory_summary=""):
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        parts = [system_prompt]
        if memory_summary:
            parts.append(f"\n## 上下文记忆（更早对话的摘要）\n{memory_summary}")
        if user_name:
            parts.append(f"\n当前用户的名字是：{user_name}")
        st = "".join(parts)
        msgs = [SystemMessage(content=st)]
        for m in history:
            msgs.append(HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]))
        msgs.append(HumanMessage(content=current_msg))
        return msgs
