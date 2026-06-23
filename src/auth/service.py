"""用户服务 — 注册/登录/JWT/限流"""
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jose.jwt

from src.config import DB_PATH, JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRATION_HOURS

logger = logging.getLogger("course_assistant.auth")

# 登录限流：用户名 → 失败时间戳列表
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_MAX_ATTEMPTS = 5
_WINDOW_SECONDS = 15 * 60  # 15 分钟


class UserService:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ── DB 初始化 ──
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    username   TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now','localtime'))
                )
            """)
            # 给 session_registry 加 user_id 列（幂等）
            try:
                conn.execute("ALTER TABLE session_registry ADD COLUMN user_id INTEGER")
                logger.info("已迁移: session_registry 增加 user_id 列")
            except sqlite3.OperationalError:
                pass  # 列已存在

    # ── 密码哈希 ──
    @staticmethod
    def hash_password(plain: str) -> str:
        return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

    # ── 登录限流 ──
    @staticmethod
    def _clean_attempts(username: str):
        now = datetime.now().timestamp()
        _LOGIN_ATTEMPTS[username] = [
            t for t in _LOGIN_ATTEMPTS.get(username, [])
            if now - t < _WINDOW_SECONDS
        ]

    @classmethod
    def is_rate_limited(cls, username: str) -> bool:
        cls._clean_attempts(username)
        return len(_LOGIN_ATTEMPTS.get(username, [])) >= _MAX_ATTEMPTS

    @classmethod
    def record_login_failure(cls, username: str):
        cls._clean_attempts(username)
        if username not in _LOGIN_ATTEMPTS:
            _LOGIN_ATTEMPTS[username] = []
        _LOGIN_ATTEMPTS[username].append(datetime.now().timestamp())

    # ── 注册 ──
    def register(self, username: str, password: str) -> dict:
        username = username.strip().lower()
        pw_hash = self.hash_password(password)
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, pw_hash),
                )
            except sqlite3.IntegrityError:
                raise ValueError("用户名已存在")
            row = conn.execute(
                "SELECT username, created_at FROM users WHERE username=?",
                (username,),
            ).fetchone()
        return {"username": row[0], "created_at": row[1]}

    # ── 登录 ──
    def authenticate(self, username: str, password: str) -> Optional[dict]:
        username = username.strip().lower()
        if self.is_rate_limited(username):
            logger.warning("登录限流触发: %s", username)
            return None  # 与密码错误返回一致，不泄露限流状态
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, username, password_hash FROM users WHERE username=?",
                (username,),
            ).fetchone()
        if row and self.verify_password(password, row[2]):
            self._clear_attempts(username)  # 登录成功，清除失败记录
            return {"user_id": row[0], "username": row[1]}
        self.record_login_failure(username)
        return None

    @classmethod
    def _clear_attempts(cls, username: str):
        _LOGIN_ATTEMPTS.pop(username, None)

    # ── 获取用户信息 ──
    def get_user(self, user_id: int) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT username, created_at FROM users WHERE id=?", (user_id,)
            ).fetchone()
        return {"username": row[0], "created_at": row[1]} if row else None

    # ── JWT ──
    @staticmethod
    def create_token(user_id: int, username: str) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user_id),
            "username": username,
            "iat": now,
            "exp": now + timedelta(hours=JWT_EXPIRATION_HOURS),
        }
        return jose.jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> Optional[dict]:
        try:
            payload = jose.jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return {"user_id": int(payload["sub"]), "username": payload["username"]}
        except (jose.jwt.JWTError, KeyError, ValueError):
            return None
