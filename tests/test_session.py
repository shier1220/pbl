"""会话管理测试"""
import pytest
from src.session.manager import SessionManager


class TestSessionManager:
    @pytest.fixture(autouse=True)
    def _migrate(self, tmp_db_path):
        """每个测试前确保 user_id 列存在"""
        from src.auth.service import UserService
        UserService(tmp_db_path)

    def test_init_db(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        # 第二次 init 应该幂等
        sm2 = SessionManager(tmp_db_path)
        assert sm2 is not None

    def test_register_session(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        sid = sm.register_session(name="测试", user_name="u1")
        assert len(sid) > 0

    def test_register_session_with_user_id(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        sid = sm.register_session(name="我的会话", user_name="u1", user_id=42)
        owner = sm.get_session_owner(sid)
        assert owner == 42

    def test_get_session_owner_nonexistent(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        assert sm.get_session_owner("no-such-id") is None

    def test_claim_session(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        sid = sm.register_session(name="旧会话")
        assert sm.get_session_owner(sid) is None
        sm.claim_session(sid, 10)
        assert sm.get_session_owner(sid) == 10

    def test_list_sessions_by_user(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        sm.register_session(name="A", user_id=1)
        sm.register_session(name="B", user_id=2)

        sessions = sm.list_sessions(user_id=1)
        session_names = {s["name"] for s in sessions}
        assert "A" in session_names
        assert "B" not in session_names

    def test_append_and_get_history(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        sid = sm.register_session()
        sm.append_history(sid, "user", "你好")
        sm.append_history(sid, "assistant", "你好！")
        history = sm.get_history(sid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_get_or_create_default(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        sid = sm.get_or_create_default_session(42, "u42")
        owner = sm.get_session_owner(sid)
        assert owner == 42
        # 再次调用应返回同一个 session
        sid2 = sm.get_or_create_default_session(42, "u42")
        assert sid2 == sid

    def test_extract_user_name(self):
        assert SessionManager.extract_user_name("我是张三") == "张三"
        assert SessionManager.extract_user_name("我叫李四，你好") == "李四"
        assert SessionManager.extract_user_name("你好") is None
        assert SessionManager.extract_user_name("我是一个很长很长很长很长很长名字的人") is None

    def test_build_messages(self, tmp_db_path):
        sm = SessionManager(tmp_db_path)
        msgs = sm.build_messages([], "测试消息", "系统提示", "小明")
        from langchain_core.messages import SystemMessage, HumanMessage
        assert isinstance(msgs[0], SystemMessage)
        assert "小明" in msgs[0].content
        assert isinstance(msgs[-1], HumanMessage)
        assert msgs[-1].content == "测试消息"
