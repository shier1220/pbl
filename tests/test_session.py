"""会话管理测试"""
import pytest
from src.session.manager import SessionManager
from src.session.context_memory import ContextMemory


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


class TestContextMemory:
    """上下文记忆单元测试 — 无外部依赖"""

    def test_short_history_all_recent(self):
        """短历史（≤窗口）全部保留，无摘要"""
        cm = ContextMemory(recent_window=6)
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "什么是 LoRA？"},
            {"role": "assistant", "content": "LoRA 是..."},
        ]
        recent, summary = cm.build_context(history)
        assert len(recent) == 4
        assert summary == ""
        assert recent == history

    def test_long_history_splits(self):
        """长历史（>窗口）分割为近期+摘要"""
        cm = ContextMemory(recent_window=4)
        history = []
        for i in range(10):
            history.append({"role": "user", "content": f"问题{i}"})
            history.append({"role": "assistant", "content": f"回答{i}"})

        recent, summary = cm.build_context(history)
        # 20条消息，窗口4，近期=最后4条
        assert len(recent) == 4
        assert recent == history[-4:]

    def test_memory_summary_includes_user_name(self):
        """摘要包含从旧消息中提取的用户名"""
        cm = ContextMemory(recent_window=2)
        history = [
            {"role": "user", "content": "你好，我是小明"},
            {"role": "assistant", "content": "你好小明！"},
            {"role": "user", "content": "什么是 RAG？"},
            {"role": "assistant", "content": "RAG 是检索增强生成..."},
        ]
        recent, summary = cm.build_context(history)
        assert len(recent) == 2
        assert "小明" in summary

    def test_memory_summary_includes_topics(self):
        """摘要包含旧的讨论主题（AIGC 技术关键词）"""
        cm = ContextMemory(recent_window=2)
        history = [
            {"role": "user", "content": "LoRA 微调怎么做？"},
            {"role": "assistant", "content": "LoRA 是一种高效微调方法..."},
            {"role": "user", "content": "那 QLoRA 呢？"},
            {"role": "assistant", "content": "QLoRA 引入了量化..."},
        ]
        recent, summary = cm.build_context(history)
        assert "LoRA" in summary or "QLoRA" in summary or "微调" in summary

    def test_memory_summary_detects_pending(self):
        """摘要包含未解决的问题信号"""
        cm = ContextMemory(recent_window=2)
        history = [
            {"role": "user", "content": "帮我查一下 vLLM 部署方案"},
            {"role": "assistant", "content": "vLLM 部署..."},
            {"role": "user", "content": "好的，另外还有一个问题，GPU 选型怎么选？"},
            {"role": "assistant", "content": "GPU 选型..."},
        ]
        recent, summary = cm.build_context(history)
        assert any(kw in summary for kw in ["vLLM", "GPU", "部署"])

    def test_empty_history(self):
        """空历史不报错"""
        cm = ContextMemory()
        recent, summary = cm.build_context([])
        assert recent == []
        assert summary == ""

    def test_has_pending_questions_true(self):
        """检测到未解决问题"""
        history = [
            {"role": "user", "content": "还有一个问题想问"},
        ]
        assert ContextMemory.has_pending_questions(history) is True

    def test_has_pending_questions_false(self):
        """无未解决问题"""
        history = [
            {"role": "user", "content": "谢谢，问题解决了"},
        ]
        assert ContextMemory.has_pending_questions(history) is False

    def test_build_messages_with_memory_summary(self):
        """build_messages 正确注入 memory_summary"""
        msgs = SessionManager.build_messages(
            history=[{"role": "user", "content": "你好"}],
            current_msg="什么是 RAG？",
            system_prompt="你是助手",
            user_name="小明",
            memory_summary="用户之前讨论了 LoRA 微调",
        )
        from langchain_core.messages import SystemMessage
        sys_content = msgs[0].content
        assert "你是助手" in sys_content
        assert "LoRA 微调" in sys_content
        assert "小明" in sys_content
