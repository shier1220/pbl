"""认证模块测试"""
import pytest
from src.auth.service import UserService
from src.auth.models import UserRegister, UserLogin, TokenResponse


class TestUserService:
    def test_register(self, tmp_db_path):
        svc = UserService(tmp_db_path)
        info = svc.register("testuser", "abc123")
        assert info["username"] == "testuser"
        assert "created_at" in info

    def test_register_duplicate(self, tmp_db_path):
        svc = UserService(tmp_db_path)
        svc.register("testuser", "abc123")
        with pytest.raises(ValueError, match="用户名已存在"):
            svc.register("testuser", "xyz789")

    def test_authenticate_success(self, tmp_db_path):
        svc = UserService(tmp_db_path)
        svc.register("testuser", "abc123")
        user = svc.authenticate("testuser", "abc123")
        assert user is not None
        assert user["username"] == "testuser"
        assert "user_id" in user

    def test_authenticate_wrong_password(self, tmp_db_path):
        svc = UserService(tmp_db_path)
        svc.register("testuser", "abc123")
        assert svc.authenticate("testuser", "wrong") is None

    def test_authenticate_unknown_user(self, tmp_db_path):
        svc = UserService(tmp_db_path)
        assert svc.authenticate("nobody", "xxx") is None

    def test_password_hashing(self):
        h = UserService.hash_password("mypassword")
        assert h.startswith("$2b$")
        assert UserService.verify_password("mypassword", h)
        assert not UserService.verify_password("wrong", h)

    def test_jwt_token(self):
        token = UserService.create_token(42, "user42")
        assert token.count(".") == 2  # JWT 三段式
        payload = UserService.decode_token(token)
        assert payload is not None
        assert payload["user_id"] == 42
        assert payload["username"] == "user42"

    def test_jwt_decode_invalid(self):
        assert UserService.decode_token("bad.token.here") is None
        assert UserService.decode_token("") is None

    def test_get_user(self, tmp_db_path):
        svc = UserService(tmp_db_path)
        info = svc.register("alice", "pass123")
        user = svc.authenticate("alice", "pass123")
        details = svc.get_user(user["user_id"])
        assert details["username"] == "alice"

    def test_get_user_nonexistent(self, tmp_db_path):
        svc = UserService(tmp_db_path)
        assert svc.get_user(99999) is None

    def test_case_insensitive_username(self, tmp_db_path):
        svc = UserService(tmp_db_path)
        svc.register("TestUser", "abc123")
        user = svc.authenticate("testuser", "abc123")
        assert user is not None


class TestAuthModels:
    def test_register_model_valid(self):
        m = UserRegister(username="abc", password="abc123")
        assert m.username == "abc"

    def test_register_model_short_username(self):
        with pytest.raises(Exception):
            UserRegister(username="ab", password="123456")

    def test_login_model(self):
        m = UserLogin(username="abc", password="123")
        assert m.username == "abc"

    def test_token_response(self):
        t = TokenResponse(access_token="eyJ...", username="u1")
        assert t.token_type == "bearer"
