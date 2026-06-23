"""Auth Pydantic 模型 — 含输入校验"""
from pydantic import BaseModel, Field, validator


class UserRegister(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=128)

    @validator("username")
    def username_format(cls, v):
        import re
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("用户名只能包含字母、数字和下划线")
        return v.strip().lower()

    @validator("password")
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError("密码至少 6 位")
        if not any(c.isalpha() for c in v):
            raise ValueError("密码至少包含 1 个字母")
        if not any(c.isdigit() for c in v):
            raise ValueError("密码至少包含 1 个数字")
        return v


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class UserInfo(BaseModel):
    username: str
    created_at: str
