"""Auth 依赖注入"""
from fastapi import Header, HTTPException

from src.auth.service import UserService


def get_current_user(authorization: str = Header(default="")):
    """从 Authorization: Bearer <token> 头提取当前用户"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")
    token = authorization[len("Bearer "):]
    payload = UserService.decode_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="令牌无效或已过期")
    return payload  # {"user_id": int, "username": str}
