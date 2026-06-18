"""
AIGC助手 API 后端 — 兼容包装器
所有实现已迁移至 src/ 包，此文件保留向后兼容

用法不变: python course_assistant_api.py
"""
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

from src.main import app

if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)
