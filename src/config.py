"""集中配置管理"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = str(BASE_DIR / "chroma_db")
DB_PATH = str(BASE_DIR / "sessions.db")

_LOCAL_MODEL = str(BASE_DIR / "instructor-xl")
MODEL_PATH = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else "hkunlp/instructor-xl"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
_default_device = "cpu"
try:
    import torch
    if torch.backends.mps.is_available(): _default_device = "mps"
    elif torch.cuda.is_available(): _default_device = "cuda"
except ImportError: pass
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", _default_device)

RAG_TOP_K = 5
RAG_THRESHOLD = 0.69
RAG_OVERFETCH = RAG_TOP_K * 2
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
RRF_K = 60
QUERY_CACHE_SIZE = 100

INTENT_CONFIDENCE_HIGH = 0.55
INTENT_CONFIDENCE_MEDIUM = 0.35

SEARCH_MAX_RESULTS = 5
SEARCH_CACHE_TTL = 3600
SEARCH_RATE_LIMIT = 30
BING_API_KEY = os.getenv("BING_API_KEY", "")
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# JWT_SECRET 未设置时自动生成临时密钥（重启后失效，仅开发可用）
if not JWT_SECRET:
    import secrets, warnings
    JWT_SECRET = secrets.token_hex(32)
    warnings.warn("⚠️ JWT_SECRET 未设置，已生成临时密钥（重启后失效）。请在 .env 中设置 JWT_SECRET=<随机64位十六进制字符串>")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_HISTORY = 10
MAX_BM25_DOCS = 50000  # BM25 索引入库文档数上限，超出则警告并截断

COLLECTION_NAME = "course_knowledge"
COLLECTION_METADATA = {"description": "AIGC助手知识库", "hnsw:space": "cosine"}
