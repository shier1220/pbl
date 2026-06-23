"""LRU 查询缓存"""
import hashlib, logging
from collections import OrderedDict
from src.config import QUERY_CACHE_SIZE

logger = logging.getLogger("course_assistant.cache")

class QueryCache:
    def __init__(self, max_size=QUERY_CACHE_SIZE):
        self._cache = OrderedDict(); self.max_size = max_size
        self._hits = self._misses = 0

    @staticmethod
    def _key(q): return hashlib.md5(q.lower().strip().encode()).hexdigest()

    def get(self, query):
        k = self._key(query)
        if k in self._cache:
            self._cache.move_to_end(k); self._hits += 1; return self._cache[k]
        self._misses += 1; return None

    def put(self, query, value):
        k = self._key(query)
        if k in self._cache:
            self._cache[k] = value  # 更新值
            self._cache.move_to_end(k)
        else:
            self._cache[k] = value
            if len(self._cache) > self.max_size: self._cache.popitem(last=False)

    def invalidate(self, query=None):
        if query: self._cache.pop(self._key(query), None)
        else: self._cache.clear(); self._hits = self._misses = 0

    @property
    def stats(self):
        total = self._hits + self._misses
        return {"size": len(self._cache), "max_size": self.max_size, "hits": self._hits, "misses": self._misses, "hit_rate": f"{self._hits/total:.1%}" if total else "0%"}
