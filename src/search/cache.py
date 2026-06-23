"""搜索缓存 + 限流"""
import time, hashlib, logging
from collections import deque
from src.config import SEARCH_CACHE_TTL, SEARCH_RATE_LIMIT
logger = logging.getLogger("course_assistant.search.cache")

class SearchCache:
    def __init__(self, ttl=SEARCH_CACHE_TTL, max_size=200):
        from collections import OrderedDict
        self._cache = OrderedDict(); self.ttl = ttl; self.max_size = max_size
    @staticmethod
    def _key(q): return hashlib.md5(q.lower().strip().encode()).hexdigest()

    def get(self, query):
        k = self._key(query)
        if k in self._cache:
            ts, results = self._cache[k]
            if time.time() - ts < self.ttl:
                self._cache.move_to_end(k)
                return results
            del self._cache[k]
        return None

    def put(self, query, results):
        k = self._key(query)
        if k in self._cache: self._cache.move_to_end(k)
        else:
            self._cache[k] = (time.time(), results)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def invalidate(self, query=None):
        if query: self._cache.pop(self._key(query), None)
        else: self._cache.clear()

    @property
    def size(self): return len(self._cache)

class RateLimiter:
    def __init__(self, max_calls=SEARCH_RATE_LIMIT, period=60):
        self.max_calls = max_calls; self.period = period; self._calls = deque()

    async def acquire(self):
        import asyncio
        now = time.time()
        while self._calls and self._calls[0] < now - self.period: self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            wait = self._calls[0] + self.period - now
            if wait > 30: raise RuntimeError(f"限流: 需等待 {wait:.0f}s")
            await asyncio.sleep(wait)
        self._calls.append(time.time())
