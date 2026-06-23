"""搜索引擎测试"""
import pytest
from src.search.cache import SearchCache, RateLimiter
from src.search.engine import SearchResult, BingChinaEngine, WebSearchEngine


class TestWebSearchEngine:
    def test_init(self):
        engine = WebSearchEngine()
        assert engine.bing is not None
        assert engine.bocha is not None

    @pytest.mark.asyncio
    async def test_search(self):
        """Bing 搜索（需要网络）"""
        engine = WebSearchEngine()
        results = await engine.search("Python教程", max_results=3)
        assert isinstance(results, list)
        if results:
            assert results[0].title


class TestSearchCache:
    def test_put_and_get(self):
        cache = SearchCache(ttl=3600)
        cache.put("test query", ["result1", "result2"])
        assert cache.get("test query") == ["result1", "result2"]
        assert cache.get("other query") is None

    def test_ttl_expiry(self, monkeypatch):
        import time
        cache = SearchCache(ttl=1)
        cache.put("q", ["r"])
        # 模拟时间前进 2 秒
        fake_time = time.time() + 2
        monkeypatch.setattr(time, "time", lambda: fake_time)
        assert cache.get("q") is None

    def test_max_size_lru(self):
        cache = SearchCache(ttl=3600, max_size=3)
        for i in range(5):
            cache.put(f"q{i}", [f"r{i}"])
        assert cache.size == 3
        assert cache.get("q0") is None  # 最早的被淘汰
        assert cache.get("q4") == ["r4"]


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_under_limit(self):
        rl = RateLimiter(max_calls=10, period=60)
        await rl.acquire()
        await rl.acquire()
        assert len(rl._calls) == 2

    @pytest.mark.asyncio
    async def test_reject_when_window_full(self):
        import time
        rl = RateLimiter(max_calls=3, period=3600)
        rl._calls.extend([time.time()] * 3)
        with pytest.raises(RuntimeError, match="限流"):
            await rl.acquire()


class TestBingChinaEngine:
    def test_init(self):
        engine = BingChinaEngine()
        assert engine.llm is None

    @pytest.mark.asyncio
    async def test_rewrite_no_llm_fallback(self):
        """无 LLM 时返回原始查询"""
        engine = BingChinaEngine()
        result = await engine._rewrite_query("帮我搜一下Python教程")
        assert result == "帮我搜一下Python教程"

    @pytest.mark.asyncio
    async def test_search_without_llm(self):
        """无 LLM 也能正常搜索"""
        engine = BingChinaEngine()
        results = await engine.search("Python教程", max_results=3)
        assert isinstance(results, list)
        if results:
            assert results[0].title
