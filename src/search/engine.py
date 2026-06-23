"""搜索引擎 — Bing 中国版 + 天气 + 股票"""
import asyncio, logging, re
from dataclasses import dataclass
from typing import List
from datetime import datetime

from src.config import SEARCH_MAX_RESULTS, BING_API_KEY, BOCHA_API_KEY

logger = logging.getLogger("course_assistant.search")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "bing_china"


# ═══════════════════════════════════════════════════════════════
# Bing 中国版 — 免费、国内直连
# ═══════════════════════════════════════════════════════════════

QUERY_REWRITE_PROMPT = """将用户消息改写成搜索引擎查询词。今天是{date}。

规则：
- 提取核心搜索意图，去掉口语化表达和礼貌用语
- 赛事类（世界杯/欧冠/NBA/比赛等）→ 加上"赛程"或"对阵"和具体日期
- 时效性信息（新闻/天气/股价）→ 加上今天日期
- 模糊的追问（如"小组赛"）→ 结合上下文补全为完整查询
- 保留关键名词和专业术语
- 只输出查询词，不要解释

用户消息："{query}"
搜索查询："""


class BingChinaEngine:
    def __init__(self, llm=None):
        self.llm = llm  # DeepSeek 用于查询改写

    async def _rewrite_query(self, query: str) -> str:
        """LLM 改写查询词"""
        if self.llm is None:
            return query
        try:
            import asyncio
            from datetime import datetime
            today = datetime.now().strftime("%Y年%m月%d日")
            prompt = QUERY_REWRITE_PROMPT.format(date=today, query=query)
            resp = await asyncio.to_thread(
                self.llm.invoke, prompt, {"max_tokens": 60}
            )
            rewritten = resp.content.strip().strip('"').strip("'")
            return rewritten if rewritten else query
        except Exception as e:
            logger.debug("查询改写失败: %s", e)
            return query

    async def search(self, query, max_results=5):
        query = await self._rewrite_query(query)
        logger.info("搜索: %s", query[:80])
        try:
            from bs4 import BeautifulSoup
            import httpx
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as cl:
                resp = await cl.get("https://cn.bing.com/search", params={"q": query}, headers=headers)
                resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for item in soup.select("li.b_algo")[:max_results]:
                title_el = item.select_one("h2 a")
                snippet_el = item.select_one(".b_caption p")
                if title_el:
                    results.append(SearchResult(
                        title=title_el.get_text(strip=True),
                        url=title_el.get("href", ""),
                        snippet=snippet_el.get_text(strip=True) if snippet_el else "",
                    ))
            return results
        except Exception as e:
            logger.warning("Bing 搜索失败: %s", e)
            return []


class BingSearchEngine:
    """Bing Web Search API — 需 BING_API_KEY"""
    def __init__(self, api_key=""):
        self.api_key = api_key

    async def search(self, query, max_results=5):
        if not self.api_key:
            return []
        try:
            import httpx
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            async with httpx.AsyncClient(timeout=10) as cl:
                resp = await cl.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    headers=headers,
                    params={"q": query, "count": max_results, "mkt": "zh-CN"},
                )
                resp.raise_for_status()
                pages = resp.json().get("webPages", {}).get("value", [])
                return [SearchResult(
                    title=r.get("name", ""), url=r.get("url", ""),
                    snippet=r.get("snippet", ""), source="bing_api",
                ) for r in pages]
        except Exception as e:
            logger.warning("Bing API 失败: %s", e)
            return []


# ═══════════════════════════════════════════════════════════════
# 天气 — wttr.in（免费、无需 Key）
# ═══════════════════════════════════════════════════════════════

class WeatherEngine:
    CITIES = [
        "北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "武汉",
        "南京", "西安", "天津", "苏州", "长沙", "郑州", "东莞", "青岛",
        "沈阳", "宁波", "昆明", "大连", "厦门", "合肥", "佛山", "福州",
        "哈尔滨", "济南", "温州", "长春", "石家庄", "常州", "泉州",
        "南宁", "贵阳", "南昌", "太原", "烟台", "珠海", "惠州", "徐州",
        "海口", "乌鲁木齐", "兰州", "呼和浩特", "三亚", "桂林",
    ]

    def extract_city(self, query: str) -> str:
        for city in self.CITIES:
            if city in query:
                return city
        return None

    async def search(self, query, max_results=1):
        city = self.extract_city(query)
        if not city:
            return None
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8) as cl:
                resp = await cl.get(
                    f"https://wttr.in/{city}?format=j1&lang=zh",
                    headers={"User-Agent": "curl/7.0"},
                )
                resp.raise_for_status()
                data = resp.json()

            current = data.get("current_condition", [{}])[0]
            weather = data.get("weather", [])
            temp = current.get("temp_C", "?")
            desc = (current.get("weatherDesc", [{}])[0].get("value", "?"))
            humidity = current.get("humidity", "?")
            wind = f"{current.get('winddir16Point','?')} {current.get('windspeedKmph','?')}km/h"

            snippet = f"{city}: {desc} {temp}°C 湿度{humidity}% {wind}"
            if weather:
                hi = weather[0].get("maxtempC", "?")
                lo = weather[0].get("mintempC", "?")
                snippet += f" | 今天 {lo}°C ~ {hi}°C"

            return [SearchResult(
                title=f"{city}天气", url=f"https://wttr.in/{city}",
                snippet=snippet, source="wttr",
            )]
        except Exception as e:
            logger.warning("天气查询失败: %s", e)
            return None


# ═══════════════════════════════════════════════════════════════
# 股票 — 新浪财经 / Yahoo Finance
# ═══════════════════════════════════════════════════════════════

class StockEngine:
    STOCKS = {
        "茅台": "sh600519", "贵州茅台": "sh600519",
        "平安银行": "sz000001", "腾讯": "hk00700", "腾讯控股": "hk00700",
        "阿里": "hk09988", "阿里巴巴": "hk09988",
        "百度": "hk09888", "京东": "hk09618", "美团": "hk03690",
        "比亚迪": "sz002594", "宁德时代": "sz300750",
        "招商银行": "sh600036", "工商银行": "sh601398",
        "特斯拉": "tsla", "苹果": "aapl", "英伟达": "nvda",
    }

    def extract_stock(self, query: str):
        for name, code in self.STOCKS.items():
            if name in query:
                return name, code
        return None, None

    async def search(self, query, max_results=1):
        name, code = self.extract_stock(query)
        if not name:
            return None
        is_cn = code.startswith("sh") or code.startswith("sz")
        try:
            import httpx
            if is_cn:
                headers = {"Referer": "https://finance.sina.com.cn"}
                async with httpx.AsyncClient(timeout=8) as cl:
                    resp = await cl.get(f"http://hq.sinajs.cn/list={code}", headers=headers)
                    text = resp.content.decode("gbk", errors="replace")
                    parts = text.split('"')[1].split(",")
                    cur_name, current = parts[0], parts[3]
                    change = "?"
                    try:
                        prev, cur = float(parts[2]), float(parts[3])
                        change = f"{(cur-prev)/prev*100:+.2f}%"
                    except Exception:
                        pass
                    snippet = f"{cur_name} {current} {change}"
                return [SearchResult(title=f"{cur_name} 实时股价",
                    url=f"https://finance.sina.com.cn/realstock/company/{code}/nc.shtml",
                    snippet=snippet, source="sina")]
            else:
                async with httpx.AsyncClient(timeout=8) as cl:
                    resp = await cl.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{code}")
                    data = resp.json()
                    m = data["chart"]["result"][0]["meta"]
                    price, prev = m["regularMarketPrice"], m.get("previousClose", price)
                    pct = (price - prev) / prev * 100 if prev else 0
                    snippet = f"{name} {m.get('currency','USD')} {price} {pct:+.2f}%"
                return [SearchResult(title=f"{name} 实时股价",
                    url=f"https://finance.yahoo.com/quote/{code}",
                    snippet=snippet, source="yahoo")]
        except Exception as e:
            logger.warning("股票查询失败 (%s): %s", code, e)
            return None


# ═══════════════════════════════════════════════════════════════
# 统一搜索引擎
# ═══════════════════════════════════════════════════════════════

class BochaEngine:
    """博查 AI Search API — 国内最强 AI 搜索引擎"""

    async def search(self, query, max_results=5):
        if not BOCHA_API_KEY:
            return None
        try:
            import httpx, json as _json
            headers = {
                "Authorization": f"Bearer {BOCHA_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {"query": query, "count": max_results, "stream": False}
            async with httpx.AsyncClient(timeout=15) as cl:
                resp = await cl.post(
                    "https://api.bocha.cn/v1/ai-search",
                    headers=headers, json=body,
                )
                resp.raise_for_status()
                data = resp.json()

            results = []
            answer = ""

            for msg in data.get("messages", []):
                ct = msg.get("content_type", "")
                content = msg.get("content", "")

                # 网页搜索结果 (JSON 字符串)
                if ct == "webpage" and isinstance(content, str):
                    try:
                        for page in _json.loads(content).get("value", []):
                            results.append(SearchResult(
                                title=page.get("name", ""),
                                url=page.get("url", ""),
                                snippet=page.get("summary", "") or page.get("snippet", ""),
                                source="bocha",
                            ))
                    except Exception:
                        pass

                # 结构化数据（天气/股票等，直接是 list）
                elif ct in ("weather_china", "stock", "exchange_rate"):
                    items = content if isinstance(content, list) else []
                    for item in items[:3]:
                        results.insert(0, SearchResult(
                            title=item.get("name", ct),
                            url=item.get("url", ""),
                            snippet=str(item.get("snippet", item))[:300],
                            source=f"bocha_{ct}",
                        ))

                # AI 回答
                elif msg.get("type") == "answer" and isinstance(content, str):
                    answer = content

            logger.info("博查: %d 结果 + %d 字回答", len(results), len(answer))
            return results, answer

        except Exception as e:
            logger.warning("博查失败: %s", e)
            return None

    async def search_stream(self, query, max_results=5):
        """流式搜索 — 边搜边返回 token"""
        if not BOCHA_API_KEY:
            return

        import httpx, json as _json
        headers = {
            "Authorization": f"Bearer {BOCHA_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {"query": query, "count": max_results, "stream": True}
        try:
            async with httpx.AsyncClient(timeout=30) as cl:
                async with cl.stream(
                    "POST", "https://api.bocha.cn/v1/ai-search",
                    headers=headers, json=body,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if not payload.strip() or payload.strip() == "[DONE]":
                            continue
                        try:
                            event = _json.loads(payload)
                            for msg in event.get("messages", []):
                                if msg.get("type") == "answer":
                                    content = msg.get("content", "")
                                    if isinstance(content, str):
                                        yield content
                        except Exception:
                            continue
        except Exception as e:
            logger.warning("博查流式失败: %s", e)
            return


class WebSearchEngine:
    def __init__(self, bing_api_key=None, llm=None):
        self.bocha = BochaEngine()
        self.bing = BingChinaEngine(llm)
        self.bing_api = BingSearchEngine(bing_api_key or BING_API_KEY)

    async def search(self, query, max_results=None):
        """返回搜索结果列表"""
        if max_results is None:
            max_results = SEARCH_MAX_RESULTS
        if BOCHA_API_KEY:
            r, _ = await self.bocha.search(query, max_results)
            if r:
                return r
        r = await self.bing.search(query, max_results)
        if r:
            return r
        return await self.bing_api.search(query, max_results)

    async def search_with_answer(self, query, max_results=None):
        """博查专用：返回搜索结果 + AI 回答"""
        if max_results is None:
            max_results = SEARCH_MAX_RESULTS
        if BOCHA_API_KEY:
            return await self.bocha.search(query, max_results)
        r = await self.bing.search(query, max_results)
        return r, ""
