"""分享模块 — 对话消息分享链接 + 公开只读页面"""

import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from fastapi.responses import HTMLResponse

SHARES_DIR = Path(__file__).resolve().parent.parent.parent / "shares"
SHARES_DIR.mkdir(exist_ok=True)

SHARE_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — AIGC 导师小课</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: #f8fafc; min-height: 100vh;
  }}
  .topbar {{
    background: linear-gradient(90deg, #5eead4 0%, #3b82f6 100%);
    color: #0f172a; padding: 14px 24px; text-align: center;
    font-size: 18px; font-weight: 700; letter-spacing: 0.5px;
  }}
  .topbar span {{ opacity: 0.7; font-size: 12px; font-weight: 400; margin-left: 8px; }}
  .container {{ max-width: 720px; margin: 0 auto; padding: 20px 16px 40px; }}
  .meta {{ text-align: center; padding: 16px 0 8px; font-size: 13px; color: #94a3b8; }}
  .msg {{ margin-bottom: 14px; }}
  .msg-user {{ text-align: right; }}
  .msg-user .bubble {{
    background: #3b82f6; color: #fff;
    border-radius: 18px 4px 18px 18px;
  }}
  .msg-assistant .bubble {{
    background: #fff; color: #1e293b;
    border-radius: 4px 18px 18px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .bubble {{
    display: inline-block; max-width: 85%; padding: 12px 16px;
    font-size: 14.5px; line-height: 1.65; text-align: left;
    white-space: pre-wrap; word-break: break-word;
  }}
  .role {{ font-size: 11.5px; color: #94a3b8; margin-bottom: 2px; }}
  .msg-user .role {{ margin-right: 4px; }}
  .msg-assistant .role {{ margin-left: 4px; }}
  .footer {{
    text-align: center; padding: 28px 0 0; font-size: 12.5px; color: #94a3b8;
    border-top: 1px solid #e2e8f0; margin-top: 24px;
  }}
  .footer span {{ color: #5eead4; }}
</style>
</head>
<body>
<div class="topbar">🤖 AIGC 导师小课<span>· 分享的对话</span></div>
<div class="container">
  <div class="meta">{title} · 分享于 {date}</div>
  {messages_html}
  <div class="footer">
    <span>◆</span> AIGC 课程助手生成
  </div>
</div>
</body>
</html>"""


def save_share(messages: list, title: str = "对话分享") -> str:
    """保存分享消息，返回 share_id"""
    share_id = uuid.uuid4().hex[:12]
    data = {
        "share_id": share_id,
        "title": title,
        "created_at": datetime.now().isoformat(),
        "messages": messages,
    }
    filepath = SHARES_DIR / f"{share_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return share_id


def get_share(share_id: str) -> dict | None:
    """获取分享数据"""
    filepath = SHARES_DIR / f"{share_id}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def render_share_page(share_id: str) -> str:
    """渲染公开分享页面 HTML"""
    data = get_share(share_id)
    if not data:
        return "<h1>分享不存在或已过期</h1>"

    messages_html = ""
    for m in data["messages"]:
        role = m.get("role", "user")
        content = m.get("content", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        role_label = "用户" if role == "user" else "小课"
        msg_class = "msg-user" if role == "user" else "msg-assistant"
        messages_html += f"""  <div class="msg {msg_class}">
    <div class="role">{role_label}</div>
    <div class="bubble">{content}</div>
  </div>
"""

    return SHARE_PAGE_HTML.format(
        title=data.get("title", "对话分享"),
        date=data.get("created_at", "")[:16].replace("T", " "),
        messages_html=messages_html,
    )
