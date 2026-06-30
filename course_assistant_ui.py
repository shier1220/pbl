# ========== 第八课：Gradio 前端对接 FastAPI 后端 ==========
# UI 和 step7.py 完全一样，但聊天逻辑改为调 FastAPI（RAG + 意图路由）
# - /chat/stream → SSE 流式对话（含 RAG 检索）
# - /upload → 文件上传入库 ChromaDB
#
# 前置条件：先启动后端 → cd /Users/dinghao/Desktop/pbl && source pbl_venv/bin/activate && python course_assistant_api.py
# 运行前端：source pbl_venv/bin/activate && python step8.py

import os
import json
import uuid
import httpx
import gradio as gr
from dotenv import load_dotenv
from datetime import datetime

# ========== 1. 配置 ==========
load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")

# ========== 2. 会话持久化（按用户隔离）==========

# 会话存储目录：.sessions/{username}.json
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

# 当前用户的内存会话
all_sessions = []
current_id = None
current_username = ""  # 当前登录的用户名


def get_user_sessions_file(username):
    """获取用户的会话文件路径"""
    return os.path.join(SESSIONS_DIR, f"{username}.json")


def load_user_sessions(username):
    """加载指定用户的会话"""
    global all_sessions, current_id, current_username
    current_username = username
    user_file = get_user_sessions_file(username)
    
    if os.path.exists(user_file):
        try:
            with open(user_file, "r", encoding="utf-8") as f:
                all_sessions = json.load(f)
        except Exception:
            all_sessions = []
    else:
        all_sessions = []
    
    # 如果该用户没有会话，创建一个
    if not all_sessions:
        first = make_session()
        all_sessions.append(first)
        save_user_sessions(username)
    
    current_id = all_sessions[-1]["id"]


def save_user_sessions(username):
    """保存指定用户的会话"""
    user_file = get_user_sessions_file(username)
    with open(user_file, "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, ensure_ascii=False, indent=2)


def make_session(title="新对话"):
    return {
        "id": str(uuid.uuid4()),  # 生成完整 UUID，避免碰撞
        "title": title,
        "created_at": datetime.now().strftime("%m-%d %H:%M"),
        "messages": [],
        "file_context": None,
    }


def get_current():
    for s in all_sessions:
        if s["id"] == current_id:
            return s
    return all_sessions[-1] if all_sessions else make_session()


def _truncate_title(title, max_len=12):
    """统一截断标题，保证列表整齐"""
    if len(title) > max_len:
        return title[:max_len] + "…"
    return title


def get_session_choices():
    """生成统一宽度的会话列表选项"""
    choices = []
    for s in reversed(all_sessions):
        title = _truncate_title(s["title"])
        label = f"💬 {title}  ({s['created_at']})"
        choices.append((label, s["id"]))
    return choices


# ========== 3. CSS（从外部文件加载）==========

def _load_css():
    """加载外部 CSS 文件"""
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "style.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[WARN] CSS 文件未找到: {css_path}，使用内联回退样式")
        return _fallback_css()


def _fallback_css():
    """回退 CSS（当外部文件不可用时）"""
    return """
    html, body { margin: 0 !important; padding: 0 !important; }
    .gradio-container, .block-container { margin: 0 !important; padding: 0 !important; max-width: 100% !important; }
    footer, .footer { display: none !important; }
    .top-bar { background: #f1f5f9 !important; color: #0f172a !important; padding: 8px 16px; display: flex; justify-content: space-between; border-bottom: 1px solid #e2e8f0; }
    .top-bar, .top-bar p, .top-bar span, .top-bar div, .top-bar .prose, .top-bar .prose p { color: #0f172a !important; }
    .top-bar-title { margin-left: auto !important; flex-shrink: 0 !important; }
    .sidebar { background: #f8fafc !important; padding: 0 16px !important; min-height: calc(100vh - 64px) !important; }
    .sidebar-new { flex: 0 0 auto !important; min-height: 32px !important; padding: 0 12px !important; }
    .new-btn button { background: linear-gradient(90deg, #5eead4 0%, #3b82f6 100%) !important; color: #0f172a !important; border: none !important; border-radius: 6px !important; font-weight: 700 !important; }
    .sidebar-header { flex: 0 0 auto !important; min-height: 40px !important; }
    .sidebar-header .prose, .sidebar-header h3 { font-size: 20px !important; color: #475569 !important; padding-left: 12px !important; }
    .history-divider { width: 100% !important; height: 2px !important; background: #5eead4 !important; display: block !important; }
    .sidebar-list { flex: 0 0 70% !important; overflow-y: auto !important; padding: 4px 0 !important; }
    .session-radio { width: 100% !important; }
    .exit-btn button { background: transparent !important; color: #0f172a !important; border: 1px solid rgba(15,23,42,0.2) !important; border-radius: 6px !important; width: 60px !important; height: 32px !important; font-size: 13px !important; }
    .chat-area { background: white; padding: 0; min-height: calc(100vh - 64px); box-sizing: border-box; display: flex !important; flex-direction: column !important; }
    .main-row { min-height: calc(100vh - 64px); align-items: stretch; gap: 0 !important; }
    .input-row { background: white; border: 1px solid #d1d5db; border-radius: 28px; padding: 4px 8px; display: flex; align-items: center; gap: 8px; flex: 0 0 10% !important; margin-bottom: 10px !important; }
    .send-btn button { background: #3b82f6 !important; color: white !important; border: none !important; min-height: 34px !important; border-radius: 14px !important; }
    .msg-input { flex: 8 1 0 !important; min-width: 0 !important; }
    .msg-input textarea { height: 100% !important; min-height: 100% !important; box-sizing: border-box !important; margin: 0 !important; }
    /* 响应式回退 */
    @media screen and (max-width: 768px) {
        .main-row { flex-direction: column !important; }
        .sidebar { min-width: 100% !important; min-height: auto !important; height: auto !important; flex-direction: row !important; }
        .sidebar-list { flex-direction: row !important; max-height: 60px !important; }
    }
    """

css = _load_css()


def _load_js():
    """加载外部 JS 文件，返回纯 JS 文本（供 gr.Blocks js 参数）"""
    js_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "interactions.js")
    try:
        with open(js_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[WARN] JS 文件未找到: {js_path}")
        return ""


interactions_js = _load_js()

def _load_js_wrapped():
    """加载外部 JS 文件，返回 <script> 包裹（供 gr.HTML 注入）"""
    js = _load_js()
    return f"<script>{js}</script>" if js else ""

# ========== 4. 回调函数 ==========

# ---- 新增：登录/注册回调 ----
def do_login(username, password):
    """登录"""
    if not username or not password:
        return "", "", "⚠️ 请输入用户名和密码", ""
    try:
        resp = httpx.post(
            f"{API_BASE}/login",
            json={"username": username, "password": password},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            # 登录成功，加载该用户的会话
            load_user_sessions(username)
            token = data["access_token"]
            user = data["username"]
            # 持久化 token 到文件（供删除会话等操作使用）
            _save_token_to_file(username, token)
            js_code = f'<script>if(window.saveAuth)window.saveAuth("{token}","{user}");</script>'
            return token, user, "✅ 登录成功！", js_code
        return "", "", resp.json().get("detail", "登录失败"), ""
    except httpx.ConnectError:
        return "", "", "⚠️ 后端未启动", ""
    except Exception as e:
        return "", "", str(e), ""


def do_register(username, password):
    """注册"""
    if not username or not password:
        return "", "", "⚠️ 请输入用户名和密码", ""
    if len(username) < 3:
        return "", "", "⚠️ 用户名至少3个字符", ""
    if len(password) < 6:
        return "", "", "⚠️ 密码至少6个字符", ""
    try:
        resp = httpx.post(
            f"{API_BASE}/register",
            json={"username": username, "password": password},
            timeout=10
        )
        if resp.status_code == 201:
            # 注册成功自动登录
            return do_login(username, password)
        return "", "", resp.json().get("detail", "注册失败"), ""
    except httpx.ConnectError:
        return "", "", "⚠️ 后端未启动", ""
    except Exception as e:
        return "", "", str(e), ""


def do_logout(token_state):
    """退出登录"""
    # 保存当前用户的会话
    global all_sessions, current_id, current_username
    if current_username:
        save_user_sessions(current_username)
    # 清空内存
    all_sessions = []
    current_id = None
    current_username = ""
    js_code = '<script>if(window.clearAuth)window.clearAuth();</script>'
    return "", "", gr.update(visible=True), gr.update(visible=False), js_code, gr.update(choices=[], value=None), []


# ---- 原有回调（完全不变）----

def on_upload(file, chatbot, token):
    """上传文件 → 调 FastAPI /upload → 入库 ChromaDB"""
    sess = get_current()
    if file is None:
        return None, chatbot

    file_path = file if isinstance(file, str) else str(file)
    original_name = os.path.basename(file_path)

    # 准备请求头（携带 JWT token）
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with open(file_path, "rb") as f:
            files = {"file": (original_name, f, "application/octet-stream")}
            resp = httpx.post(f"{API_BASE}/upload", files=files, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()

        summary = f"📎 已上传：{original_name} → 知识库（{data['chunks']} 个片段）\n现在你可以基于AIGC资料向我提问了。"
        chatbot.append({"role": "assistant", "content": summary})
        sess["messages"] = chatbot
        sess["file_context"] = original_name  # 记录文件名
        save_user_sessions(current_username)
        return original_name, chatbot

    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        err = f"⚠️ 上传失败：{detail}"
        chatbot.append({"role": "assistant", "content": err})
        sess["messages"] = chatbot
        save_user_sessions(current_username)
        return sess["file_context"], chatbot
    except httpx.ConnectError:
        err = "⚠️ 后端服务未启动，请先运行 course_assistant_api.py"
        chatbot.append({"role": "assistant", "content": err})
        sess["messages"] = chatbot
        save_user_sessions(current_username)
        return sess["file_context"], chatbot
    except Exception as e:
        err = f"⚠️ 上传失败：{e}"
        chatbot.append({"role": "assistant", "content": err})
        sess["messages"] = chatbot
        save_user_sessions(current_username)
        return sess["file_context"], chatbot


def respond(user_message, chatbot, file_context, token):
    """发消息 → 调 FastAPI /chat/stream（SSE 流式）"""
    sess = get_current()
    session_id = sess["id"]

    # 先追加用户消息到本地
    sess["messages"].append({"role": "user", "content": user_message})
    sess["messages"].append({"role": "assistant", "content": ""})
    save_user_sessions(current_username)

    # 🔑 立即 yield，让用户消息马上显示（不等 AI 回复）
    yield "", sess["messages"], sess["file_context"]

    ai_reply = ""

    # 准备请求头（携带 JWT token）
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}/chat/stream",
            json={"message": user_message, "session_id": session_id},
            headers=headers,
            timeout=60,
        ) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                payload = line[6:]  # 去掉 "data: " 前缀
                if not payload.strip():
                    continue

                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                thinking = event.get("thinking", False)
                token = event.get("token", "")
                done = event.get("done", False)

                if thinking:
                    sess["messages"][-1]["content"] = "⏳ 正在思考..."
                    save_user_sessions(current_username)
                    yield "", sess["messages"], sess["file_context"]
                    continue

                if token:
                    ai_reply += token
                    sess["messages"][-1]["content"] = ai_reply
                    save_user_sessions(current_username)
                    yield "", sess["messages"], sess["file_context"]

                if done:
                    source = event.get("source", "unknown")
                    docs = event.get("docs", [])
                    followup = event.get("followup", [])
                    # 追加来源/模式标记
                    if source == "rag":
                        mode_note = f"\n\n📚 参考：{'、'.join(docs[:3])}"
                    elif source == "casual":
                        mode_note = f"\n\n💬 闲聊模式"
                    else:
                        mode_note = ""
                    if mode_note:
                        ai_reply += mode_note
                    # 追加后续推荐问题
                    if followup:
                        ai_reply += "\n\n💡 **你可能还想问：**\n"
                        for q in followup:
                            ai_reply += f"• {q}\n"
                    if mode_note or followup:
                        sess["messages"][-1]["content"] = ai_reply
                        save_user_sessions(current_username)
                        yield "", sess["messages"], sess["file_context"]
                    break

    except httpx.ConnectError:
        err = "⚠️ 后端服务未启动，请先运行 course_assistant_api.py"
        sess["messages"][-1]["content"] = err
        save_user_sessions(current_username)
        yield "", sess["messages"], sess["file_context"]
    except Exception as e:
        err = f"⚠️ 请求失败：{e}"
        sess["messages"][-1]["content"] = err
        save_user_sessions(current_username)
        yield "", sess["messages"], sess["file_context"]

    # 更新会话标题（取用户第一条消息前 15 字）
    if len(sess["messages"]) == 2:
        title = user_message[:15]
        sess["title"] = title
        save_user_sessions(current_username)


def new_conversation():
    """新建对话"""
    global current_id
    new_sess = make_session()
    all_sessions.append(new_sess)
    current_id = new_sess["id"]
    save_user_sessions(current_username)
    return [], None, gr.update(choices=get_session_choices(), value=current_id)


def switch_session(session_id):
    """切换到选中会话"""
    global current_id
    current_id = session_id
    sess = get_current()
    return sess["messages"], sess["file_context"]


def _delete_session_on_backend(sid):
    """调用后端 API 删除数据库中的会话数据"""
    try:
        token = _get_stored_token()
        if not token:
            return
        headers = {"Authorization": f"Bearer {token}"}
        httpx.delete(f"{API_BASE}/sessions/{sid}", headers=headers, timeout=5)
    except Exception:
        pass  # 后端不可达时静默忽略


def _save_token_to_file(username, token):
    """持久化 token 到文件"""
    token_file = os.path.join(SESSIONS_DIR, f"{username}_token.txt")
    try:
        with open(token_file, "w") as f:
            f.write(token)
    except Exception:
        pass


def _get_stored_token():
    """从文件读取持久化的 token"""
    token_file = os.path.join(SESSIONS_DIR, f"{current_username}_token.txt")
    try:
        with open(token_file, "r") as f:
            return f.read().strip()
    except Exception:
        return ""


def delete_current_session():
    """删除当前选中的会话（前端本地 + 后端数据库）"""
    global all_sessions, current_id
    if len(all_sessions) <= 1:
        return gr.update(choices=get_session_choices(), value=current_id), [], None
    sid = current_id
    all_sessions = [s for s in all_sessions if s["id"] != sid]
    current_id = all_sessions[-1]["id"]
    save_user_sessions(current_username)
    # 同步删除后端数据库中的会话
    _delete_session_on_backend(sid)
    sess = get_current()
    return (
        gr.update(choices=get_session_choices(), value=current_id),
        sess["messages"],
        sess["file_context"],
    )


def do_clear_chat():
    """清空当前会话消息（UI + 后端同步）"""
    sess = get_current()
    sess["messages"] = []
    save_user_sessions(current_username)
    return [], None


def do_copy_chat():
    """返回对话文本"""
    return copy_chat_text()


def do_share_chat():
    """整个对话分享 → 返回分享链接（JS 轮询复制）"""
    sess = get_current()
    msgs = sess["messages"]
    if not msgs:
        return "⚠️ 暂无对话"

    try:
        token = _get_stored_token()
        if not token:
            return "⚠️ 请先登录"
        headers = {"Authorization": f"Bearer {token}"}
        resp = httpx.post(
            f"{API_BASE}/share",
            json={"messages": msgs, "title": sess.get("title", "对话")},
            headers=headers, timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return f"{API_BASE}{data['url']}"
        return "⚠️ 分享失败"
    except Exception as e:
        return f"⚠️ {e}"


def clear_chat():
    """清空当前会话所有消息"""
    sess = get_current()
    sess["messages"] = []
    save_user_sessions(current_username)
    return [], None


def export_chat_markdown():
    """导出对话为 Markdown 文本"""
    sess = get_current()
    msgs = sess["messages"]
    if not msgs:
        return ""
    lines = [f"# {sess['title']}", f"日期：{sess['created_at']}", "", "---", ""]
    for m in msgs:
        role = "**用户**" if m["role"] == "user" else "**小课**"
        lines.append(f"{role}：\n{m['content']}\n")
    return "\n".join(lines)


def copy_chat_text():
    """返回纯文本格式的对话（供 JS 复制）"""
    sess = get_current()
    msgs = sess["messages"]
    if not msgs:
        return ""
    lines = []
    for m in msgs:
        role = "用户" if m["role"] == "user" else "小课"
        lines.append(f"{role}：{m['content']}")
    return "\n\n".join(lines)


# ========== 5. 搭建界面 ==========

with gr.Blocks(theme=gr.themes.Default()) as demo:
    # 全局状态
    token_state = gr.State("")
    current_user = gr.State("")

    # 隐藏组件：JS 执行桥接（visible=False 保证 script 执行但不显示）
    _js_bridge = gr.HTML("", visible=False)

    # 注入 JavaScript（处理 localStorage 认证恢复，visible=False 避免 script 标签裸露）
    gr.HTML("""
    <script>
    (function() {
        // 保存认证信息到 localStorage
        window.saveAuth = function(token, username) {
            localStorage.setItem('pbl_token', token);
            localStorage.setItem('pbl_username', username);
        };

        // 清除认证信息
        window.clearAuth = function() {
            localStorage.removeItem('pbl_token');
            localStorage.removeItem('pbl_username');
        };

        // 页面加载后检查是否有保存的会话，如果有则显示恢复按钮
        setTimeout(function() {
            var token = localStorage.getItem('pbl_token');
            var username = localStorage.getItem('pbl_username');
            if (token && username) {
                var restoreBtn = document.querySelector('.restore-btn');
                if (restoreBtn) {
                    restoreBtn.style.display = 'block';
                    var btn = restoreBtn.querySelector('button');
                    if (btn) {
                        btn.addEventListener('click', function() {
                            var userInput = document.querySelector('#login-username-input input, #login-username-input textarea');
                            if (userInput) {
                                userInput.value = username;
                                userInput.dispatchEvent(new Event('input', {bubbles: true}));
                            }
                            var pwdInput = document.querySelector('#login-password-input input, #login-password-input textarea');
                            if (pwdInput) pwdInput.focus();
                            var msgEl = document.querySelector('.login-msg');
                            if (msgEl) {
                                msgEl.innerHTML = '<p style=\"color:#2563eb;\">已恢复用户名，请输入密码登录</p>';
                            }
                        });
                    }
                }
            }
        }, 600);
    })();
    </script>
    """, visible=False)

    # 交互 JS 已通过 gr.Blocks(js=...) 加载，无需额外注入

    # ========== 登录表单（默认显示）==========
    with gr.Column(visible=True, elem_classes="login-form") as login_form:
        gr.HTML('<div class="login-card"><h2>🔐 登录 / 注册</h2><p class="subtitle">AIGC 课程助手 · 智能问答平台</p></div>')
        with gr.Row():
            with gr.Column():
                login_user = gr.Textbox(
                    label="用户名", placeholder="至少3个字符",
                    elem_id="login-username-input",
                )
                login_pwd = gr.Textbox(
                    label="密码", type="password", placeholder="至少6个字符",
                    elem_id="login-password-input",
                )
                with gr.Row():
                    login_btn = gr.Button("登录", variant="primary", elem_classes="btn-login")
                    reg_btn = gr.Button("注册", elem_classes="btn-register")
                # 恢复会话按钮（默认隐藏，JS 会显示它）
                restore_btn = gr.Button("🔄 恢复会话", visible=False, elem_classes="restore-btn")
                login_msg = gr.Markdown("", elem_classes="login-msg")

    # ========== 聊天界面（默认隐藏，登录后显示）==========
    with gr.Column(visible=False) as chat_interface:
        file_ctx = gr.State(None)

        # 顶栏
        with gr.Row(elem_classes="top-bar", elem_id="top-bar"):
            welcome_text = gr.Markdown("欢迎！", scale=8)
            exit_btn = gr.Button("退出", elem_classes="exit-btn", scale=0, min_width=60)
            gr.HTML('<div style="white-space:nowrap;font-weight:500;text-align:right;" role="heading" aria-level="1">🤖 AIGC咨询助手</div>', scale=1, elem_classes="top-bar-title")

        # 主体
        with gr.Row(equal_height=True, elem_classes="main-row"):
            # 侧边栏
            with gr.Column(scale=2, min_width=180, elem_classes="sidebar", elem_id="sidebar"):
                with gr.Column(elem_classes="sidebar-new"):
                    new_btn = gr.Button("＋ 新建对话", elem_classes="new-btn", elem_id="new-conversation-btn")

                with gr.Column(elem_classes="sidebar-header"):
                    gr.Markdown("### 聊天记录")
                    gr.HTML('<div class="history-divider" role="separator" aria-hidden="true"></div>')

                with gr.Column(elem_classes="sidebar-list"):
                    session_radio = gr.Radio(
                        choices=get_session_choices(),
                        label="会话列表",
                        show_label=False,
                        value=current_id,
                        elem_classes="session-radio",
                        elem_id="session-radio-list",
                    )
                    delete_btn = gr.Button(
                        "🗑 删除会话",
                        elem_classes="delete-session-btn",
                        elem_id="session-delete-btn",
                        variant="stop",
                        size="sm",
                    )

            # 聊天区
            with gr.Column(scale=8, elem_classes="chat-area", elem_id="chat-area"):
                gr.Markdown("## 欢迎使用AIGC咨询助手", elem_classes="chat-title")

                # 聊天室工具栏按钮（出现在 Chatbot 右上角原生面板）
                _toolbar_clear = gr.Button("🗑 清空对话", size="sm")
                _toolbar_share = gr.Button("🔗 分享链接", size="sm")

                chatbot = gr.Chatbot(
                    layout="bubble", feedback_options=None,
                    value=get_current()["messages"],
                    elem_id="main-chatbot",
                    label="对话区",
                    show_label=False,
                    buttons=["copy_all", _toolbar_clear, _toolbar_share],
                )

                with gr.Row(elem_classes="input-row"):
                    upload_btn = gr.UploadButton(
                        "📎 上传", elem_classes="upload-file",
                        file_types=[".pdf", ".docx", ".txt", ".pptx", ".html", ".ipynb"],
                        elem_id="upload-file-btn",
                    )
                    msg_box = gr.Textbox(
                        placeholder="输入你的消息...（Enter 发送，Shift+Enter 换行）",
                        container=False, elem_classes="msg-input",
                        elem_id="message-input",
                        label="消息输入",
                        show_label=False,
                        lines=1,
                        max_lines=6,
                        autofocus=True,
                    )
                    send_btn = gr.Button("📤 发送", elem_classes="send-btn", elem_id="send-message-btn")

        # 隐藏桥接（CSS 隐藏但保持 DOM 事件）
        _share_result = gr.Textbox(value="", elem_classes="hidden-btn", elem_id="share-result")
        _btn_clear = gr.Button("clear", elem_classes="hidden-btn", elem_id="btn-clear-chat")
        _btn_share = gr.Button("share", elem_classes="hidden-btn", elem_id="btn-share-chat")

    # ========== 6. 绑定事件 ==========
    # 登录成功 → 隐藏登录表单，显示聊天界面
    def on_login_success(token, username, js_code):
        if token:
            sess = get_current()
            return (
                gr.update(visible=False),  # 隐藏登录表单
                gr.update(visible=True),   # 显示聊天界面
                f"欢迎，{username}！",      # 更新欢迎语
                js_code,                    # 执行 JS 保存 token（隐藏组件）
                gr.update(choices=get_session_choices(), value=current_id),
                sess["messages"]
            )
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            "",
            gr.update(choices=[], value=None),
            []
        )

    login_btn.click(
        do_login, [login_user, login_pwd], [token_state, current_user, login_msg, _js_bridge]
    ).then(
        on_login_success, [token_state, current_user, _js_bridge], [login_form, chat_interface, welcome_text, _js_bridge, session_radio, chatbot]
    )

    # 注册成功 → 自动登录
    reg_btn.click(
        do_register, [login_user, login_pwd], [token_state, current_user, login_msg, _js_bridge]
    ).then(
        on_login_success, [token_state, current_user, _js_bridge], [login_form, chat_interface, welcome_text, _js_bridge, session_radio, chatbot]
    )

        # 退出 → 显示登录表单，隐藏聊天界面
    exit_btn.click(
        do_logout, [token_state], [token_state, current_user, login_form, chat_interface, _js_bridge, session_radio, chatbot]
    )

    # 聊天功能
    send_btn.click(respond, [msg_box, chatbot, file_ctx, token_state], [msg_box, chatbot, file_ctx])
    msg_box.submit(respond, [msg_box, chatbot, file_ctx, token_state], [msg_box, chatbot, file_ctx])
    new_btn.click(new_conversation, [], [chatbot, file_ctx, session_radio])
    upload_btn.upload(on_upload, [upload_btn, chatbot, token_state], [file_ctx, chatbot])
    session_radio.change(switch_session, [session_radio], [chatbot, file_ctx])
    delete_btn.click(delete_current_session, [], [session_radio, chatbot, file_ctx])
    _btn_clear.click(do_clear_chat, [], [chatbot, file_ctx])
    _btn_share.click(do_share_chat, [], [_share_result])

    # 原生工具栏按钮
    _toolbar_clear.click(do_clear_chat, [], [chatbot, file_ctx])
    _toolbar_share.click(do_share_chat, [], [_share_result])
    chatbot.clear(do_clear_chat, [], [chatbot, file_ctx])



# ========== 7. 启动 ==========
if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0", server_port=7860, inbrowser=True, css=css,
        js=interactions_js,
    )
