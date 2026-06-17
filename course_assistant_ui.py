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

# ========== 2. 会话持久化（本地 JSON，UI 层面）==========

SESSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.json")
all_sessions = []
current_id = None


def load_sessions():
    global all_sessions
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                all_sessions = json.load(f)
        except Exception:
            all_sessions = []
    else:
        all_sessions = []


def save_sessions():
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, ensure_ascii=False, indent=2)


def make_session(title="新对话"):
    return {
        "id": str(uuid.uuid4())[:8],
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


def get_session_choices():
    choices = []
    for s in reversed(all_sessions):
        label = f"💠 {s['title']}  ({s['created_at']})"
        choices.append((label, s["id"]))
    return choices


# 启动时加载
load_sessions()
if not all_sessions:
    first = make_session()
    all_sessions.append(first)
    save_sessions()
current_id = all_sessions[-1]["id"]


# ========== 3. CSS（和 step7.py 完全一样）==========

css = """
/* ===== 全局重置 ===== */
html, body { margin: 0 !important; padding: 0 !important; }
.gradio-container, .block-container { margin: 0 !important; padding: 0 !important; max-width: 100% !important; }
.app { margin: 0 !important; padding: 0 !important; width: 100% !important; height: 100% !important; }
footer, .footer { display: none !important; }

/* 隐藏侧边栏内可能出现的额外元素 */
.sidebar footer, .sidebar .footer,
.sidebar > div:last-of-type:empty {
    display: none !important;
}

/* ===== 顶栏 ===== */
.top-bar {
    background: #1e293b !important; color: white !important;
    padding: 8px 16px; border-radius: 0;
    display: flex; justify-content: space-between; margin-bottom: 0 !important;
}
.top-bar, .top-bar p, .top-bar span, .top-bar div, .top-bar .prose, .top-bar .prose p { color: white !important; }

/* ===== 侧边栏容器 ===== */
.sidebar,
div[id*="sidebar"],
.column.sidebar {
    background: #f8fafc !important;
    padding: 0 16px 0 16px !important;
    min-height: calc(100vh - 64px) !important;
    height: calc(100vh - 64px) !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}
.sidebar > [style*="flex-grow: 1"] {
    flex-grow: 0 !important;
}
.sidebar *,
.sidebar *::before,
.sidebar *::after {
    background-color: transparent !important;
}
.sidebar .column,
.sidebar > .column,
.sidebar [class*="column"] {
    background: #f8fafc !important;
}

/* ===== 新建对话按钮区域（占 5%）===== */
.sidebar-new {
    flex: 0 0 auto !important;
    height: 5% !important;
    min-height: 32px !important;
    display: flex !important;
    align-items: center !important;
    overflow: hidden !important;
    padding: 0 12px !important;
}

/* ===== 新建对话按钮样式 ===== */
.new-btn, .new-btn button, .new-btn button span {
    background: linear-gradient(90deg, #5eead4 0%, #3b82f6 100%) !important;
    color: #0f172a !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    height: 24px !important;
    line-height: 18px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* ===== 头部：聊天记录 + 分割线 ===== */
.sidebar-header {
    flex: 0 0 auto !important;
    height: 15% !important;
    min-height: 40px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    overflow: hidden !important;
}
.sidebar-header .prose, .sidebar-header h3 {
    font-size: 20px !important;
    color: #475569 !important;
    padding: 0 0 0 12px !important;
    overflow: hidden !important;
}

/* ===== 分割线 ===== */
.history-divider {
    width: 100% !important;
    height: 2px !important;
    background: #5eead4 !important;
    margin: 2px 0 0 0 !important;
    display: block !important;
}

/* ===== 会话列表区域 ===== */
.sidebar-list {
    flex: 0 0 70% !important;
    height: 70% !important;
    overflow-y: auto !important;
    margin: 0 !important;
    padding: 4px 0 !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}
.sidebar-list::-webkit-scrollbar {
    display: none !important;
}
.sidebar-list {
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

/* 隐藏 Radio label */
.sidebar-list > div > label:first-child,
.sidebar-list label:first-child {
    display: none !important;
}
.session-radio {
    width: 100% !important;
    height: 100% !important;
    flex: 1 1 auto !important;
}
.session-radio .radio-group,
.session-radio [role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
    width: 100% !important;
    height: 100% !important;
}

/* ===== 退出按钮 ===== */
.exit-btn, .exit-btn button, .exit-btn [role="button"] {
    background: #ef4444 !important; color: white !important; border: none !important;
    border-radius: 14px !important; width: auto !important; min-width: 0 !important;
    height: 34px !important; min-height: 34px !important; font-size: 12px !important;
    padding: 0 12px !important; display: flex !important;
    align-items: center !important; justify-content: center !important;
}

/* ===== 聊天区 ===== */
.chat-area { background: white; padding: 0; min-height: calc(100vh - 64px); box-sizing: border-box;
             display: flex !important; flex-direction: column !important; }
.chat-title { height: 5% !important; min-height: 5% !important; margin: 0 !important;
              overflow: hidden !important; flex: 0 0 5% !important; text-align: center !important; }
.chat-area .chatbot { margin-top: 2px !important; flex: 0 0 85% !important; min-height: 85% !important; }

/* ===== 主体区域 ===== */
.main-row { min-height: calc(100vh - 64px); align-items: stretch; gap: 0 !important; margin-top: 0 !important; }
.main-row > .column { margin: 0 !important; padding: 0 !important; }

/* ===== 输入区外框 ===== */
.input-row {
    background: white; border: 1px solid #d1d5db; border-radius: 28px;
    padding: 4px 8px; display: flex; align-items: center; gap: 8px;
    flex: 0 0 10% !important; min-height: 10% !important; height: 10% !important;
    margin-bottom: 10px !important;
}
.input-row > * { min-width: 0 !important; }

/* ===== 上传/发送按钮 ===== */
.upload-file, .upload-file > button, .upload-file [role="button"],
.send-btn, .send-btn button, .send-btn [role="button"] {
    min-height: 34px !important; height: 34px !important; margin: 0 !important;
    width: 100% !important; min-width: 0 !important; padding: 0 !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    border-radius: 14px !important;
}
.send-btn, .send-btn button, .send-btn [role="button"] {
    background: #3b82f6 !important; color: white !important; border: none !important;
}
.send-btn:hover { background: #2563eb !important; }

.msg-input { flex: 8 1 0 !important; min-width: 0 !important;
              height: calc(100% - 20%); display: flex !important; }
.msg-input textarea { height: 100% !important; min-height: 100% !important;
                     box-sizing: border-box !important; padding-top: 0 !important;
                     padding-bottom: 0 !important; margin: 0 !important; }
"""


# ========== 4. 回调函数 ==========

def on_upload(file, chatbot):
    """上传文件 → 调 FastAPI /upload → 入库 ChromaDB"""
    sess = get_current()
    if file is None:
        return None, chatbot

    file_path = file if isinstance(file, str) else str(file)
    original_name = os.path.basename(file_path)

    try:
        with open(file_path, "rb") as f:
            files = {"file": (original_name, f, "application/octet-stream")}
            resp = httpx.post(f"{API_BASE}/upload", files=files, timeout=60)
            resp.raise_for_status()
            data = resp.json()

        summary = f"📎 已上传：{original_name} → 知识库（{data['chunks']} 个片段）\n现在你可以基于课程资料向我提问了。"
        chatbot.append({"role": "assistant", "content": summary})
        sess["messages"] = chatbot
        sess["file_context"] = original_name  # 记录文件名
        save_sessions()
        return original_name, chatbot

    except httpx.ConnectError:
        err = "⚠️ 后端服务未启动，请先运行 course_assistant_api.py"
        chatbot.append({"role": "assistant", "content": err})
        sess["messages"] = chatbot
        save_sessions()
        return sess["file_context"], chatbot
    except Exception as e:
        err = f"⚠️ 上传失败：{e}"
        chatbot.append({"role": "assistant", "content": err})
        sess["messages"] = chatbot
        save_sessions()
        return sess["file_context"], chatbot


def respond(user_message, chatbot, file_context):
    """发消息 → 调 FastAPI /chat/stream（SSE 流式）"""
    sess = get_current()
    session_id = sess["id"]

    # 先追加用户消息到本地
    sess["messages"].append({"role": "user", "content": user_message})
    sess["messages"].append({"role": "assistant", "content": ""})
    save_sessions()

    ai_reply = ""

    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}/chat/stream",
            json={"message": user_message, "session_id": session_id},
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

                token = event.get("token", "")
                done = event.get("done", False)

                if token:
                    ai_reply += token
                    sess["messages"][-1]["content"] = ai_reply
                    save_sessions()
                    yield "", sess["messages"], sess["file_context"]

                if done:
                    source = event.get("source", "unknown")
                    docs = event.get("docs", [])
                    # 追加来源/模式标记（让用户看到意图路由结果）
                    if source == "rag":
                        mode_note = f"\n\n📚 参考：{'、'.join(docs[:3])}"
                    elif source == "casual":
                        mode_note = f"\n\n💬 闲聊模式"
                    else:
                        mode_note = ""
                    if mode_note:
                        ai_reply += mode_note
                        sess["messages"][-1]["content"] = ai_reply
                        save_sessions()
                        yield "", sess["messages"], sess["file_context"]
                    break

    except httpx.ConnectError:
        err = "⚠️ 后端服务未启动，请先运行 course_assistant_api.py"
        sess["messages"][-1]["content"] = err
        save_sessions()
        yield "", sess["messages"], sess["file_context"]
    except Exception as e:
        err = f"⚠️ 请求失败：{e}"
        sess["messages"][-1]["content"] = err
        save_sessions()
        yield "", sess["messages"], sess["file_context"]

    # 更新会话标题（取用户第一条消息前 15 字）
    if len(sess["messages"]) == 2:
        title = user_message[:15]
        sess["title"] = title
        save_sessions()


def new_conversation():
    """新建对话"""
    global current_id
    new_sess = make_session()
    all_sessions.append(new_sess)
    current_id = new_sess["id"]
    save_sessions()
    return [], None, gr.update(choices=get_session_choices(), value=current_id)


def switch_session(session_id):
    """切换到选中会话"""
    global current_id
    current_id = session_id
    sess = get_current()
    title = sess["title"]
    if len(title) > 15:
        title = title[:15] + "..."
    return sess["messages"], sess["file_context"]


# ========== 5. 搭建界面 ==========

with gr.Blocks() as demo:
    file_ctx = gr.State(None)

    # 顶栏
    with gr.Row(elem_classes="top-bar"):
        gr.Markdown("欢迎，18121339701！", scale=8)
        gr.Button("🔴 退出", elem_classes="exit-btn", scale=2)

    # 主体
    with gr.Row(equal_height=True, elem_classes="main-row"):
        # 侧边栏
        with gr.Column(scale=2, min_width=180, elem_classes="sidebar"):
            with gr.Column(elem_classes="sidebar-new"):
                new_btn = gr.Button("＋ 新建对话", elem_classes="new-btn")

            with gr.Column(elem_classes="sidebar-header"):
                gr.Markdown("### 聊天记录")
                gr.HTML('<div class="history-divider"></div>')

            with gr.Column(elem_classes="sidebar-list"):
                session_radio = gr.Radio(
                    choices=get_session_choices(),
                    label="",
                    show_label=False,
                    value=current_id,
                    elem_classes="session-radio",
                )

        # 聊天区
        with gr.Column(scale=8, elem_classes="chat-area"):
            gr.Markdown("## 欢迎使用近屿智能课程咨询助手", elem_classes="chat-title")
            chatbot = gr.Chatbot(
                layout="bubble", feedback_options=None,
                value=get_current()["messages"],
            )

            with gr.Row(elem_classes="input-row"):
                upload_btn = gr.UploadButton(
                    "📤 上传", elem_classes="upload-file",
                    file_types=[".pdf", ".docx", ".txt", ".pptx", ".html", ".ipynb"],
                )
                msg_box = gr.Textbox(placeholder="输入你的消息...", container=False, elem_classes="msg-input")
                send_btn = gr.Button("📤 发送", elem_classes="send-btn")

    # ========== 6. 绑定事件 ==========
    send_btn.click(respond, [msg_box, chatbot, file_ctx], [msg_box, chatbot, file_ctx])
    msg_box.submit(respond, [msg_box, chatbot, file_ctx], [msg_box, chatbot, file_ctx])
    new_btn.click(new_conversation, [], [chatbot, file_ctx, session_radio])
    upload_btn.upload(on_upload, [upload_btn, chatbot], [file_ctx, chatbot])
    session_radio.change(switch_session, [session_radio], [chatbot, file_ctx])

# ========== 7. 启动 ==========
if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0", server_port=7860, inbrowser=True, css=css,
    )
