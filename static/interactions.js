/**
 * Course Assistant — 交互增强脚本 v2.0
 * 
 * 功能：
 * 1. Enter 发送 / Shift+Enter 换行
 * 2. 自动聚焦消息输入框
 * 3. Toast 通知系统
 * 4. 键盘快捷键 (Ctrl+N 新建, Ctrl+数字 切换会话)
 * 5. 加载状态指示器
 * 6. 会话列表 hover 删除按钮
 * 7. 聊天界面可见性监听（登录后自动聚焦）
 * 8. 暗色模式切换
 */

(function () {
  "use strict";

  // ========== 工具函数 ==========

  /** 等待 DOM 中出现匹配的元素 */
  function waitFor(selector, timeout = 8000) {
    return new Promise((resolve, reject) => {
      const el = document.querySelector(selector);
      if (el) return resolve(el);

      const observer = new MutationObserver(() => {
        const el = document.querySelector(selector);
        if (el) {
          observer.disconnect();
          resolve(el);
        }
      });

      observer.observe(document.body, { childList: true, subtree: true });

      setTimeout(() => {
        observer.disconnect();
        reject(new Error(`等待超时: ${selector}`));
      }, timeout);
    });
  }

  /** 查找 Gradio 的 textarea */
  function findMessageTextarea() {
    // Gradio Textbox 渲染为一个容器，内含 textarea
    const textareas = document.querySelectorAll('textarea[data-testid="textbox"]');
    if (textareas.length > 0) return textareas[textareas.length - 1];

    // 回退：找所有可见 textarea
    const allTextareas = document.querySelectorAll('textarea:not([style*="display: none"])');
    for (const ta of allTextareas) {
      if (ta.offsetParent !== null && ta.placeholder && ta.placeholder.includes("消息")) {
        return ta;
      }
    }
    return allTextareas[allTextareas.length - 1] || null;
  }

  /** 查找发送按钮 */
  function findSendButton() {
    // 查找包含"发送"文字的按钮
    const buttons = document.querySelectorAll('button');
    for (const btn of buttons) {
      if (btn.textContent && btn.textContent.includes("发送")) {
        return btn;
      }
    }
    return null;
  }

  // ========== 1. Enter 发送 / Shift+Enter 换行 ==========

  let enterBound = false;

  function installEnterHandler() {
    if (enterBound) return;
    enterBound = true;

    // 使用 capture 阶段，确保在 Gradio 内部处理器之前拦截
    document.addEventListener("keydown", function (e) {
      const textarea = findMessageTextarea();
      if (!textarea) return;

      // 只处理聚焦在消息输入框时
      if (document.activeElement !== textarea) return;

      // 忽略 IME 输入法组合中的 Enter
      if (e.isComposing || e.keyCode === 229) return;

      // Enter 发送（不按 Shift/Ctrl/Meta）
      if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

        const sendBtn = findSendButton();
        if (sendBtn) {
          sendBtn.click();
        }
      }
    }, true); // capture: true — 在冒泡前拦截

    console.log("[交互] Enter 发送 / Shift+Enter 换行 已安装");
  }

  // ========== 2. 自动聚焦消息输入框 ==========

  function autoFocusMessageInput() {
    setTimeout(() => {
      const textarea = findMessageTextarea();
      if (textarea) {
        textarea.focus();
        console.log("[交互] 消息输入框已自动聚焦");
      }
    }, 300);
  }

  // ========== 3. Toast 通知系统 ==========

  /** 显示 Toast 通知 */
  function showToast(message, type = "info", duration = 3000) {
    // 移除旧 toast
    const oldToast = document.querySelector(".wb-toast");
    if (oldToast) oldToast.remove();

    const toast = document.createElement("div");
    toast.className = `wb-toast wb-toast-${type}`;
    toast.innerHTML = message;

    // 样式
    const colors = {
      success: { bg: "#ecfdf5", border: "#6ee7b7", text: "#065f46" },
      error: { bg: "#fef2f2", border: "#fca5a5", text: "#991b1b" },
      warning: { bg: "#fffbeb", border: "#fcd34d", text: "#92400e" },
      info: { bg: "#eff6ff", border: "#93c5fd", text: "#1e40af" },
    };
    const c = colors[type] || colors.info;

    Object.assign(toast.style, {
      position: "fixed",
      top: "16px",
      left: "50%",
      transform: "translateX(-50%)",
      padding: "10px 20px",
      background: c.bg,
      border: `1px solid ${c.border}`,
      color: c.text,
      borderRadius: "8px",
      fontSize: "13px",
      fontWeight: "500",
      zIndex: "9999",
      boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
      opacity: "0",
      transition: "opacity 200ms ease, transform 200ms ease",
      maxWidth: "90vw",
      textAlign: "center",
    });

    document.body.appendChild(toast);

    // 入场动画
    requestAnimationFrame(() => {
      toast.style.opacity = "1";
      toast.style.transform = "translateX(-50%) translateY(0)";
    });

    // 自动消失
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(-50%) translateY(-8px)";
      setTimeout(() => toast.remove(), 200);
    }, duration);
  }

  // 暴露到 window（Python 可通过 JS 调用）
  window.showToast = showToast;

  // ========== 4. 键盘快捷键 ==========

  function installKeyboardShortcuts() {
    document.addEventListener("keydown", function (e) {
      // 不在输入框内时生效
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;

      // Ctrl+N / Cmd+N：新建对话
      if ((e.ctrlKey || e.metaKey) && e.key === "n") {
        e.preventDefault();
        const newBtn = document.querySelector('.new-btn button, button:has(span:contains("＋ 新建"))');
        if (newBtn) newBtn.click();
      }

      // Ctrl+数字：切换会话
      if ((e.ctrlKey || e.metaKey) && /^[1-9]$/.test(e.key)) {
        e.preventDefault();
        const idx = parseInt(e.key) - 1;
        const radios = document.querySelectorAll('.session-radio [role="radio"], .session-radio input[type="radio"]');
        if (radios[idx]) radios[idx].click();
      }
    });

    console.log("[交互] 键盘快捷键已安装");
  }

  // ========== 5. 加载状态指示器 ==========

  /** 在消息区域显示"思考中"指示器 */
  function showThinkingIndicator() {
    const chatbot = document.querySelector(".chatbot");
    if (!chatbot) return;

    // 检查是否已有指示器
    if (document.querySelector(".typing-indicator")) return;

    const indicator = document.createElement("div");
    indicator.className = "typing-indicator";
    indicator.innerHTML = `
      <span></span><span></span><span></span>
    `;

    // 内联样式
    const style = document.createElement("style");
    style.textContent = `
      .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 8px 16px;
        margin: 8px 0;
      }
      .typing-indicator span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #93c5fd;
        animation: typingBounce 1.4s infinite ease-in-out both;
      }
      .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
      .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
      .typing-indicator span:nth-child(3) { animation-delay: 0s; }
      @keyframes typingBounce {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
      }
    `;
    document.head.appendChild(style);

    const messagesContainer = chatbot.querySelector(".messages, .message-wrap, [class*='messages']");
    if (messagesContainer) {
      messagesContainer.appendChild(indicator);
    } else {
      chatbot.appendChild(indicator);
    }
  }

  /** 隐藏"思考中"指示器 */
  function hideThinkingIndicator() {
    const indicator = document.querySelector(".typing-indicator");
    if (indicator) indicator.remove();
  }

  window.showThinkingIndicator = showThinkingIndicator;
  window.hideThinkingIndicator = hideThinkingIndicator;

  // ========== 6. 会话列表 hover 删除 x 注入 ==========

  function injectDeleteIcons() {
    var labels = document.querySelectorAll('#session-radio-list label');
    if (!labels.length) {
      // Gradio 可能用不同的包装结构
      labels = document.querySelectorAll('.session-radio label');
    }
    if (!labels.length) return;

    labels.forEach(function (label) {
      if (label.querySelector('.session-del-x')) return;

      // 确保 label 是 relative 定位
      label.style.position = 'relative';

      var x = document.createElement('span');
      x.className = 'session-del-x';
      x.textContent = '\u00d7';  // ×
      x.title = '删除会话';

      x.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        if (!confirm('确定删除该会话？')) return;

        // 第一步：选中该会话的 radio input
        var radio = label.querySelector('input[type="radio"]');
        if (radio) radio.click();

        // 第二步：延迟点击隐藏的删除按钮触发 Python 回调
        setTimeout(function () {
          var btn = document.querySelector('#session-delete-btn button');
          if (!btn) {
            var allBtns = document.querySelectorAll('button');
            for (var i = 0; i < allBtns.length; i++) {
              if (allBtns[i].textContent.indexOf('删除') !== -1) {
                btn = allBtns[i]; break;
              }
            }
          }
          if (btn) btn.click();
        }, 200);
      });

      label.appendChild(x);
    });
  }

  // 立即尝试 + MutationObserver 兜底（Radio 可能异步渲染）
  function initDeleteIcons() {
    injectDeleteIcons();
    var observer = new MutationObserver(function () {
      injectDeleteIcons();
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  // ========== 7. 聊天界面可见性监听（登录后自动聚焦）==========

  let chatVisibilityObserver = null;

  function watchChatVisibility() {
    if (chatVisibilityObserver) return;

    chatVisibilityObserver = new MutationObserver(() => {
      // 检查聊天界面是否变得可见
      const chatColumn = document.querySelector('.chat-area');
      if (chatColumn && chatColumn.offsetParent !== null) {
        // 聊天界面可见，自动聚焦输入框
        autoFocusMessageInput();
      }
    });

    chatVisibilityObserver.observe(document.body, {
      attributes: true,
      attributeFilter: ['style', 'class'],
      subtree: true,
    });

    console.log("[交互] 聊天界面可见性监听已安装");
  }


  // ========== 7. 原生面板按钮 → 后端回调桥接 ==========

  function clickHiddenBtn(id) {
    var el = document.getElementById(id);
    if (!el) return;
    var btn = el.querySelector('button');
    if (btn) btn.click();
  }

  var _shareWatching = false, _lastShare = '';
  function watchShareResult() {
    if (_shareWatching) return;
    _shareWatching = true;
    setInterval(function () {
      var el = document.getElementById('share-result');
      if (!el) return;
      var ta = el.querySelector('textarea') || el.querySelector('input');
      if (!ta) return;
      var val = ta.value;
      if (!val || val === _lastShare) return;
      _lastShare = val;
      if (val.startsWith('http')) {
        navigator.clipboard.writeText(val).then(function () {
          window.showToast && window.showToast('分享链接已复制！', 'success');
        });
      } else if (val.startsWith('⚠')) {
        window.showToast && window.showToast(val, 'error');
      }
      var s = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set
            || Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;
      if (s) s.call(ta, ''); else ta.value = '';
      _lastShare = '';
    }, 500);
  }

  function initMessageActions() {
    watchShareResult();
  }

  // ========== 8. 强制禁用暗色模式（Gradio 会根据系统偏好加 .dark）==========

  function removeDarkClasses() {
    document.documentElement.classList.remove("dark");
    document.documentElement.removeAttribute("data-theme");
    document.documentElement.style.colorScheme = "light";
    document.body.classList.remove("dark");
    document.querySelectorAll(".dark").forEach(function (el) {
      el.classList.remove("dark");
    });
  }

  function watchDarkClass() {
    removeDarkClasses();
    const observer = new MutationObserver(function (mutations) {
      let needRemove = false;
      for (const m of mutations) {
        if (m.type === "attributes" && m.attributeName === "class") {
          if (m.target.classList && m.target.classList.contains("dark")) {
            needRemove = true;
          }
        }
      }
      if (needRemove) removeDarkClasses();
    });
    observer.observe(document.documentElement, { attributes: true, subtree: true, attributeFilter: ["class"] });
    // 兜底：定时扫描
    setInterval(removeDarkClasses, 2000);
  }

  // ── init ──

  let initialized = false;

  function init() {
    if (initialized) return;
    initialized = true;

    // 强制始终使用亮色模式（覆盖系统/浏览器暗色偏好）
    document.documentElement.style.colorScheme = "light";
    document.documentElement.removeAttribute("data-theme");
    localStorage.removeItem("pbl_theme");
    watchDarkClass();

    installEnterHandler();
    installKeyboardShortcuts();
    initDeleteIcons();
    initMessageActions();
    watchChatVisibility();

    // 监听 DOM 变化，持续安装 Enter 处理器（处理动态渲染）
    const observer = new MutationObserver(() => {
      installEnterHandler();
    });
    observer.observe(document.body, { childList: true, subtree: true });

    console.log("[交互] 交互增强模块已初始化");
  }

  // DOM 就绪后初始化
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
