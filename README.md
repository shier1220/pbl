# AI 课程助手 (AI Engineer Mentor)

AIGC 大模型应用工程师课程咨询助手，阶段一交付物。

## 技术栈

- **后端**: FastAPI + DeepSeek LLM + ChromaDB (RAG)
- **前端**: Gradio 6.x + SSE 流式对话
- **嵌入模型**: instructor-xl
- **知识库**: AIGC 课程知识库 7章 35片段

## 项目结构

```
├── course_assistant_api.py    # FastAPI 后端（意图路由 + RAG + LLM）
├── course_assistant_ui.py     # Gradio 前端（SSE 流式 + 文件上传）
├── embedding.py               # 嵌入模型统一管理
├── build_knowledge_base.py    # 知识库构建脚本
├── aigc_knowledge_base.txt    # 知识库源文件
├── chroma_db/                 # ChromaDB 向量数据库
├── 技术方案确认.md             # 技术方案文档
├── 课程助手全景图.md           # 项目全景图
├── 课程助手系统架构图.svg       # 系统架构图
└── 课程助手全景图.xmind         # XMind 思维导图
```

## 启动方式

```bash
# 1. 启动后端
cd /path/to/pbl && source pbl_venv/bin/activate
python course_assistant_api.py

# 2. 新终端启动前端
python course_assistant_ui.py
```

## 核心特性

- **EmbeddingIntentRouter**: 一次嵌入同时完成意图路由和知识库检索
- **SSE 流式对话**: httpx 流式接收后端事件
- **文件上传**: 支持 txt/pdf/docx 文件解析并注入对话上下文
- **会话管理**: JSON 持久化，支持新建/切换/清除会话
