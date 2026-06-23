# AI 课程助手 (AI Engineer Mentor)

AIGC 大模型应用工程师课程咨询助手 v0.7.0

## 技术栈

- **后端**: FastAPI + DeepSeek LLM + ChromaDB (RAG)
- **前端**: Gradio 6.x + SSE 流式对话
- **搜索引擎**: 博查 AI Search（主）+ Bing 中国版（回退）
- **嵌入模型**: instructor-xl (M1 MPS 加速)
- **认证**: bcrypt + JWT (HS256)
- **知识库**: AIGC 课程知识库 60 片段

## 项目结构

```
├── src/
│   ├── main.py            # FastAPI 入口
│   ├── auth/              # JWT 认证
│   ├── intent/            # LLM 意图分类 + k-NN 回退
│   ├── rag/               # 混合检索 (ChromaDB + BM25 + RRF)
│   ├── parser/            # 8 格式文件解析
│   ├── search/            # 博查 AI Search + Bing 回退
│   └── session/           # SQLite 会话管理
├── course_assistant_ui.py  # Gradio 前端
├── build_knowledge_base.py # 知识库构建
├── chroma_db/              # 向量数据库
├── tests/                  # 42 个单元测试
└── requirements.txt
```

## 启动方式

```bash
cd /path/to/pbl && source pbl_venv/bin/activate
python src/main.py          # 后端
python course_assistant_ui.py  # 前端
```

## 核心特性

- **联网搜索**: 博查 AI Search + 流式输出 + LLM 查询改写
- **意图分类**: LLM 直接分类 + 对话历史上下文 + k-NN 回退
- **混合检索**: 稠密(ChromaDB) + 稀疏(BM25+jiaba) → RRF 融合
- **JWT 认证**: bcrypt 密码哈希 + 会话用户隔离
- **SSE 流式**: httpx 流式接收后端事件，3s 首 token
- **并行加速**: 意图分类和搜索请求同时执行
