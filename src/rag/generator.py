"""RAG 生成器 — 流式/非流式 LLM 调用"""
import json, logging
from typing import AsyncGenerator, List
from langchain_core.messages import HumanMessage
from src.rag.prompts import SYSTEM_PROMPT, build_rag_prompt, format_context_with_labels, format_sources
from src.rag.retriever import RetrievalResult

logger = logging.getLogger("course_assistant.generator")

class RAGGenerator:
    def __init__(self, llm, session_mgr):
        self.llm = llm; self.session_mgr = session_mgr

    async def stream_rag(self, query, retrieval_results, session_id, history=None, user_name="", web_context="", user_id=None, memory_summary="", followup_recommender=None) -> AsyncGenerator[str, None]:
        if history is None: history = self.session_mgr.get_history(session_id)
        context = format_context_with_labels(retrieval_results) if retrieval_results else ""
        # 低置信度 RAG 时注入网络搜索结果
        if web_context:
            context = context + "\n\n" + web_context if context else web_context
        sources_text = format_sources(retrieval_results) if retrieval_results else ""
        sources_list = sources_text.split("\n") if sources_text else []
        history_text = ""
        for m in history[-6:]: history_text += f"{'用户' if m['role']=='user' else '小课'}：{m['content']}\n"
        full_prompt = build_rag_prompt(SYSTEM_PROMPT, query, context, history_text, user_name or "未知", sources_text, memory_summary)
        full_answer = ""
        try:
            async for chunk in self.llm.astream([HumanMessage(content=full_prompt)]):
                token = chunk.content
                if token: full_answer += token; yield f"data: {json.dumps({'token':token,'done':False},ensure_ascii=False)}\n\n"
            self.session_mgr.append_history(session_id, "user", query, user_id)
            self.session_mgr.append_history(session_id, "assistant", full_answer, user_id)
            # 构建 done 事件，附带后续问题推荐
            done_event = {"token": "", "done": True, "source": "rag", "docs": sources_list}
            if followup_recommender and full_answer and len(full_answer) > 20:
                done_event["followup"] = followup_recommender.recommend(
                    query, full_answer, "course_question"
                )
            yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("RAG 流式失败")
            yield f"data: {json.dumps({'token':'⚠️ 服务暂不可用','done':True,'source':'error'},ensure_ascii=False)}\n\n"

    async def stream_casual(self, query, session_id, history=None, user_name="", user_id=None, memory_summary="") -> AsyncGenerator[str, None]:
        if history is None: history = self.session_mgr.get_history(session_id)
        messages = self.session_mgr.build_messages(history, query, SYSTEM_PROMPT, user_name, memory_summary)
        full_answer = ""
        try:
            async for chunk in self.llm.astream(messages):
                token = chunk.content
                if token: full_answer += token; yield f"data: {json.dumps({'token':token,'done':False},ensure_ascii=False)}\n\n"
            self.session_mgr.append_history(session_id, "user", query, user_id)
            self.session_mgr.append_history(session_id, "assistant", full_answer, user_id)
            yield f"data: {json.dumps({'token':'','done':True,'source':'casual','docs':[]},ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("闲聊流式失败")
            yield f"data: {json.dumps({'token':'⚠️ 服务暂不可用','done':True,'source':'error'},ensure_ascii=False)}\n\n"

    async def generate_rag(self, query, retrieval_results, session_id, history=None, user_name="", web_context="", user_id=None, memory_summary="") -> dict:
        import asyncio
        if history is None: history = self.session_mgr.get_history(session_id)
        context = format_context_with_labels(retrieval_results) if retrieval_results else ""
        if web_context:
            context = context + "\n\n" + web_context if context else web_context
        sources_text = format_sources(retrieval_results) if retrieval_results else ""
        sources_list = sources_text.split("\n") if sources_text else []
        history_text = ""
        for m in history[-6:]: history_text += f"{'用户' if m['role']=='user' else '小课'}：{m['content']}\n"
        prompt = build_rag_prompt(SYSTEM_PROMPT, query, context, history_text, user_name or "未知", sources_text, memory_summary)
        resp = await asyncio.to_thread(self.llm.invoke, [HumanMessage(content=prompt)])
        answer = resp.content
        self.session_mgr.append_history(session_id, "user", query, user_id)
        self.session_mgr.append_history(session_id, "assistant", answer, user_id)
        return {"answer": answer, "source": "rag", "docs": sources_list}
