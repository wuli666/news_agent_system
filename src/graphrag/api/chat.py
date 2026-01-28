"""
Chat API Endpoints

Provides REST API for:
- Q&A over knowledge graph
- Chat history management
- Streaming responses
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Generator
from flask import request, jsonify, Response, stream_with_context

from . import chat_bp
from ..agents.retriever import RetrieverAgent
from ..agents.reasoner import ReasonerAgent
from ..config import get_config

logger = logging.getLogger(__name__)

# Simple in-memory chat history (would use database in production)
_chat_history: Dict[str, List[Dict[str, Any]]] = {}


def run_async(coro):
    """Helper to run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@chat_bp.route('', methods=['POST'])
def chat():
    """
    Send a question and get an answer.

    Request body:
    - question: User question
    - session_id: Optional session ID for history
    - graph_id: Optional target graph ID
    - mode: Search mode ('quick', 'panorama', 'insight')

    Returns:
    - answer: Generated answer
    - sources: List of source snippets
    - insights: List of analytical insights
    """
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        session_id = data.get('session_id', 'default')
        graph_id = data.get('graph_id')
        mode = data.get('mode', 'quick')

        if not question:
            return jsonify({
                "success": False,
                "error": "Question is required"
            }), 400

        # Initialize session history if needed
        if session_id not in _chat_history:
            _chat_history[session_id] = []

        # Add user message to history
        _chat_history[session_id].append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat()
        })

        # Use default graph if not specified
        from ..services.zep_client import ZepGraphClient
        zep_client = ZepGraphClient()
        if not graph_id:
            graph_id = zep_client.get_or_create_graph("news_graph")

        # Retrieve relevant information
        retriever = RetrieverAgent(zep_client=zep_client)
        retrieval = run_async(retriever.search(question, mode, graph_id))

        # Generate answer
        reasoner = ReasonerAgent()
        result = run_async(reasoner.reason(question, retrieval))

        # Add assistant response to history
        _chat_history[session_id].append({
            "role": "assistant",
            "content": result.answer,
            "timestamp": datetime.now().isoformat(),
            "confidence": result.confidence,
            "sources": result.sources[:5],
            "insights": result.insights,
        })

        return jsonify({
            "success": True,
            "data": {
                "answer": result.answer,
                "confidence": result.confidence,
                "sources": result.sources,
                "insights": result.insights,
                "retrieval_mode": retrieval.mode,
                "result_count": len(retrieval.results),
            }
        })

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@chat_bp.route('/stream', methods=['POST'])
def chat_stream():
    """
    Stream a response using Server-Sent Events with Agentic RAG.

    Request body:
    - question: User question
    - mode: 'quick' (简单RAG) or 'agentic' (智能Agent)

    Returns:
    - SSE stream with status updates and answer chunks
    """
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        mode = data.get('mode', 'agentic')

        if not question:
            return jsonify({"success": False, "error": "Question is required"}), 400

        def generate():
            try:
                yield f"data: {json.dumps({'type': 'start'})}\n\n"

                if mode == 'quick':
                    # 简单模式：直接检索 + 生成
                    from ..services.zep_client import ZepGraphClient
                    zep_client = ZepGraphClient()
                    graph_id = zep_client.get_or_create_graph("news_graph")

                    retriever = RetrieverAgent(zep_client=zep_client)
                    retrieval = run_async(retriever.search(question, 'quick', graph_id))

                    yield f"data: {json.dumps({'type': 'status', 'message': f'检索到 {len(retrieval.results)} 条信息'})}\n\n"

                    reasoner = ReasonerAgent()
                    if not retrieval.results:
                        yield f"data: {json.dumps({'type': 'chunk', 'content': '知识库中暂无相关信息。'})}\n\n"
                        yield f"data: {json.dumps({'type': 'done', 'data': {}})}\n\n"
                        return

                    context = reasoner._format_context(retrieval)
                    from langchain_core.messages import HumanMessage
                    from ..agents.reasoner import REASONING_SYSTEM_PROMPT

                    prompt = f"## 问题\n{question}\n\n## 信息\n{context}\n\n请回答："
                    messages = [HumanMessage(content=REASONING_SYSTEM_PROMPT + "\n\n" + prompt)]

                    for chunk in reasoner.llm.stream(messages):
                        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        if content:
                            yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"

                    yield f"data: {json.dumps({'type': 'done', 'data': {'steps': 1}})}\n\n"

                else:
                    # Agentic 模式：规划 + 多工具 + 生成
                    from ..agents.agentic_rag import AgenticRAG
                    agent = AgenticRAG()

                    for event in agent.stream(question):
                        yield f"data: {json.dumps(event)}\n\n"

            except Exception as e:
                logger.error(f"Stream error: {e}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            }
        )

    except Exception as e:
        logger.error(f"Stream setup failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@chat_bp.route('/history', methods=['GET'])
def get_chat_history():
    """
    Get chat history for a session.

    Query params:
    - session_id: Session identifier (default: 'default')
    - limit: Max messages (default: 50)

    Returns:
    - messages: List of chat messages
    """
    try:
        session_id = request.args.get('session_id', 'default')
        limit = int(request.args.get('limit', 50))

        history = _chat_history.get(session_id, [])

        return jsonify({
            "success": True,
            "data": {
                "session_id": session_id,
                "messages": history[-limit:],
                "total": len(history)
            }
        })

    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@chat_bp.route('/history', methods=['DELETE'])
def clear_chat_history():
    """
    Clear chat history for a session.

    Query params:
    - session_id: Session identifier (default: 'default')

    Returns:
    - success status
    """
    try:
        session_id = request.args.get('session_id', 'default')

        if session_id in _chat_history:
            del _chat_history[session_id]

        return jsonify({
            "success": True,
            "message": f"History cleared for session {session_id}"
        })

    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@chat_bp.route('/sessions', methods=['GET'])
def list_sessions():
    """
    List all chat sessions.

    Returns:
    - sessions: List of session IDs with message counts
    """
    try:
        sessions = [
            {
                "session_id": sid,
                "message_count": len(messages),
                "last_message": messages[-1]["timestamp"] if messages else None
            }
            for sid, messages in _chat_history.items()
        ]

        return jsonify({
            "success": True,
            "data": {
                "sessions": sessions,
                "total": len(sessions)
            }
        })

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@chat_bp.route('/suggest', methods=['GET'])
def suggest_questions():
    """
    Get suggested questions based on graph content.

    Query params:
    - graph_id: Optional target graph ID

    Returns:
    - suggestions: List of suggested questions
    """
    try:
        # Static suggestions for now
        # In production, could analyze graph content
        suggestions = [
            "最近有哪些关于AI的热点新闻？",
            "今天的科技新闻主要关注什么话题？",
            "有哪些公司被频繁提及？",
            "最近发生了哪些重要事件？",
            "不同新闻平台关注的热点有什么不同？",
        ]

        return jsonify({
            "success": True,
            "data": {
                "suggestions": suggestions
            }
        })

    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
