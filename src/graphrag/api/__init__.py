"""
GraphRAG API Module

Flask blueprints for:
- Graph data endpoints
- Chat/Q&A endpoints
"""

from flask import Blueprint

graph_bp = Blueprint('graph', __name__)
chat_bp = Blueprint('chat', __name__)

from . import graph
from . import chat

__all__ = ["graph_bp", "chat_bp"]
