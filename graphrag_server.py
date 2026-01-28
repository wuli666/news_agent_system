"""
GraphRAG Flask Server

Entry point for the GraphRAG API service.
Runs on port 5002 by default.
"""

import os
import sys
import logging
from datetime import datetime

from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure Flask application."""

    app = Flask(__name__)

    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'graphrag-secret-key')
    app.config['JSON_AS_ASCII'] = False  # Support Chinese in JSON
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Request/Response logging middleware
    @app.before_request
    def log_request():
        from flask import request
        logger.debug(f"Request: {request.method} {request.path}")

    @app.after_request
    def log_response(response):
        from flask import request
        logger.debug(f"Response: {request.method} {request.path} -> {response.status_code}")
        return response

    # Register blueprints
    from src.graphrag.api import graph_bp, chat_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(chat_bp, url_prefix='/api/chat')

    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": "graphrag",
            "timestamp": datetime.now().isoformat()
        })

    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        return jsonify({
            "service": "GraphRAG API",
            "version": "1.0.0",
            "endpoints": {
                "graph": {
                    "GET /api/graph/data": "Get graph nodes and edges",
                    "GET /api/graph/entity/<id>": "Get entity details",
                    "GET /api/graph/timeline": "Get timeline data",
                    "GET /api/graph/search?q=": "Search graph",
                    "GET /api/graph/stats": "Get graph statistics",
                    "POST /api/graph/ingest": "Ingest news items",
                    "GET /api/graph/task/<id>": "Get task status",
                },
                "chat": {
                    "POST /api/chat": "Ask a question",
                    "GET /api/chat/history": "Get chat history",
                    "DELETE /api/chat/history": "Clear history",
                    "GET /api/chat/sessions": "List sessions",
                    "GET /api/chat/suggest": "Get suggested questions",
                },
                "health": {
                    "GET /health": "Health check",
                }
            }
        })

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "Not found"
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

    return app


def validate_config():
    """Validate required configuration."""
    from src.graphrag.config import get_config

    config = get_config()
    errors = config.validate()

    if errors:
        logger.warning("Configuration warnings:")
        for error in errors:
            logger.warning(f"  - {error}")

    return len(errors) == 0


def main():
    """Main entry point."""
    # Validate configuration
    if not validate_config():
        logger.warning("Some configuration is missing, GraphRAG may not work properly")

    # Create app
    app = create_app()

    # Get port from environment or use default
    port = int(os.environ.get('GRAPHRAG_PORT', 5002))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting GraphRAG server on port {port}")
    logger.info(f"Debug mode: {debug}")

    # Run server
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
