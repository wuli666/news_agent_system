"""
GraphRAG Configuration Module

Manages Zep Cloud API settings, chunking parameters, and ontology definitions.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class GraphRAGConfig:
    """GraphRAG configuration settings."""

    # Zep Cloud Settings
    ZEP_API_KEY: str = field(default_factory=lambda: os.getenv("ZEP_API_KEY", ""))

    # Graph Settings
    GRAPH_NAME: str = field(default_factory=lambda: os.getenv("GRAPHRAG_GRAPH_NAME", "news_knowledge_graph"))

    # Chunking Settings (for Chonkie)
    CHUNK_SIZE: int = field(default_factory=lambda: int(os.getenv("GRAPHRAG_CHUNK_SIZE", "512")))
    CHUNK_OVERLAP: int = field(default_factory=lambda: int(os.getenv("GRAPHRAG_CHUNK_OVERLAP", "50")))

    # Processing Settings
    BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("GRAPHRAG_BATCH_SIZE", "5")))
    MAX_RETRIES: int = field(default_factory=lambda: int(os.getenv("GRAPHRAG_MAX_RETRIES", "3")))
    RETRY_DELAY: float = field(default_factory=lambda: float(os.getenv("GRAPHRAG_RETRY_DELAY", "1.0")))

    # Feature Flags
    ENABLED: bool = field(default_factory=lambda: os.getenv("GRAPHRAG_ENABLED", "true").lower() == "true")
    AUTO_INGEST: bool = field(default_factory=lambda: os.getenv("GRAPHRAG_AUTO_INGEST", "true").lower() == "true")

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.ZEP_API_KEY:
            errors.append("ZEP_API_KEY is not configured")
        return errors

    @classmethod
    def from_env(cls) -> "GraphRAGConfig":
        """Create config from environment variables."""
        return cls()


# News Entity Ontology Definition
NEWS_ONTOLOGY: Dict[str, Any] = {
    "entity_types": [
        {
            "name": "Person",
            "description": "Individual person mentioned in news (politicians, celebrities, executives, etc.)",
            "attributes": [
                {"name": "role", "type": "text", "description": "Person's role or title"},
                {"name": "organization", "type": "text", "description": "Associated organization"},
            ]
        },
        {
            "name": "Organization",
            "description": "Company, institution, government agency, or other organization",
            "attributes": [
                {"name": "type", "type": "text", "description": "Organization type (company, government, NGO, etc.)"},
                {"name": "industry", "type": "text", "description": "Industry sector"},
            ]
        },
        {
            "name": "Location",
            "description": "Geographic location (country, city, region)",
            "attributes": [
                {"name": "type", "type": "text", "description": "Location type (country, city, region)"},
            ]
        },
        {
            "name": "Event",
            "description": "Specific event or incident mentioned in news",
            "attributes": [
                {"name": "date", "type": "text", "description": "Event date if known"},
                {"name": "type", "type": "text", "description": "Event type (conference, accident, announcement, etc.)"},
            ]
        },
        {
            "name": "Topic",
            "description": "News topic or theme (AI, economy, politics, etc.)",
            "attributes": [
                {"name": "category", "type": "text", "description": "Topic category"},
            ]
        },
        {
            "name": "Product",
            "description": "Product, service, or technology mentioned",
            "attributes": [
                {"name": "company", "type": "text", "description": "Company that produces this product"},
                {"name": "type", "type": "text", "description": "Product type"},
            ]
        },
        {
            "name": "News",
            "description": "A news article or report",
            "attributes": [
                {"name": "source", "type": "text", "description": "News source platform"},
                {"name": "published_at", "type": "text", "description": "Publication date"},
                {"name": "hot_score", "type": "text", "description": "Popularity/trending score"},
            ]
        },
    ],
    "edge_types": [
        {
            "name": "MENTIONED_IN",
            "description": "Entity is mentioned in a news article",
            "source_types": ["Person", "Organization", "Location", "Event", "Topic", "Product"],
            "target_types": ["News"],
        },
        {
            "name": "RELATED_TO",
            "description": "General relationship between entities",
            "source_types": ["Person", "Organization", "Location", "Event", "Topic", "Product"],
            "target_types": ["Person", "Organization", "Location", "Event", "Topic", "Product"],
        },
        {
            "name": "WORKS_FOR",
            "description": "Person works for an organization",
            "source_types": ["Person"],
            "target_types": ["Organization"],
        },
        {
            "name": "LOCATED_IN",
            "description": "Entity is located in a place",
            "source_types": ["Person", "Organization", "Event"],
            "target_types": ["Location"],
        },
        {
            "name": "CAUSED_BY",
            "description": "Event caused by another event or entity",
            "source_types": ["Event"],
            "target_types": ["Event", "Person", "Organization"],
        },
        {
            "name": "PRODUCES",
            "description": "Organization produces a product",
            "source_types": ["Organization"],
            "target_types": ["Product"],
        },
        {
            "name": "COMPETES_WITH",
            "description": "Organization competes with another",
            "source_types": ["Organization"],
            "target_types": ["Organization"],
        },
        {
            "name": "PART_OF",
            "description": "Entity is part of another entity",
            "source_types": ["Person", "Organization", "Location"],
            "target_types": ["Organization", "Location", "Event"],
        },
    ]
}


# Singleton config instance
_config: Optional[GraphRAGConfig] = None


def get_config() -> GraphRAGConfig:
    """Get or create the singleton config instance."""
    global _config
    if _config is None:
        _config = GraphRAGConfig.from_env()
    return _config
