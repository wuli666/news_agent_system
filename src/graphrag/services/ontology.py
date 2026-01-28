"""
Ontology Manager for News Knowledge Graph

Manages entity types and relationship types for the news domain.
"""

from typing import Dict, List, Any, Optional
from ..config import NEWS_ONTOLOGY


class OntologyManager:
    """
    Manages the ontology (schema) for the news knowledge graph.

    Provides methods to:
    - Get entity and edge type definitions
    - Validate entities against the ontology
    - Generate prompts for entity extraction
    """

    def __init__(self, ontology: Optional[Dict[str, Any]] = None):
        """
        Initialize with custom or default ontology.

        Args:
            ontology: Custom ontology definition, uses NEWS_ONTOLOGY if not provided
        """
        self.ontology = ontology or NEWS_ONTOLOGY

    @property
    def entity_types(self) -> List[Dict[str, Any]]:
        """Get list of entity type definitions."""
        return self.ontology.get("entity_types", [])

    @property
    def edge_types(self) -> List[Dict[str, Any]]:
        """Get list of edge type definitions."""
        return self.ontology.get("edge_types", [])

    def get_entity_type_names(self) -> List[str]:
        """Get list of entity type names."""
        return [e["name"] for e in self.entity_types]

    def get_edge_type_names(self) -> List[str]:
        """Get list of edge type names."""
        return [e["name"] for e in self.edge_types]

    def get_entity_type(self, name: str) -> Optional[Dict[str, Any]]:
        """Get entity type definition by name."""
        for e in self.entity_types:
            if e["name"] == name:
                return e
        return None

    def get_edge_type(self, name: str) -> Optional[Dict[str, Any]]:
        """Get edge type definition by name."""
        for e in self.edge_types:
            if e["name"] == name:
                return e
        return None

    def is_valid_entity_type(self, type_name: str) -> bool:
        """Check if entity type is valid."""
        return type_name in self.get_entity_type_names()

    def is_valid_edge_type(self, type_name: str) -> bool:
        """Check if edge type is valid."""
        return type_name in self.get_edge_type_names()

    def get_extraction_prompt(self) -> str:
        """
        Generate a prompt describing the ontology for LLM extraction.

        Returns:
            Formatted prompt string
        """
        lines = ["## 实体类型"]

        for et in self.entity_types:
            lines.append(f"- **{et['name']}**: {et.get('description', '')}")
            if et.get("attributes"):
                attrs = ", ".join(a["name"] for a in et["attributes"])
                lines.append(f"  属性: {attrs}")

        lines.append("\n## 关系类型")

        for rt in self.edge_types:
            lines.append(f"- **{rt['name']}**: {rt.get('description', '')}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ontology to dictionary."""
        return self.ontology.copy()


# Default ontology manager instance
default_ontology = OntologyManager()
