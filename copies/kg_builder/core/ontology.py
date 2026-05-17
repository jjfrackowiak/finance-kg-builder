"""Ontology models and management."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class NodeType:
    """Ontology node type definition."""

    label: str
    properties: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class RelationshipType:
    """Ontology relationship type definition."""

    label: str
    properties: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class OntologyCandidate:
    """Ontology candidate for evaluation and evolution.

    This represents a complete ontology schema with node types, relationships,
    and patterns. It can be used to build a knowledge graph via SimpleKGPipeline.
    """

    candidate_tag: str
    schema: Dict[str, Any]
    description: str = "Ontology candidate"
    parent_tag: Optional[str] = None
    step_index: int = 0


def create_base_ontology() -> OntologyCandidate:
    """Create the base ontology candidate.

    Returns:
        OntologyCandidate with schema for SimpleKGPipeline
    """
    logger.info("Creating base ontology")

    schema = {
        "node_types": [
            {
                "label": "Company",
                "properties": [{"name": "symbol", "type": "STRING", "required": True}],
            },
            {
                "label": "Article",
                "properties": [
                    {"name": "headline", "type": "STRING"},
                    {"name": "timestamp", "type": "STRING"},
                    {"name": "url", "type": "STRING"},
                    {"name": "source", "type": "STRING"},
                ],
            },
            {"label": "Day", "properties": [{"name": "date", "type": "STRING", "required": True}]},
            {
                "label": "Author",
                "properties": [
                    {"name": "name", "type": "STRING"},
                    {"name": "email", "type": "STRING"},
                ],
            },
        ],
        "relationship_types": ["MENTIONS", "RELATES_TO", "WRITTEN_BY", "PUBLISHED_ON"],
        "patterns": [
            ["Article", "MENTIONS", "Company"],
            ["Article", "WRITTEN_BY", "Author"],
            ["Article", "PUBLISHED_ON", "Day"],
        ],
        "additional_node_types": True,
        "additional_relationship_types": True,
        "additional_patterns": True,
    }

    return OntologyCandidate(
        candidate_tag="base",
        schema=schema,
        description="Base ontology with Company, Article, Author, Day nodes",
        step_index=0,
    )
