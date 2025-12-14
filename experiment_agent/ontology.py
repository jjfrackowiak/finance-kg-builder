# file: finance_kg_experiment/ontology.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import copy

@dataclass
class OntologyCandidate:
    name: str
    schema: Dict[str, Any]
    description: str
    parent_name: Optional[str] = None
    step_index: int = 0


def base_ontology() -> OntologyCandidate:
    """
    Very simple ontology: Company, Article, Day with basic relationships.
    """
    schema = {
        "node_types": [
            {"label": "Company", "properties": [
                {"name": "symbol", "type": "STRING", "required": True}
            ]},
            {"label": "Article", "properties": [
                {"name": "headline", "type": "STRING"},
                {"name": "timestamp", "type": "STRING"},
                {"name": "url", "type": "STRING"},
                {"name": "source", "type": "STRING"},
            ]},
            {"label": "Day", "properties": [
                {"name": "date", "type": "STRING", "required": True}
            ]},
        ],
        "relationship_types": ["MENTIONS", "RELATES_TO", "INVOLVES", "PUBLISHED_ON"],
        "patterns": [
            ("Article", "MENTIONS", "Company"),
            ("Article", "PUBLISHED_ON", "Day"),
        ],
        "additional_node_types": True,
        "additional_relationship_types": True,
        "additional_patterns": True,
    }
    return OntologyCandidate(
        name="base_v1",
        schema=schema,
        description="Base ontology with Company, Article, Day and MENTIONS/PUBLISHED_ON.",
        parent_name=None,
        step_index=0,
    )


def augment_ontology(
    previous: OntologyCandidate,
    variant_index: int,
    llm_client=None,
) -> OntologyCandidate:
    """
    Modify the augmentation process to avoid duplicating the schema and instead focus on attaching new entities
    and relationships to the existing graph.
    """
    new_schema = previous.schema  # Use the existing schema directly without copying

    # Variant 0 — Expanded Financial Entities Ontology
    if variant_index == 0:
        # Attach new financial entities
        new_schema["node_types"].extend([
            {"label": "Person", "properties": [
                {"name": "name", "type": "STRING"},
                {"name": "role", "type": "STRING"},
            ]},
            {"label": "Organization", "properties": [
                {"name": "name", "type": "STRING"},
            ]},
        ])

        new_schema["relationship_types"].extend([
            "WORKS_FOR",
            "PARTNERS_WITH",
        ])

        new_schema["patterns"].extend([
            ("Person", "WORKS_FOR", "Organization"),
            ("Organization", "PARTNERS_WITH", "Organization"),
        ])

        name_suffix = "expanded_financial_entities"
        desc = "Adds Person, Organization nodes with financial relations."

    # Variant 1 — Expanded Event & Semantic Roles Ontology
    elif variant_index == 1:
        # Attach new event-based entities
        new_schema["node_types"].extend([
            {"label": "Event", "properties": [
                {"name": "event_type", "type": "STRING"},
                {"name": "timestamp", "type": "STRING"},
            ]},
            {"label": "Action", "properties": [
                {"name": "action_type", "type": "STRING"},
                {"name": "description", "type": "STRING"},
            ]},
        ])

        new_schema["relationship_types"].extend([
            "TRIGGERS",
            "CAUSES",
        ])

        new_schema["patterns"].extend([
            ("Event", "TRIGGERS", "Action"),
            ("Action", "CAUSES", "Event"),
        ])

        name_suffix = "expanded_semantic_events"
        desc = "Adds Event and Action nodes for event-based semantic extraction."

    # Variant 2 — Sentiment and Emotion Ontology
    elif variant_index == 2:
        # Attach sentiment-related entities
        new_schema["node_types"].extend([
            {"label": "Sentiment", "properties": [
                {"name": "polarity", "type": "STRING"},
                {"name": "score", "type": "FLOAT"},
            ]},
            {"label": "Emotion", "properties": [
                {"name": "type", "type": "STRING"},
                {"name": "intensity", "type": "FLOAT"},
            ]},
        ])

        new_schema["relationship_types"].extend([
            "EXPRESSES",
            "ASSOCIATED_WITH",
        ])

        new_schema["patterns"].extend([
            ("Article", "EXPRESSES", "Sentiment"),
            ("Sentiment", "ASSOCIATED_WITH", "Emotion"),
        ])

        name_suffix = "sentiment_emotion"
        desc = "Adds Sentiment and Emotion nodes with relationships to Articles."

    candidate = OntologyCandidate(
        name=f"{previous.name}_variant{variant_index}",
        schema=new_schema,
        description=f"Augmented ontology variant {variant_index}: {desc}",
        parent_name=previous.name,
        step_index=previous.step_index + 1,
    )
    return candidate