"""LLM-based ontology evolution and candidate generation."""

import json
import logging
from typing import Optional

from neo4j_graphrag.llm import OpenAILLM
from pydantic import BaseModel, ConfigDict

from kg_builder_llm.core.ontology import OntologyCandidate
from kg_builder_llm.ml.modeling import ModelMetrics

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Structured Output Schema (for OpenAI's response_format)
# ============================================================================


class PropertyDefModel(BaseModel):
    """Property definition for nodes."""

    name: str
    type: str
    required: Optional[bool] = False

    model_config = ConfigDict(extra="ignore")


class NodeTypeModel(BaseModel):
    """Node type definition."""

    label: str
    properties: list[PropertyDefModel]

    model_config = ConfigDict(extra="ignore")


class OntologySchemaModel(BaseModel):
    """
    The ontology schema that the LLM must output.
    Matches the structure expected by neo4j_graphrag's SimpleKGPipeline.
    """

    node_types: list[NodeTypeModel]
    relationship_types: list[str]
    patterns: list[list[str]]

    # Flags for graph construction flexibility
    additional_node_types: bool = True
    additional_relationship_types: bool = True
    additional_patterns: bool = True

    model_config = ConfigDict(extra="ignore")


# ============================================================================
# Ontology Evolution Agent
# ============================================================================


class OntologyEvolutionAgent:
    """Evolves ontologies using LLM based on previous performance metrics."""

    def __init__(self, llm: OpenAILLM):
        """Initialize evolution agent.

        Args:
            llm: OpenAI LLM instance configured with API key and model
        """
        self.llm = llm

    async def propose_new_candidate(
        self,
        previous: OntologyCandidate,
        metrics: ModelMetrics,
        step_index: int,
        variant_index: int,
    ) -> OntologyCandidate:
        """Propose evolved ontology candidate based on previous performance.

        Args:
            previous: Previous ontology candidate
            metrics: Performance metrics (AUC, F1) of previous candidate
            step_index: Current step number
            variant_index: Variant number within this step

        Returns:
            New OntologyCandidate with evolved schema
        """
        logger.info(
            "Proposing candidate variant %d for step %d " "(prev AUC=%.4f, F1=%.4f)",
            variant_index,
            step_index,
            metrics.auc,
            metrics.f1,
        )

        # Create prompt for LLM
        prompt = self._create_evolution_prompt(previous, metrics, variant_index)

        # Call LLM to generate new schema
        logger.debug("Calling LLM for ontology evolution...")
        response = await self.llm.ainvoke(prompt)

        # Extract text from LLMResponse object if needed
        if hasattr(response, "content"):
            schema_json = response.content
        elif isinstance(response, str):
            schema_json = response
        else:
            schema_json = str(response)

        # Parse response
        try:
            schema_dict = self._parse_schema_response(schema_json)
            logger.info("Successfully evolved schema for candidate")

            # Validate and fix relationship types
            schema_dict = self._validate_and_fix_relationships(schema_dict)

        except Exception as e:
            logger.error(
                "Failed to parse evolved schema: %s. Response was: %s", e, schema_json[:200]
            )
            logger.warning("Using deterministic variant instead - LLM evolution failed!")
            # Fall back to deterministic variant
            schema_dict = self._create_variant_schema(previous.schema, variant_index)

        # Create new candidate
        new_candidate = OntologyCandidate(
            candidate_tag=f"step_{step_index}_candidate_{variant_index}",
            schema=schema_dict,
            description=f"Evolved from {previous.candidate_tag} (variant {variant_index})",
            parent_tag=previous.candidate_tag,
            step_index=step_index,
        )

        logger.info("Generated candidate: %s", new_candidate.candidate_tag)
        return new_candidate

    async def generate_manual_variants(
        self,
        base: OntologyCandidate,
        num_variants: int,
        step_index: int,
    ) -> list[OntologyCandidate]:
        """Generate multiple manual variants from base ontology.

        For step 0, we may want manual variations instead of evolution.
        This provides deterministic variants without LLM calls.

        Args:
            base: Base ontology to create variants from
            num_variants: Number of variants to generate
            step_index: Current step number

        Returns:
            List of variant candidates
        """
        variants = []

        for i in range(num_variants):
            variant_schema = self._create_variant_schema(base.schema, i)

            variant = OntologyCandidate(
                candidate_tag=f"step_{step_index}_candidate_{i}",
                schema=variant_schema,
                description=f"Manual variant {i} from base ontology",
                parent_tag=base.candidate_tag,
                step_index=step_index,
            )
            variants.append(variant)

        logger.info("Generated %d manual variants for step %d", num_variants, step_index)
        return variants

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _create_evolution_prompt(
        self,
        previous: OntologyCandidate,
        metrics: ModelMetrics,
        variant_index: int,
    ) -> str:
        """Create prompt for LLM to evolve ontology.

        Args:
            previous: Previous ontology candidate
            metrics: Performance metrics
            variant_index: Which variant to create

        Returns:
            Prompt string for LLM
        """
        # Extract existing node types and relationships from schema
        existing_nodes = [n["label"] for n in previous.schema.get("node_types", [])]
        existing_rels = previous.schema.get("relationship_types", [])

        return f"""You are a knowledge graph ontology designer optimizing for financial news analysis.

TASK: Evolve the following ontology by ADDING NEW entity types and relationships not present in the current schema.

Previous ontology:
```json
{json.dumps(previous.schema, indent=2)}
```

EXISTING ENTITY TYPES IN CURRENT SCHEMA:
{', '.join(existing_nodes)}

EXISTING RELATIONSHIP TYPES:
{', '.join(existing_rels)}

Performance metrics:
- AUC: {metrics.auc:.4f}
- F1: {metrics.f1:.4f}
- Max hops in training period: {metrics.max_hops_train}
- Max hops in validation period: {metrics.max_hops_val}

Variant index: {variant_index}

CRITICAL INSTRUCTIONS FOR EVOLUTION:
1. MUST ADD NEW entity types that are NOT in {existing_nodes}
2. Examples of NEW entity types to consider: Market, Sector, Index, Deal, Contract, Regulation, Product, Event, Quarter, Competitor, Insider, Executive, Fund, Portfolio, Sentiment, PriceDecrease, PriceIncrease
3. For property types, ONLY use: STRING, INTEGER, FLOAT, BOOLEAN
4. NEVER use NUMBER - use FLOAT or INTEGER instead
5. MUST ADD NEW relationships involving the new entity types
6. Keep all existing node_types and relationship_types
7. Patterns MUST be formatted as ["NodeA", "RELATIONSHIP", "NodeB"]
8. Return ONLY valid JSON with all required fields

For variant {variant_index}:
- If metrics.auc < 0.55: Focus on financial domain entities (Market, Sector, Index, Deal)
- If 0.55 <= metrics.auc < 0.60: Add analyst/market-related entities (Analyst, Fund, Quarter)
- Otherwise: Add competitive intelligence entities (Competitor, Product, Regulation)

Return ONLY valid JSON, no prose."""

    def _parse_schema_response(self, response: str) -> dict:
        """Parse LLM response to extract schema.

        Args:
            response: LLM response text

        Returns:
            Parsed schema dictionary

        Raises:
            ValueError: If parsing fails
        """
        # Try to parse as JSON directly
        try:
            if isinstance(response, dict):
                return response

            # Handle markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response

            schema = json.loads(json_str)

            # Validate structure
            if not all(k in schema for k in ["node_types", "relationship_types", "patterns"]):
                raise ValueError("Missing required schema keys")

            # Fix invalid property types
            schema = self._fix_property_types(schema)

            return schema

        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from LLM response: %s", e)
            raise ValueError(f"Invalid JSON in response: {e}")

    def _fix_property_types(self, schema: dict) -> dict:
        """Fix invalid property types in schema.

        Converts invalid types like 'NUMBER' to valid types like 'FLOAT'.
        Valid types: STRING, INTEGER, FLOAT, BOOLEAN, DATE, DURATION, etc.

        Args:
            schema: Schema dictionary with node_types and properties

        Returns:
            Fixed schema dictionary
        """
        valid_types = {
            "STRING",
            "INTEGER",
            "FLOAT",
            "BOOLEAN",
            "DATE",
            "DURATION",
            "LOCAL_DATETIME",
            "LOCAL_TIME",
            "ZONED_DATETIME",
            "ZONED_TIME",
            "POINT",
            "LIST",
        }

        type_mappings = {
            "NUMBER": "FLOAT",
            "INT": "INTEGER",
            "BOOL": "BOOLEAN",
            "TEXT": "STRING",
            "TIMESTAMP": "ZONED_DATETIME",
        }

        for node_type in schema.get("node_types", []):
            for prop in node_type.get("properties", []):
                prop_type = prop.get("type", "STRING").upper()

                # Map invalid types to valid ones
                if prop_type in type_mappings:
                    prop["type"] = type_mappings[prop_type]
                    logger.debug("Fixed property type: %s → %s", prop_type, prop["type"])
                elif prop_type not in valid_types:
                    # Default unknown types to STRING
                    prop["type"] = "STRING"
                    logger.debug("Unknown property type %s, defaulting to STRING", prop_type)

        return schema

    def _create_variant_schema(self, base_schema: dict, variant_index: int) -> dict:
        """Create a manual variant from base schema.

        Used for deterministic variant generation without LLM calls.

        Args:
            base_schema: Base schema to vary from
            variant_index: Which variant to create (0, 1, 2, etc.)

        Returns:
            Variant schema
        """
        import copy

        schema = copy.deepcopy(base_schema)

        # Variant 0: Add financial entities (Person, Organization)
        if variant_index == 0:
            schema["node_types"].extend(
                [
                    {
                        "label": "Person",
                        "properties": [
                            {"name": "name", "type": "STRING"},
                            {"name": "role", "type": "STRING"},
                        ],
                    },
                    {
                        "label": "Organization",
                        "properties": [
                            {"name": "name", "type": "STRING"},
                        ],
                    },
                ]
            )
            schema["relationship_types"].extend(["WORKS_FOR", "PARTNERS_WITH"])
            schema["patterns"].extend(
                [
                    ["Person", "WORKS_FOR", "Organization"],
                    ["Organization", "PARTNERS_WITH", "Organization"],
                ]
            )

        # Variant 1: Add event entities (Event, Action)
        elif variant_index == 1:
            schema["node_types"].extend(
                [
                    {
                        "label": "Event",
                        "properties": [
                            {"name": "event_type", "type": "STRING"},
                            {"name": "timestamp", "type": "STRING"},
                        ],
                    },
                    {
                        "label": "Action",
                        "properties": [
                            {"name": "action_type", "type": "STRING"},
                        ],
                    },
                ]
            )
            schema["relationship_types"].extend(["TRIGGERS", "CAUSES"])
            schema["patterns"].extend(
                [
                    ["Event", "TRIGGERS", "Action"],
                    ["Action", "CAUSES", "Event"],
                ]
            )

        # Variant 2: Add sentiment entities (Sentiment, Emotion)
        elif variant_index == 2:
            schema["node_types"].extend(
                [
                    {
                        "label": "Sentiment",
                        "properties": [
                            {"name": "polarity", "type": "STRING"},
                            {"name": "score", "type": "FLOAT"},
                        ],
                    },
                    {
                        "label": "Emotion",
                        "properties": [
                            {"name": "type", "type": "STRING"},
                            {"name": "intensity", "type": "FLOAT"},
                        ],
                    },
                ]
            )
            schema["relationship_types"].extend(["EXPRESSES", "ASSOCIATED_WITH"])
            schema["patterns"].extend(
                [
                    ["Article", "EXPRESSES", "Sentiment"],
                    ["Sentiment", "ASSOCIATED_WITH", "Emotion"],
                ]
            )

        return schema

    def _validate_and_fix_relationships(self, schema: dict) -> dict:
        """Validate and fix relationship types in patterns.

        Ensures all relationship types used in patterns are defined in relationship_types.
        Fixes common typos like RELATED_TO -> RELATES_TO.

        Args:
            schema: Schema dictionary

        Returns:
            Fixed schema dictionary
        """
        rel_types = set(schema.get("relationship_types", []))
        patterns = schema.get("patterns", [])

        # Common typo mappings
        typo_fixes = {
            "RELATED_TO": "RELATES_TO",
            "RELATED": "RELATES_TO",
            "REFERENCES": "MENTIONS",
            "AUTHORED_BY": "WRITTEN_BY",
            "PUBLISHED": "PUBLISHED_ON",
            "POSTS": "WRITTEN_BY",
            "CREATED_BY": "WRITTEN_BY",
        }

        # Fix patterns and collect all used relationship types
        fixed_patterns = []
        for pattern in patterns:
            if len(pattern) == 3:
                node_a, rel, node_b = pattern

                # Fix typo if it exists
                if rel in typo_fixes:
                    rel = typo_fixes[rel]

                fixed_patterns.append([node_a, rel, node_b])
                rel_types.add(rel)

        # Update schema
        schema["patterns"] = fixed_patterns
        schema["relationship_types"] = list(rel_types)

        return schema
