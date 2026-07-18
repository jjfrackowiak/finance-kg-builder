"""LLM-based ontology evolution and candidate generation."""

import json
import logging
from pathlib import Path
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

    def __init__(
        self,
        llm: OpenAILLM,
        prompt_template_path: str = "default",
        single_addition: bool = False,
    ):
        """Initialize evolution agent.

        Args:
            llm: OpenAI LLM instance configured with API key and model
            prompt_template_path: Path to custom prompt template file or "default"
            single_addition: Restrict each candidate to exactly one new node type
                and one new relationship type vs its parent (atomic evolution)
        """
        self.llm = llm
        self.prompt_template_path = prompt_template_path
        self.single_addition = single_addition
        self.custom_prompt_template = None

        # Load custom prompt template if specified
        if prompt_template_path != "default":
            try:
                template_path = Path(prompt_template_path)
                if template_path.exists():
                    self.custom_prompt_template = template_path.read_text()
                    logger.info("Loaded custom prompt template from %s", prompt_template_path)
                else:
                    logger.warning(
                        "Custom prompt template not found: %s, using default", prompt_template_path
                    )
            except Exception as e:
                logger.warning("Failed to load custom prompt template: %s, using default", e)

    async def propose_new_candidate(
        self,
        previous: OntologyCandidate,
        metrics: ModelMetrics,
        step_index: int,
        variant_index: int,
        addition_history: Optional[list] = None,
    ) -> OntologyCandidate:
        """Propose evolved ontology candidate based on previous performance.

        Args:
            previous: Previous ontology candidate
            metrics: Performance metrics (AUC, F1) of previous candidate
            step_index: Current step number
            variant_index: Variant number within this step
            addition_history: In single-addition mode, prior additions with their
                ΔAUC outcomes (fed back into the prompt to avoid repeats)

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
        if self.single_addition:
            prompt = self._create_single_addition_prompt(
                previous, metrics, step_index, variant_index, addition_history or []
            )
        else:
            prompt = self._create_evolution_prompt(previous, metrics, step_index, variant_index)

        # In single-addition mode retry when the LLM repeats an already-tried
        # combination or proposes nothing new; the prompt alone does not
        # reliably prevent repeats at low temperature.
        tried_combos = {
            (entry.get("node"), entry.get("relationship")) for entry in (addition_history or [])
        }
        max_attempts = 3 if self.single_addition else 1

        addition = None
        schema_dict = previous.schema
        for attempt in range(max_attempts):
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

            if not self.single_addition:
                break

            schema_dict, addition = self.enforce_single_addition(schema_dict, previous.schema)
            combo = (addition.get("node"), addition.get("relationship"))
            if combo != (None, None) and combo not in tried_combos:
                break
            logger.warning(
                "Single-addition attempt %d/%d proposed %s — already tried or empty, retrying",
                attempt + 1,
                max_attempts,
                combo,
            )
            prompt += (
                f"\n\nIMPORTANT: Your previous answer proposed "
                f"{combo[0]} + {combo[1]}, which was ALREADY TRIED (see the list above). "
                f"You MUST propose a different node type and relationship type."
            )

        if addition is not None:
            logger.info(
                "Single-addition candidate: node=%s rel=%s (%d patterns)",
                addition.get("node"),
                addition.get("relationship"),
                len(addition.get("patterns", [])),
            )

        # Create new candidate
        new_candidate = OntologyCandidate(
            candidate_tag=f"step_{step_index}_candidate_{variant_index}",
            schema=schema_dict,
            description=f"Evolved from {previous.candidate_tag} (variant {variant_index})",
            parent_tag=previous.candidate_tag,
            step_index=step_index,
            addition=addition,
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

    # Generic relationship types that should be specialised in later steps.
    _GENERIC_RELS = ["RELATES_TO", "INVOLVES", "ASSOCIATED_WITH"]

    # Concrete specialisations the LLM can propose to replace generic rels.
    _REL_SPECIALISATIONS = {
        "RELATES_TO": [
            "COMPETES_WITH",
            "PARTNERS_WITH",
            "SUPPLIES_TO",
            "ACQUIRES",
            "INVESTED_IN",
            "SPUN_OFF_FROM",
        ],
        "INVOLVES": [
            "ACQUIRES",
            "MERGES_WITH",
            "INVESTS_IN",
            "ISSUES",
            "UNDERWRITES",
        ],
        "ASSOCIATED_WITH": [
            "CORRELATED_WITH",
            "BENCHMARKED_AGAINST",
            "HEDGES",
        ],
    }

    @staticmethod
    def _node_label(node) -> str:
        """Extract the label from a node-type entry (dict or plain string)."""
        return node["label"] if isinstance(node, dict) else str(node)

    @classmethod
    def enforce_single_addition(cls, schema: dict, parent_schema: dict) -> tuple[dict, dict]:
        """Reduce an evolved schema to the parent plus at most one node and one rel.

        Keeps the first new node type and first new relationship type the LLM
        proposed, drops the rest, and keeps only patterns that involve the kept
        addition with endpoints/rels known to the resulting schema.

        Args:
            schema: Evolved schema proposed by the LLM (full schema)
            parent_schema: Schema of the parent candidate

        Returns:
            (enforced schema, addition record) where the addition record is
            {"node": label | None, "relationship": rel | None, "patterns": [...]}
        """
        import copy

        parent_labels = [cls._node_label(n) for n in parent_schema.get("node_types", [])]
        parent_rels = list(parent_schema.get("relationship_types", []))

        new_nodes = [
            n for n in schema.get("node_types", []) if cls._node_label(n) not in parent_labels
        ]
        new_rels = [r for r in schema.get("relationship_types", []) if r not in parent_rels]
        if len(new_nodes) > 1 or len(new_rels) > 1:
            logger.warning(
                "Single-addition mode: LLM proposed %d new nodes / %d new rels — "
                "trimming to one of each",
                len(new_nodes),
                len(new_rels),
            )

        kept_node = new_nodes[0] if new_nodes else None
        kept_rel = new_rels[0] if new_rels else None
        kept_label = cls._node_label(kept_node) if kept_node is not None else None

        result = copy.deepcopy(parent_schema)
        allowed_labels = set(parent_labels)
        allowed_rels = set(parent_rels)
        if kept_node is not None:
            result["node_types"] = list(result.get("node_types", [])) + [copy.deepcopy(kept_node)]
            allowed_labels.add(kept_label)
        if kept_rel is not None:
            result["relationship_types"] = list(result.get("relationship_types", [])) + [kept_rel]
            allowed_rels.add(kept_rel)

        seen = {tuple(p) for p in result.get("patterns", []) if len(p) == 3}
        kept_patterns = []
        for pattern in schema.get("patterns", []):
            if len(pattern) != 3:
                continue
            node_a, rel, node_b = pattern
            if node_a not in allowed_labels or node_b not in allowed_labels:
                continue
            if rel not in allowed_rels:
                continue
            involves_addition = kept_label in (node_a, node_b) or rel == kept_rel
            if involves_addition and tuple(pattern) not in seen:
                kept_patterns.append([node_a, rel, node_b])
                seen.add(tuple(pattern))
        result["patterns"] = list(result.get("patterns", [])) + kept_patterns

        addition = {"node": kept_label, "relationship": kept_rel, "patterns": kept_patterns}
        return result, addition

    @staticmethod
    def _format_addition_history(history: list) -> str:
        """Render prior additions and their ΔAUC outcomes for the prompt."""
        if not history:
            return "  (none yet)"
        lines = []
        for entry in history:
            delta = entry.get("delta_auc")
            delta_txt = f"ΔAUC={delta:+.4f}" if delta is not None else "result pending"
            node, rel = entry.get("node"), entry.get("relationship")
            if node and rel:
                combo = f"{node} + {rel}"
            elif rel:
                combo = f"{rel} (rel-only, between existing types)"
            else:
                combo = f"{node} (node-only)"
            lines.append(f"  - {combo}: {delta_txt} ({entry.get('status', 'pending')})")
        return "\n".join(lines)

    def _create_single_addition_prompt(
        self,
        previous: OntologyCandidate,
        metrics: ModelMetrics,
        step_index: int,
        variant_index: int,
        addition_history: list,
    ) -> str:
        """Prompt for single-addition mode: exactly one new node + one new rel.

        Args:
            previous: Previous ontology candidate
            metrics: Performance metrics of the previous winner
            step_index: Current pipeline step (1-based)
            variant_index: Which variant to create within this step
            addition_history: Prior additions with their ΔAUC outcomes

        Returns:
            Prompt string for LLM
        """
        existing_nodes = [self._node_label(n) for n in previous.schema.get("node_types", [])]
        existing_rels = previous.schema.get("relationship_types", [])
        history_block = self._format_addition_history(addition_history)

        auc_signal = (
            f"AUC is {metrics.auc:.4f} (+{metrics.auc - 0.5:.4f} above random)"
            if metrics.auc > 0.5
            else f"AUC is {metrics.auc:.4f} (at or below random)"
        )

        return f"""\
You are a knowledge graph ontology designer optimizing for financial news analysis.

TASK — step {step_index}, variant {variant_index} (SINGLE-ADDITION MODE):
Evolve the schema with EXACTLY ONE atomic addition. Choose ONE of:
  (a) one new node type PLUS one new relationship type connecting it to the schema, or
  (b) one new relationship type between EXISTING node types (no new node type).
Nothing else may change. This isolates the causal effect of your addition on downstream
stock-movement prediction (AUC). For (a), choose the single most promising financially
relevant concept (e.g. a corporate action, macro event, instrument, regulatory body,
supply-chain actor, geographic region, credit rating, ...). For (b), choose the single
most informative missing link between existing types (e.g. COMPETES_WITH or SUPPLIES_TO
between Company nodes).

ADDITIONS ALREADY TRIED (do NOT re-propose these combinations; learn from their ΔAUC):
{history_block}

Current schema:
```json
{json.dumps(previous.schema, indent=2)}
```

EXISTING ENTITY TYPES (do NOT re-add): {', '.join(existing_nodes)}
EXISTING RELATIONSHIP TYPES (do NOT re-add): {', '.join(existing_rels)}

Performance signal: {auc_signal}
Max relationship-chain hops — train: {metrics.max_hops_train}, val: {metrics.max_hops_val}

RULES:
1. Option (a): exactly 1 new node type (with 1–3 properties) + exactly 1 new relationship
   type. Option (b): exactly 1 new relationship type and NO new node type
2. Add 1–3 patterns, each of which must involve the new node type or the new relationship
3. Patterns MUST be ["NodeA", "RELATIONSHIP", "NodeB"] using only known node types
4. Keep ALL existing node_types, relationship_types and patterns — only add, never remove
5. Property types: STRING, INTEGER, FLOAT, BOOLEAN only
6. Return the FULL updated schema as valid JSON with keys: node_types, relationship_types,
   patterns — no prose"""

    def _create_evolution_prompt(
        self,
        previous: OntologyCandidate,
        metrics: ModelMetrics,
        step_index: int,
        variant_index: int,
    ) -> str:
        """Create prompt for LLM to evolve ontology.

        Args:
            previous: Previous ontology candidate
            metrics: Performance metrics of the previous winner
            step_index: Current pipeline step (1-based)
            variant_index: Which variant to create within this step

        Returns:
            Prompt string for LLM
        """
        existing_nodes = [
            n["label"] if isinstance(n, dict) else str(n)
            for n in previous.schema.get("node_types", [])
        ]
        existing_rels = previous.schema.get("relationship_types", [])

        # If custom prompt template is loaded, use it
        if self.custom_prompt_template:
            return self.custom_prompt_template.format(
                schema_json=json.dumps(previous.schema, indent=2),
                existing_nodes=", ".join(existing_nodes),
                existing_rels=", ".join(existing_rels),
                auc=metrics.auc,
                f1=metrics.f1,
                max_hops_train=metrics.max_hops_train,
                max_hops_val=metrics.max_hops_val,
                step_index=step_index,
                variant_index=variant_index,
            )

        # Identify generic rels still present that could be specialised
        generic_present = [r for r in self._GENERIC_RELS if r in existing_rels]
        specialisation_lines = []
        for rel in generic_present:
            options = ", ".join(self._REL_SPECIALISATIONS.get(rel, []))
            specialisation_lines.append(f"  - {rel}  →  consider: {options}")
        specialisation_block = (
            "\n".join(specialisation_lines)
            if specialisation_lines
            else "  (none — all generic rels already specialised)"
        )

        auc_signal = (
            f"AUC improved to {metrics.auc:.4f} (+{metrics.auc - 0.5:.4f} above random)"
            if metrics.auc > 0.5
            else f"AUC is {metrics.auc:.4f} (at or below random — relationships may be too generic)"
        )

        return f"""You are a knowledge graph ontology designer optimizing for financial news analysis.

TASK — step {step_index}, variant {variant_index}:
Evolve the schema by doing BOTH of the following:

1. ADD NEW ENTITY TYPES — propose 3–5 new node types NOT already in the schema that would
   capture financially relevant concepts (e.g. corporate actions, macro events, instruments,
   regulatory bodies, supply-chain actors, geographic regions, credit ratings, etc.).
   Be creative and domain-specific; do not repeat existing types.

2. SPECIALISE GENERIC RELATIONSHIPS — replace vague relationship types with semantically
   precise ones. Generic rels still present that should be specialised:
{specialisation_block}

Current schema:
```json
{json.dumps(previous.schema, indent=2)}
```

EXISTING ENTITY TYPES (do NOT re-add): {', '.join(existing_nodes)}
EXISTING RELATIONSHIP TYPES: {', '.join(existing_rels)}

Performance signal: {auc_signal}
Max relationship-chain hops — train: {metrics.max_hops_train}, val: {metrics.max_hops_val}

RULES:
1. Property types: STRING, INTEGER, FLOAT, BOOLEAN only
2. Patterns MUST be ["NodeA", "RELATIONSHIP", "NodeB"]
3. Keep ALL existing node_types and relationship_types — only add, never remove
4. Return ONLY valid JSON with keys: node_types, relationship_types, patterns — no prose"""

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
