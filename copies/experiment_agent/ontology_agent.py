from __future__ import annotations
from typing import List, Optional
import json
from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict
from ontology import OntologyCandidate

# ------------------------------------------------------------
# 1. Allowed Property Types
# ------------------------------------------------------------

ALLOWED_PROP_TYPES = {"STRING", "INTEGER", "FLOAT", "BOOLEAN"}


# ------------------------------------------------------------
# 2. Pydantic Structured Output Schema
# ------------------------------------------------------------

class PropertyDef(BaseModel):
    name: str
    type: str
    required: Optional[bool] = False

    model_config = ConfigDict(extra="ignore")


class NodeType(BaseModel):
    label: str
    properties: List[PropertyDef]

    model_config = ConfigDict(extra="ignore")


class OntologySchemaModel(BaseModel):
    """
    The ontology schema that the LLM must output.
    This MUST match the structure SimpleKGPipeline expects.
    """

    node_types: List[NodeType]
    relationship_types: List[str]
    patterns: List[List[str]]

    # GraphRAG-required flags (all booleans, not lists!)
    additional_node_types: bool = True
    additional_relationship_types: bool = True
    additional_patterns: bool = True

    model_config = ConfigDict(extra="ignore")


# ------------------------------------------------------------
# 3. Ontology Evolution Agent
# ------------------------------------------------------------

class OntologyEvolutionAgent:
    """Produces improved ontology schemas using OpenAI structured output."""

    def __init__(self, llm=None, api_key=None, model="gpt-4o-mini"):
        self.neo4j_llm = llm
        self.client = OpenAI(api_key=api_key)
        self.model = model

    async def propose_new_candidate(
        self,
        previous: OntologyCandidate,
        metrics: dict,
        step_index: int,
        variant_index: int,
    ) -> OntologyCandidate:

        prompt = f"""
You evolve an ontology for financial news graph construction.

Constraints:
- Modify ONLY node_types, relationship_types, patterns.
- Never remove existing node types unless truly redundant.
- Do not rename nodes or relationships.
- Avoid duplicates.
- Patterns MUST be formatted as ["NodeA", "REL", "NodeB"] (list, not tuple or dict).
- You MUST output a valid JSON object matching OntologySchemaModel.

ENTITY RESOLUTION NOTES:
- Node properties should include a canonical identifier field (e.g., "ticker" for companies, "slug" for articles)
- Encourage extraction systems to use consistent, normalized keys:
  * Ticker symbols: UPPERCASE (NVDA, AAPL, INTC)
  * Names/entities: lowercase, strip legal suffixes (apple, tesla, intel - NOT "apple_inc")
  * Documents: snake_case (article_1, market_update_2021)
- STRIP LEGAL SUFFIXES: Remove Inc., Corp., LLC., Ltd., Co., Corporation, Company, etc.
  * "Apple Inc." → "apple"
  * "Tesla, Inc." → "tesla"
  * "Intel Corporation" → "intel"
- Duplicate resolution is handled by key normalization, so consistent key formats in ontology improve deduplication

Previous ontology:
{json.dumps(previous.schema, indent=2)}

Model metrics:
AUC={metrics.get("auc")}, ACC={metrics.get("acc")}
"""

        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON matching OntologySchemaModel. No prose."
                },
                {"role": "user", "content": prompt},
            ],
            response_format=OntologySchemaModel,
        )

        parsed: OntologySchemaModel = completion.choices[0].message.parsed

        with open("ontology_debug.log", "a") as f:
            f.write("\n===== DEBUG: Ontology Schema =====\n")
            f.write(json.dumps(parsed.model_dump(), indent=2) + "\n")

        # Convert to dict + sanitize thoroughly
        schema_dict = sanitize_schema(parsed.model_dump())

        return OntologyCandidate(
            name=f"{previous.name}_step{step_index}_variant{variant_index}",
            schema=schema_dict,
            description="Ontology evolved using structured OpenAI output",
            parent_name=previous.name,
            step_index=previous.step_index + 1,
        )

def sanitize_schema(schema: dict) -> dict:
    """
    Fully sanitizes an ontology schema into a form guaranteed to pass
    SimpleKGPipelineConfig validation for GraphRAG.
    """

    # 1. Fix common alias mistakes (if LLM ever returns them)
    alias_map = {
        "nodes": "node_types",
        "types": "node_types",
        "entities": "node_types",
        "rels": "relationship_types",
        "edges": "relationship_types",
        "relationships": "relationship_types",
        "links": "relationship_types",
    }
    for src, dst in alias_map.items():
        if src in schema and dst not in schema:
            schema[dst] = schema.pop(src)

    # 2. Keep only allowed keys
    allowed_top = {
        "node_types",
        "relationship_types",
        "patterns",
        "additional_node_types",
        "additional_relationship_types",
        "additional_patterns",
    }
    schema = {k: v for k, v in schema.items() if k in allowed_top}

    # 3. Ensure core lists exist
    if not isinstance(schema.get("node_types"), list):
        schema["node_types"] = []
    if not isinstance(schema.get("relationship_types"), list):
        schema["relationship_types"] = []
    if not isinstance(schema.get("patterns"), list):
        schema["patterns"] = []

    # 4. Coerce the three flags to booleans (critical!)
    schema["additional_node_types"] = bool(schema.get("additional_node_types", True))
    schema["additional_relationship_types"] = bool(
        schema.get("additional_relationship_types", True)
    )
    schema["additional_patterns"] = bool(schema.get("additional_patterns", True))

    # 5. Normalize relationship types (list of unique strings)
    rtypes = schema["relationship_types"]
    schema["relationship_types"] = list(dict.fromkeys(str(r).strip() for r in rtypes))

    # 6. Normalize node types
    def sanitize_node(nt):
        if not isinstance(nt, dict):
            return None
        label = nt.get("label", "")
        if not isinstance(label, str) or not label.strip():
            return None
        label = label.strip()

        props = nt.get("properties", [])
        if not isinstance(props, list):
            props = []

        clean_props = []
        for p in props:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name", "")).strip()
            if not name:
                continue
            typ = str(p.get("type", "STRING")).upper()
            if typ not in ALLOWED_PROP_TYPES:
                typ = "STRING"
            required = bool(p.get("required", False))
            clean_props.append({"name": name, "type": typ, "required": required})

        return {"label": label, "properties": clean_props}

    node_types = schema["node_types"]
    schema["node_types"] = [
        nt for nt in (sanitize_node(nt) for nt in node_types) if nt is not None
    ]

    # 7. Normalize patterns into ["A","REL","B"] lists
    def normalize_pattern(p):
        if isinstance(p, dict) and {"source", "relationship", "target"} <= p.keys():
            return [str(p["source"]), str(p["relationship"]), str(p["target"])]
        if isinstance(p, tuple):
            p = list(p)
        if isinstance(p, list) and len(p) == 3:
            return [str(x) for x in p]
        return None

    patterns = schema["patterns"]
    fixed_patterns = []
    for p in patterns:
        np = normalize_pattern(p)
        if np is not None:
            fixed_patterns.append(np)
    schema["patterns"] = fixed_patterns

    return schema
