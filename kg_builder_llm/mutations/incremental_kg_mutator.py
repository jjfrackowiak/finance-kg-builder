import asyncio
import logging
from typing import Any, Dict, List, Optional

from neo4j import Driver
from neo4j_graphrag.llm import LLMInterface
from pydantic import BaseModel

from kg_builder_llm.core.entity_resolution import canonical_key
from kg_builder_llm.core.ids import article_text_id

logger = logging.getLogger(__name__)


class ExtractedNode(BaseModel):
    label: str
    key: str
    properties: Dict[str, Any]


class ExtractedRelation(BaseModel):
    type: str
    from_key: str
    to_key: str
    properties: Dict[str, Any] = {}


class ExtractionResult(BaseModel):
    nodes: List[ExtractedNode]
    relationships: List[ExtractedRelation]


class IncrementalArticleKGMutator:
    """
    Incrementally mutates an existing article subgraph using an LLM for extraction.

    Uses Neo4j's LLMInterface pattern (supports both sync and async LLMs from neo4j_graphrag).

    Key properties:
    - accepted_tags: List of tags for entities that new extractions can link to
    - candidate_tag: Tag applied to newly extracted entities to identify their source
    - Isolation constraint: Entities from different candidates in same step cannot link to each other
    """

    def __init__(self, driver: Driver, llm: LLMInterface):
        """
        Initialize the incremental KG mutator.

        Args:
            driver: Neo4j driver instance
            llm: LLMInterface instance (supports both sync .run() and async .ainvoke())
        """
        self.driver = driver
        self.llm = llm

    async def mutate_article(
        self,
        article_id: str,
        text: str,
        ontology: Dict[str, Any],
        accepted_tags: Optional[List[str]] = None,
        candidate_tag: Optional[str] = None,
        article_date: Optional[str] = None,
        text_embedding: Optional[List[float]] = None,
        write_lock: Optional[asyncio.Lock] = None,
    ) -> None:
        """
        Extract entities and relationships from text, then apply them to the graph
        with isolation constraints.

        The isolation constraint ensures that:
        - New entities can only link to entities with accepted_tags or to this candidate
        - New entities cannot link to entities from other candidates in the same step

        Args:
            article_id: ID of the article being processed
            text: Text content to extract from
            ontology: Schema guiding extraction (dict with 'nodes' and 'relationships')
            accepted_tags: Tags of entities new extractions can link to (isolation constraint)
            candidate_tag: Tag to apply to newly extracted entities to identify their origin
            article_date: Publication date of the article (YYYY-MM-DD format) for temporal features
            text_embedding: Deterministic text embedding vector (1536-dim for text-embedding-3-small)
        """
        text_hash = article_text_id(text)
        extraction = await self._extract(text, ontology)
        logger.info(
            "Extracted from article %s: %d nodes, %d relationships",
            article_id,
            len(extraction.nodes),
            len(extraction.relationships),
        )

        if len(extraction.nodes) == 0:
            logger.warning(
                "⚠️  NO NODES EXTRACTED from article %s (text length: %d chars)",
                article_id,
                len(text),
            )
        if len(extraction.relationships) == 0:
            logger.warning("⚠️  NO RELATIONSHIPS EXTRACTED from article %s", article_id)

        # Normalize all node keys using canonical_key (ticker lookup + name-slug)
        # and build a mapping llm_key -> canonical_key so that relationship
        # from_key/to_key references stay consistent.
        key_map: dict[str, str] = {}
        for node in extraction.nodes:
            name = node.properties.get("name", "")
            ck = canonical_key(node.key, name, node.label)
            if ck != node.key:
                logger.debug(
                    "Key normalised [%s]: %r -> %r (name=%r)",
                    node.label, node.key, ck, name,
                )
            key_map[node.key] = ck
            node.key = ck

        for rel in extraction.relationships:
            rel.from_key = key_map.get(rel.from_key, canonical_key(rel.from_key, "", ""))
            rel.to_key = key_map.get(rel.to_key, canonical_key(rel.to_key, "", ""))

        # Apply mutation with isolation constraints. The Neo4j session is
        # synchronous, so run the write in a thread to keep the event loop free
        # for other articles' LLM calls. Concurrent writes MERGE the same hub
        # nodes and deadlock, so callers pass a lock to serialize them.
        def _write() -> None:
            with self.driver.session() as session:
                session.execute_write(
                    self._apply_mutation,
                    article_id,
                    text_hash,
                    extraction,
                    accepted_tags,
                    candidate_tag,
                    article_date,
                    text_embedding,
                )

        if write_lock is not None:
            async with write_lock:
                await asyncio.to_thread(_write)
        else:
            await asyncio.to_thread(_write)

        logger.debug("Applied mutation to graph for article_id=%s", article_id)

    async def _extract(self, text: str, ontology: Dict[str, Any]) -> ExtractionResult:
        """
        Extract entities and relationships using the configured LLM.

        The LLM is called with a structured prompt asking for JSON output
        in the format: {"nodes": [...], "relationships": [...]}
        
        Key normalization is handled automatically by the system, but LLM is
        instructed to create consistent, identifier-like keys for better deduplication.
        """
        # ~800 token prompt overhead; keep article under 6000 tokens (~24000 chars) for 8192 limit
        text = text[:24000]

        prompt = f"""Extract entities and relationships from the text below using the provided ontology. Return ONLY valid JSON — no prose, no markdown.

ONTOLOGY:
{ontology}

TEXT:
{text}

KEY RULES — follow exactly, one rule per label:
- Company   → stock ticker in UPPERCASE, no suffixes  (e.g., "NVDA", "AAPL", "INTC", "TSLA", "TSM")
              If ticker unknown, use lowercase slug of the name: "taiwan_semiconductor"
- Sector    → lowercase slug of the sector name       (e.g., "technology", "semiconductors", "healthcare")
- Market    → lowercase slug of the market name       (e.g., "us_equity_market", "crypto_market")
- Index     → lowercase slug of the index name        (e.g., "sp500", "nasdaq_composite", "dow_jones")
- Author / Insider / Executive → lowercase slug of full name (e.g., "jensen_huang", "lisa_su")
- Deal / Event / Regulation   → short lowercase slug describing the event (e.g., "arm_acquisition_2023", "gdpr")
- Fund      → lowercase slug of the fund name         (e.g., "ark_innovation_etf")

STRICT PROHIBITIONS:
- NEVER output nodes with label "Article" or "Day" — these are managed by the system
- NEVER use generic keys like "company_1", "deal_1", "market_1", "article_1", "publication_date"
- NEVER invent relationship types not in the ontology

Return this exact JSON structure:
{{
  "nodes": [
    {{"label": "Type", "key": "canonical_key", "properties": {{"name": "Full Name"}}}},
    ...
  ],
  "relationships": [
    {{"type": "REL_TYPE", "from_key": "source_key", "to_key": "target_key", "properties": {{}}}},
    ...
  ]
}}

Example:
{{
  "nodes": [
    {{"label": "Company", "key": "NVDA", "properties": {{"name": "NVIDIA"}}}},
    {{"label": "Company", "key": "INTC", "properties": {{"name": "Intel"}}}},
    {{"label": "Sector",  "key": "semiconductors", "properties": {{"name": "Semiconductors"}}}},
    {{"label": "Index",   "key": "sp500", "properties": {{"name": "S&P 500"}}}}
  ],
  "relationships": [
    {{"type": "PART_OF", "from_key": "NVDA", "to_key": "semiconductors", "properties": {{}}}},
    {{"type": "PART_OF", "from_key": "INTC", "to_key": "semiconductors", "properties": {{}}}},
    {{"type": "TRACKS",  "from_key": "semiconductors", "to_key": "sp500", "properties": {{}}}}
  ]
}}
"""
        # Support both async (ainvoke) and sync (run) LLM interfaces
        try:
            if hasattr(self.llm, "ainvoke"):
                logger.debug("Using LLM.ainvoke() method")
                response = await self.llm.ainvoke(prompt)
                # Handle response that might have .content attribute
                if hasattr(response, "content"):
                    json_str = response.content
                else:
                    json_str = response
            elif hasattr(self.llm, "run"):
                logger.debug("Using LLM.run() method")
                response = await self.llm.run(prompt)
                if hasattr(response, "content"):
                    json_str = response.content
                else:
                    json_str = response
            else:
                raise AttributeError(
                    f"LLM {type(self.llm)} must have either 'ainvoke' or 'run' method"
                )

            logger.debug("LLM response received, parsing JSON")
        except Exception as e:
            logger.error("LLM extraction failed: %s", str(e), exc_info=True)
            raise

        # Parse JSON response (handle markdown code blocks)
        json_str = json_str.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

        try:
            result = ExtractionResult.model_validate_json(json_str)
            logger.info(
                "Successfully extracted %d nodes and %d relationships",
                len(result.nodes),
                len(result.relationships),
            )
            return result
        except Exception as e:
            logger.error("JSON parsing failed: %s. JSON string: %s", str(e), json_str[:200])
            raise

    @staticmethod
    def _apply_mutation(
        tx,
        article_id: str,
        text_hash: str,
        extraction: ExtractionResult,
        accepted_tags: Optional[List[str]] = None,
        candidate_tag: Optional[str] = None,
        article_date: Optional[str] = None,
        text_embedding: Optional[List[float]] = None,
    ):
        """
        Apply the extracted entities and relationships to the Neo4j graph with constraints.

        This method:
        1. Updates the article's lastUpdated timestamp and stores date/embedding
        2. Creates/merges extracted nodes with candidate_tag
        3. Creates relationships respecting the isolation constraint:
           - If accepted_tags is provided, can only link to entities with those tags
           - Cannot link to entities from other candidates
        """
        logger.debug(
            "Applying mutation with article_id=%s, text_hash=%s, candidate_tag=%s, accepted_tags=%s",
            article_id,
            text_hash,
            candidate_tag,
            accepted_tags,
        )

        # Update article with date and embedding
        article_update_query = """
            MATCH (a:Article {id: $article_id})
            SET a.lastUpdated = datetime()
        """
        params = {"article_id": text_hash}
        
        if article_date:
            article_update_query += ", a.date = $article_date"
            params["article_date"] = article_date
        
        if text_embedding is not None:
            article_update_query += ", a.text_embedding = $text_embedding"
            params["text_embedding"] = text_embedding
        
        tx.run(article_update_query, **params)
        logger.debug("Updated article timestamp for %s (text_hash=%s)", article_id, text_hash)

        # Create/merge nodes with candidate_tags
        nodes_created = 0
        for node in extraction.nodes:
            # Skip infrastructure node labels (Article, Day) - these are managed separately
            if node.label in ["Article", "Day"]:
                logger.warning(
                    "Skipping LLM-extracted node with infrastructure label: %s (key: %s). "
                    "Article and Day nodes are managed by the orchestrator, not extracted by LLM.",
                    node.label,
                    node.key,
                )
                continue
            
            # Key is already canonical (normalised in mutate_article before this call)
            normalized_key = node.key

            query = f"""
                MERGE (n:{node.label} {{key: $key}})
                SET n += $props
                SET n.candidate_tags = CASE
                    WHEN n.candidate_tags IS NULL THEN [$candidate_tag]
                    WHEN NOT $candidate_tag IN n.candidate_tags THEN n.candidate_tags + [$candidate_tag]
                    ELSE n.candidate_tags
                END
                WITH n
                MATCH (a:Article {{id: $article_id}})
                SET n.first_seen = CASE
                    WHEN n.first_seen IS NULL THEN date($article_date)
                    WHEN date($article_date) < n.first_seen THEN date($article_date)
                    ELSE n.first_seen
                END
                MERGE (a)-[:MENTIONS]->(n)
                RETURN n
                """
            try:
                result = tx.run(
                    query,
                    key=normalized_key,
                    props=node.properties,
                    candidate_tag=candidate_tag,
                    article_id=text_hash,
                    article_date=article_date,
                )
                records = result.consume()
                if records.counters.nodes_created > 0:
                    nodes_created += 1
                logger.debug(
                    "Node: %s{key: %s} (normalized from %s) with candidate_tag=%s (counters: %s)",
                    node.label,
                    normalized_key,
                    node.key,
                    candidate_tag,
                    records.counters,
                )
            except Exception as e:
                logger.error(
                    "FAILED to create node %s{%s}: %s", node.label, node.key, str(e), exc_info=True
                )

        logger.info(
            "Applied %d nodes to graph for article %s with candidate_tag=%s. Created: %d new",
            len(extraction.nodes),
            article_id,
            candidate_tag,
            nodes_created,
        )

        # Create relationships with isolation constraints
        rels_created = 0
        for rel in extraction.relationships:
            # Keys are already canonical (normalised in mutate_article)
            normalized_from_key = rel.from_key
            normalized_to_key = rel.to_key
            
            if accepted_tags is not None:
                # Constraint: Can only link to entities with accepted_tags OR self's candidate_tag
                # Use COALESCE to safely handle NULL candidate_tags
                query = f"""
                    MATCH (s {{key: $from_key}})
                    MATCH (t {{key: $to_key}})
                    WHERE
                      (ANY(tag IN COALESCE(s.candidate_tags, []) WHERE tag IN $accepted_tags) OR $candidate_tag IN COALESCE(s.candidate_tags, []))
                      AND
                      (ANY(tag IN COALESCE(t.candidate_tags, []) WHERE tag IN $accepted_tags) OR $candidate_tag IN COALESCE(t.candidate_tags, []))
                    MERGE (s)-[r:{rel.type}]->(t)
                    SET r += $props
                    SET r.candidate_tags = CASE
                        WHEN r.candidate_tags IS NULL THEN [$candidate_tag]
                        WHEN NOT $candidate_tag IN r.candidate_tags THEN r.candidate_tags + [$candidate_tag]
                        ELSE r.candidate_tags
                    END
                    RETURN r
                    """
                try:
                    result = tx.run(
                        query,
                        from_key=normalized_from_key,
                        to_key=normalized_to_key,
                        props=rel.properties,
                        accepted_tags=accepted_tags or [],
                        candidate_tag=candidate_tag,
                    )
                    records = result.consume()
                    if records.counters.relationships_created > 0:
                        rels_created += 1
                    logger.debug(
                        "Relationship (with isolation): %s{%s->%s} (normalized from %s->%s) with candidate_tag=%s (counters: %s)",
                        rel.type,
                        normalized_from_key,
                        normalized_to_key,
                        rel.from_key,
                        rel.to_key,
                        candidate_tag,
                        records.counters,
                    )
                except Exception as e:
                    logger.error(
                        "FAILED to create relationship %s{%s->%s}: %s",
                        rel.type,
                        rel.from_key,
                        rel.to_key,
                        str(e),
                        exc_info=True,
                    )
            else:
                # No constraint: link freely
                query = f"""
                    MATCH (s {{key: $from_key}})
                    MATCH (t {{key: $to_key}})
                    MERGE (s)-[r:{rel.type}]->(t)
                    SET r += $props
                    SET r.candidate_tags = CASE
                        WHEN r.candidate_tags IS NULL THEN [$candidate_tag]
                        WHEN NOT $candidate_tag IN r.candidate_tags THEN r.candidate_tags + [$candidate_tag]
                        ELSE r.candidate_tags
                    END
                    RETURN r
                    """
                try:
                    result = tx.run(
                        query,
                        from_key=normalized_from_key,
                        to_key=normalized_to_key,
                        props=rel.properties,
                        candidate_tag=candidate_tag,
                    )
                    records = result.consume()
                    if records.counters.relationships_created > 0:
                        rels_created += 1
                    logger.debug(
                        "Relationship (no isolation): %s{%s->%s} (normalized from %s->%s) with candidate_tag=%s (counters: %s)",
                        rel.type,
                        normalized_from_key,
                        normalized_to_key,
                        rel.from_key,
                        rel.to_key,
                        candidate_tag,
                        records.counters,
                    )
                except Exception as e:
                    logger.error(
                        "FAILED to create relationship (no constraint) %s{%s->%s}: %s",
                        rel.type,
                        rel.from_key,
                        rel.to_key,
                        str(e),
                        exc_info=True,
                    )

        logger.info(
            "Applied %d relationships to graph for article %s with candidate_tag=%s. Created: %d new",
            len(extraction.relationships),
            article_id,
            candidate_tag,
            rels_created,
        )
