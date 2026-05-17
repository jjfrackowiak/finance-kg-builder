from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from neo4j import Driver
from neo4j_graphrag.llm import LLMInterface
import logging

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
        """
        logger.debug("Starting mutate_article for article_id=%s, candidate_tag=%s", article_id, candidate_tag)
        
        # Extract using LLM
        extraction = await self._extract(text, ontology)
        logger.info("Extracted from article %s: %d nodes, %d relationships", 
                   article_id, len(extraction.nodes), len(extraction.relationships))
        
        if extraction.nodes:
            logger.debug("Extracted node types: %s", [n.label for n in extraction.nodes])
        if extraction.relationships:
            logger.debug("Extracted relationship types: %s", [r.type for r in extraction.relationships])
        
        # Apply mutation with isolation constraints
        with self.driver.session() as session:
            session.execute_write(
                self._apply_mutation,
                article_id,
                extraction,
                accepted_tags,
                candidate_tag,
            )
        
        logger.debug("Applied mutation to graph for article_id=%s", article_id)

    async def _extract(
        self, text: str, ontology: Dict[str, Any]
    ) -> ExtractionResult:
        """
        Extract entities and relationships using the configured LLM.
        
        The LLM is called with a structured prompt asking for JSON output
        in the format: {"nodes": [...], "relationships": [...]}
        """
        logger.debug("Starting LLM extraction with ontology containing %d node types", 
                    len(ontology.get("node_types", [])))
        
        prompt = f"""Extract entities and relationships as JSON. Return ONLY valid JSON.

Ontology:
{ontology}

Text:
{text}

Return this JSON structure exactly:
{{
  "nodes": [
    {{"label": "Type", "key": "identifier", "properties": {{"name": "Name"}}}},
    ...
  ],
  "relationships": [
    {{"type": "REL_TYPE", "from_key": "source_id", "to_key": "target_id", "properties": {{}}}},
    ...
  ]
}}
"""
        # Support both async (ainvoke) and sync (run) LLM interfaces
        try:
            if hasattr(self.llm, 'ainvoke'):
                logger.debug("Using LLM.ainvoke() method")
                response = await self.llm.ainvoke(prompt)
                # Handle response that might have .content attribute
                if hasattr(response, 'content'):
                    json_str = response.content
                else:
                    json_str = response
            elif hasattr(self.llm, 'run'):
                logger.debug("Using LLM.run() method")
                response = await self.llm.run(prompt)
                if hasattr(response, 'content'):
                    json_str = response.content
                else:
                    json_str = response
            else:
                raise AttributeError(f"LLM {type(self.llm)} must have either 'ainvoke' or 'run' method")
            
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
            logger.info("Successfully extracted %d nodes and %d relationships", 
                       len(result.nodes), len(result.relationships))
            return result
        except Exception as e:
            logger.error("JSON parsing failed: %s. JSON string: %s", str(e), json_str[:200])
            raise

    @staticmethod
    def _apply_mutation(
        tx,
        article_id: str,
        extraction: ExtractionResult,
        accepted_tags: Optional[List[str]] = None,
        candidate_tag: Optional[str] = None,
    ):
        """
        Apply the extracted entities and relationships to the Neo4j graph with constraints.
        
        This method:
        1. Updates the article's lastUpdated timestamp
        2. Creates/merges extracted nodes with candidate_tag
        3. Creates relationships respecting the isolation constraint:
           - If accepted_tags is provided, can only link to entities with those tags
           - Cannot link to entities from other candidates
        """
        logger.debug("Applying mutation with article_id=%s, candidate_tag=%s, accepted_tags=%s", 
                    article_id, candidate_tag, accepted_tags)
        
        # Update article
        tx.run(
            """
            MATCH (a:Article {id: $article_id})
            SET a.lastUpdated = datetime()
            """,
            article_id=article_id,
        )
        logger.debug("Updated article timestamp for %s", article_id)

        # Create/merge nodes
        nodes_created = 0
        for node in extraction.nodes:
            query = f"""
                MERGE (n:{node.label} {{key: $key}})
                SET n += $props
                """
            if candidate_tag:
                query += """
                SET n.candidate_tags = CASE
                    WHEN n.candidate_tags IS NULL THEN [$candidate_tag]
                    ELSE n.candidate_tags + [$candidate_tag]
                END
                """
            query += """
                WITH n
                MATCH (a:Article {id: $article_id})
                MERGE (n)-[:MENTIONED_IN]->(a)
                RETURN n
                """
            result = tx.run(
                query,
                key=node.key,
                props=node.properties,
                candidate_tag=candidate_tag,
                article_id=article_id,
            )
            # Count actual creations
            if result.consume().counters.nodes_created > 0:
                nodes_created += 1
            logger.debug("Created/merged node: %s{key: %s} with candidate_tag=%s", 
                        node.label, node.key, candidate_tag)
        
        logger.info("Applied %d nodes to graph for article %s with candidate_tag=%s", 
                   len(extraction.nodes), article_id, candidate_tag)

        # Create relationships with isolation constraints
        rels_created = 0
        for rel in extraction.relationships:
            if accepted_tags is not None:
                # Constraint: Can only link to entities with accepted_tags OR self's candidate_tag
                query = f"""
                    MATCH (s {{key: $from_key}})
                    MATCH (t {{key: $to_key}})
                    WHERE
                      (ANY(tag IN s.candidate_tags WHERE tag IN $accepted_tags) OR ANY(tag IN s.candidate_tags WHERE tag = $candidate_tag))
                      AND
                      (ANY(tag IN t.candidate_tags WHERE tag IN $accepted_tags) OR ANY(tag IN t.candidate_tags WHERE tag = $candidate_tag))
                    MERGE (s)-[r:{rel.type}]->(t)
                    SET r += $props
                    """
                if candidate_tag:
                    query += """
                    SET r.candidate_tags = CASE
                        WHEN r.candidate_tags IS NULL THEN [$candidate_tag]
                        ELSE r.candidate_tags + [$candidate_tag]
                    END
                    """
                query += "RETURN r"
                result = tx.run(
                    query,
                    from_key=rel.from_key,
                    to_key=rel.to_key,
                    props=rel.properties,
                    accepted_tags=accepted_tags or [],
                    candidate_tag=candidate_tag,
                )
                if result.consume().counters.relationships_created > 0:
                    rels_created += 1
                logger.debug("Created relationship (with isolation): %s{%s->%s} with candidate_tag=%s", 
                            rel.type, rel.from_key, rel.to_key, candidate_tag)
            else:
                # No constraint: link freely
                query = f"""
                    MATCH (s {{key: $from_key}})
                    MATCH (t {{key: $to_key}})
                    MERGE (s)-[r:{rel.type}]->(t)
                    SET r += $props
                    """
                if candidate_tag:
                    query += """
                    SET r.candidate_tags = CASE
                        WHEN r.candidate_tags IS NULL THEN [$candidate_tag]
                        ELSE r.candidate_tags + [$candidate_tag]
                    END
                    """
                query += "RETURN r"
                result = tx.run(
                    query,
                    from_key=rel.from_key,
                    to_key=rel.to_key,
                    props=rel.properties,
                    candidate_tag=candidate_tag,
                )
                if result.consume().counters.relationships_created > 0:
                    rels_created += 1
                logger.debug("Created relationship (no isolation): %s{%s->%s} with candidate_tag=%s", 
                            rel.type, rel.from_key, rel.to_key, candidate_tag)
        
        logger.info("Applied %d relationships to graph for article %s with candidate_tag=%s", 
                   len(extraction.relationships), article_id, candidate_tag)
