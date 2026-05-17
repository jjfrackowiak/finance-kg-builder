from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from neo4j import Driver
from neo4j_graphrag.llm import LLMInterface


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
        # Extract using LLM
        extraction = await self._extract(text, ontology)
        
        # Apply mutation with isolation constraints
        with self.driver.session() as session:
            session.execute_write(
                self._apply_mutation,
                article_id,
                extraction,
                accepted_tags,
                candidate_tag,
            )

    async def _extract(
        self, text: str, ontology: Dict[str, Any]
    ) -> ExtractionResult:
        """
        Extract entities and relationships using the configured LLM.
        
        The LLM is called with a structured prompt asking for JSON output
        in the format: {"nodes": [...], "relationships": [...]}
        """
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
        if hasattr(self.llm, 'ainvoke'):
            response = await self.llm.ainvoke(prompt)
            # Handle response that might have .content attribute
            if hasattr(response, 'content'):
                json_str = response.content
            else:
                json_str = response
        elif hasattr(self.llm, 'run'):
            response = await self.llm.run(prompt)
            if hasattr(response, 'content'):
                json_str = response.content
            else:
                json_str = response
        else:
            raise AttributeError(f"LLM {type(self.llm)} must have either 'ainvoke' or 'run' method")
        
        # Parse JSON response (handle markdown code blocks)
        json_str = json_str.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()
        
        return ExtractionResult.model_validate_json(json_str)

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
        # Update article
        tx.run(
            """
            MATCH (a:Article {id: $article_id})
            SET a.lastUpdated = datetime()
            """,
            article_id=article_id,
        )

        # Create/merge nodes
        for node in extraction.nodes:
            query = f"""
                MERGE (n:{node.label} {{key: $key}})
                SET n += $props
                """
            if candidate_tag:
                query += """
                SET n.candidate_tag = $candidate_tag
                """
            query += """
                WITH n
                MATCH (a:Article {id: $article_id})
                MERGE (n)-[:MENTIONED_IN]->(a)
                """
            tx.run(
                query,
                key=node.key,
                props=node.properties,
                candidate_tag=candidate_tag,
                article_id=article_id,
            )

        # Create relationships with isolation constraints
        for rel in extraction.relationships:
            if accepted_tags is not None:
                # Constraint: Can only link to entities with accepted_tags OR self's candidate_tag
                query = f"""
                    MATCH (s {{key: $from_key}})
                    MATCH (t {{key: $to_key}})
                    WHERE
                      (ANY(tag IN s.candidate_tags WHERE tag IN $accepted_tags) OR s.candidate_tag = $candidate_tag)
                      AND
                      (ANY(tag IN t.candidate_tags WHERE tag IN $accepted_tags) OR t.candidate_tag = $candidate_tag)
                    MERGE (s)-[r:{rel.type}]->(t)
                    SET r += $props
                    """
                if candidate_tag:
                    query += """
                    SET r.candidate_tag = $candidate_tag
                    """
                tx.run(
                    query,
                    from_key=rel.from_key,
                    to_key=rel.to_key,
                    props=rel.properties,
                    accepted_tags=accepted_tags or [],
                    candidate_tag=candidate_tag,
                )
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
                    SET r.candidate_tag = $candidate_tag
                    """
                tx.run(
                    query,
                    from_key=rel.from_key,
                    to_key=rel.to_key,
                    props=rel.properties,
                    candidate_tag=candidate_tag,
                )
