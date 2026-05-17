import pytest
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from incremental_kg_mutator import IncrementalArticleKGMutator

# Load environment variables from parent directory .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class MockLLM:
    """Mock LLM for testing. Returns extracted entities as JSON."""
    
    async def run(self, prompt: str) -> str:
        """Sync-style LLM interface for testing."""
        return """{
  "nodes": [
    {"label": "Person", "key": "elon_musk", "properties": {"name": "Elon Musk"}},
    {"label": "Company", "key": "spacex", "properties": {"name": "SpaceX"}}
  ],
  "relationships": [
    {"type": "FOUNDED", "from_key": "elon_musk", "to_key": "spacex", "properties": {"year": 2002}}
  ]
}"""


@pytest.fixture(scope="module")
def neo4j_driver():
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    yield driver
    driver.close()


@pytest.mark.asyncio
async def test_incremental_article_mutation_new_entities(neo4j_driver):
    """Test adding new entities to an article"""
    with neo4j_driver.session() as session:
        session.run(
            """
            MERGE (:Article {id: "2024-09-10-tech-new"})
            """
        )

    mutator = IncrementalArticleKGMutator(
        driver=neo4j_driver,
        llm=MockLLM(),
    )

    await mutator.mutate_article(
        article_id="2024-09-10-tech-new",
        text="Elon Musk founded SpaceX.",
        ontology={
            "nodes": ["Person", "Company"],
            "relationships": ["FOUNDED"],
        },
    )

    with neo4j_driver.session() as session:
        person = session.run(
            """
            MATCH (p:Person {key: "elon_musk"})
            RETURN p.name AS name
            """
        ).single()
        assert person["name"] == "Elon Musk"

        company = session.run(
            """
            MATCH (c:Company {key: "spacex"})
            RETURN c.name AS name
            """
        ).single()
        assert company["name"] == "SpaceX"

        relationship = session.run(
            """
            MATCH (p:Person {key: "elon_musk"})-[r:FOUNDED]->(c:Company {key: "spacex"})
            RETURN r.year AS year
            """
        ).single()
        assert relationship["year"] == 2002

        mentioned = session.run(
            """
            MATCH (p:Person {key: "elon_musk"})-[:MENTIONED_IN]->(a:Article {id: "2024-09-10-tech-new"})
            RETURN count(*) AS cnt
            """
        ).single()
        assert mentioned["cnt"] == 1


@pytest.mark.asyncio
async def test_incremental_article_mutation_reuse_old_entities(neo4j_driver):
    """Test reusing existing entities and connecting them with new ones"""
    article_id = "2024-09-11-tech-reuse"
    
    with neo4j_driver.session() as session:
        session.run(
            """
            MERGE (a:Article {id: $article_id})
            MERGE (p:Person {key: "elon_musk", name: "Elon Musk"})
            MERGE (p)-[:MENTIONED_IN]->(a)
            """,
            article_id=article_id,
        )

    class MockLLMWithNewEntity:
        async def run(self, prompt: str) -> str:
            return """{
  "nodes": [
    {"label": "Person", "key": "elon_musk", "properties": {"name": "Elon Musk"}},
    {"label": "Company", "key": "spacex", "properties": {"name": "SpaceX"}}
  ],
  "relationships": [
    {"type": "FOUNDED", "from_key": "elon_musk", "to_key": "spacex", "properties": {"year": 2002}}
  ]
}"""

    mutator = IncrementalArticleKGMutator(
        driver=neo4j_driver,
        llm=MockLLMWithNewEntity(),
    )

    await mutator.mutate_article(
        article_id=article_id,
        text="Elon Musk founded SpaceX in 2002.",
        ontology={
            "nodes": ["Person", "Company"],
            "relationships": ["FOUNDED"],
        },
    )

    with neo4j_driver.session() as session:
        elon_still_exists = session.run(
            """
            MATCH (p:Person {key: "elon_musk"})
            RETURN p.name AS name
            """
        ).single()
        assert elon_still_exists["name"] == "Elon Musk"

        spacex_created = session.run(
            """
            MATCH (c:Company {key: "spacex"})
            RETURN c.name AS name
            """
        ).single()
        assert spacex_created["name"] == "SpaceX"

        old_and_new_connected = session.run(
            """
            MATCH (p:Person {key: "elon_musk"})-[r:FOUNDED]->(c:Company {key: "spacex"})
            RETURN r.year AS year
            """
        ).single()
        assert old_and_new_connected["year"] == 2002

        both_mentioned = session.run(
            """
            MATCH (a:Article {id: $article_id})
            WITH a
            OPTIONAL MATCH (p:Person {key: "elon_musk"})-[:MENTIONED_IN]->(a)
            OPTIONAL MATCH (c:Company {key: "spacex"})-[:MENTIONED_IN]->(a)
            RETURN count(distinct p) + count(distinct c) AS cnt
            """,
            article_id=article_id,
        ).single()
        assert both_mentioned["cnt"] == 2


@pytest.mark.asyncio
async def test_isolation_constraint_blocks_cross_candidate_links(neo4j_driver):
    """Test that entities from different candidates in same step cannot link"""
    article_id = "2024-09-12-isolation-test"
    
    # Setup: Create base entities that both candidates can link to
    with neo4j_driver.session() as session:
        session.run(
            """
            MERGE (a:Article {id: $article_id})
            MERGE (c:Company {key: "tesla", name: "Tesla"})
            SET c.candidate_tags = ["base"]
            MERGE (c)-[:MENTIONED_IN]->(a)
            """,
            article_id=article_id,
        )
    
    # Candidate 0: Creates Person "Tim Cook" from CANDIDATE_0
    class MockLLMCandidate0:
        async def run(self, prompt: str) -> str:
            return """{
  "nodes": [
    {"label": "Person", "key": "tim_cook", "properties": {"name": "Tim Cook"}}
  ],
  "relationships": [
    {"type": "WORKS_FOR", "from_key": "tim_cook", "to_key": "tesla", "properties": {}}
  ]
}"""
    
    mutator0 = IncrementalArticleKGMutator(neo4j_driver, MockLLMCandidate0())
    
    await mutator0.mutate_article(
        article_id=article_id,
        text="Tim Cook works at Tesla.",
        ontology={"nodes": ["Person"], "relationships": ["WORKS_FOR"]},
        accepted_tags=["base"],  # Can only link to base entities
        candidate_tag="step_1_candidate_0",
    )
    
    # Candidate 1: Creates Product "iPhone" from CANDIDATE_1
    class MockLLMCandidate1:
        async def run(self, prompt: str) -> str:
            return """{
  "nodes": [
    {"label": "Product", "key": "iphone", "properties": {"name": "iPhone"}}
  ],
  "relationships": [
    {"type": "MAKES", "from_key": "tesla", "to_key": "iphone", "properties": {}}
  ]
}"""
    
    mutator1 = IncrementalArticleKGMutator(neo4j_driver, MockLLMCandidate1())
    
    await mutator1.mutate_article(
        article_id=article_id,
        text="Tesla makes iPhone.",
        ontology={"nodes": ["Product"], "relationships": ["MAKES"]},
        accepted_tags=["base"],  # Can only link to base entities
        candidate_tag="step_1_candidate_1",
    )
    
    # Now try to create a relationship BETWEEN candidate_0 and candidate_1 entities
    # This should be BLOCKED by the constraint
    class MockLLMCrossCandidateLink:
        async def run(self, prompt: str) -> str:
            return """{
  "nodes": [],
  "relationships": [
    {"type": "USES", "from_key": "tim_cook", "to_key": "iphone", "properties": {}}
  ]
}"""
    
    mutator_cross = IncrementalArticleKGMutator(neo4j_driver, MockLLMCrossCandidateLink())
    
    await mutator_cross.mutate_article(
        article_id=article_id,
        text="Tim Cook uses iPhone.",
        ontology={},
        accepted_tags=["base"],  # CONSTRAINT: Can only link to base, NOT to other candidates
        candidate_tag="step_1_candidate_2",
    )
    
    # Verify the cross-candidate relationship was NOT created
    with neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (p:Person {key: "tim_cook"})-[r:USES]->(prod:Product {key: "iphone"})
            RETURN count(*) AS cnt
            """
        ).single()
        
        # Should be 0 because Tim (candidate_0) cannot link to iPhone (candidate_1)
        assert result["cnt"] == 0, "Cross-candidate relationship should be blocked by constraint"
        
        # Verify that Tim -> Tesla relationship WAS created (both link to base)
        result_valid = session.run(
            """
            MATCH (p:Person {key: "tim_cook"})-[r:WORKS_FOR]->(c:Company {key: "tesla"})
            RETURN count(*) AS cnt
            """
        ).single()
        assert result_valid["cnt"] == 1, "Relationship to accepted entity should exist"

