"""Neo4j graph operations."""

import logging
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase

logger = logging.getLogger(__name__)


class GraphDriver:
    """Neo4j driver wrapper for consistent operations."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """Initialize the driver.

        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
            database: Database name (default: neo4j)
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: Optional[Driver] = None

    @property
    def driver(self) -> Driver:
        """Get or create the driver."""
        if self._driver is None:
            logger.info("Connecting to Neo4j at %s", self.uri)
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self) -> None:
        """Close the driver."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the database."""
        logger.info("Clearing graph from database '%s'", self.database)
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.debug("Graph cleared")

    def run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Run a query and return results.

        Args:
            query: Cypher query
            parameters: Query parameters

        Returns:
            List of result dictionaries
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Execute a write query.

        Args:
            query: Cypher query
            parameters: Query parameters
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, parameters or {})

    def get_count(self, label: Optional[str] = None) -> int:
        """Get node count.

        Args:
            label: Optional node label to count specific type

        Returns:
            Number of nodes
        """
        query = f"MATCH (n{':' + label if label else ''}) RETURN count(n) AS count"
        result = self.run_query(query)
        return result[0]["count"] if result else 0

    def get_relationship_count(self) -> int:
        """Get relationship count."""
        result = self.run_query("MATCH ()-[r]->() RETURN count(r) AS count")
        return result[0]["count"] if result else 0

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
