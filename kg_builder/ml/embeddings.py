"""Embedding models."""

import logging
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from karateclub.node_embedding.neighbourhood import HOPE
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def compute_hope_embeddings(edges: List[Tuple[str, str]], dim: int = 128) -> Dict[str, np.ndarray]:
    """Compute HOPE embeddings for a graph.

    Args:
        edges: List of (src, dst) tuples representing edges
        dim: Embedding dimension (default 128)

    Returns:
        Dict mapping node_id to embedding vector
    """
    logger.info("Building HOPE graph from %d edges", len(edges))

    # Collect nodes from edges
    original_nodes = sorted({n for src, dst in edges for n in (src, dst)})
    N = len(original_nodes)
    logger.info("Graph contains %d unique nodes", N)

    if N < 2:
        logger.warning("Graph has <2 nodes — cannot compute HOPE. Returning empty dict.")
        return {}

    # Build remapping
    remap = {orig: i for i, orig in enumerate(original_nodes)}
    inv_remap = {i: orig for orig, i in remap.items()}

    # Build graph
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from([(remap[src], remap[dst]) for src, dst in edges])

    max_valid_dim = 2 * (N - 1)

    if dim >= max_valid_dim:
        logger.warning(
            "HOPE dim=%d too large for graph (N=%d). Capping to %d.", dim, N, max_valid_dim
        )
        dim = max_valid_dim

    if dim < 2:
        logger.warning("HOPE dim=%d is too small; minimum usable dim is 2. Raising to 2.", dim)
        dim = 2

    logger.info("Fitting HOPE model with final dim=%d (k=%d)", dim, dim // 2)

    model = HOPE(dimensions=dim)

    try:
        model.fit(G)
    except ValueError as e:
        logger.error("HOPE failed with ValueError: %s", e)
        logger.error("dim=%d, N=%d, edges=%d", dim, N, len(edges))
        raise

    emb = model.get_embedding()

    logger.info("HOPE embedding computed successfully for %d nodes", N)

    return {inv_remap[i]: emb[i] for i in range(N)}


def create_embedder(api_key: str, model: str = "text-embedding-ada-002") -> OpenAIEmbeddings:
    """Create an OpenAI embedder.

    Args:
        api_key: OpenAI API key
        model: Embedding model name

    Returns:
        OpenAIEmbeddings instance
    """
    logger.info("Creating embedder with model: %s", model)
    return OpenAIEmbeddings(api_key=api_key, model=model)
