"""Embedding models."""

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# Lazy-load sentence transformer model
_local_model = None


def get_local_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Get or create the local sentence transformer model.
    
    Args:
        model_name: Name of the sentence-transformers model
    
    Returns:
        SentenceTransformer model instance
    """
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedding model: {model_name}")
        _local_model = SentenceTransformer(model_name)
        logger.info(f"✓ Model loaded: {model_name} ({_local_model.get_sentence_embedding_dimension()} dims)")
    return _local_model


def embed_text_local(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed text using local sentence-transformers model.
    
    Uses all-MiniLM-L6-v2 (384 dimensions) by default - fast, efficient, and runs locally.
    No API key required, completely free.
    
    Args:
        text: Text to embed
        model_name: Sentence-transformers model name (default: all-MiniLM-L6-v2)
    
    Returns:
        Embedding vector (numpy array, shape: (384,) for all-MiniLM-L6-v2)
    """
    model = get_local_embedder(model_name)
    logger.debug("Embedding text (length=%d) with local model=%s", len(text), model_name)
    
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding.astype(np.float32)


def get_embedding_dim(embedding_type: str = "local", local_model: str = "all-MiniLM-L6-v2") -> int:
    """Get embedding dimension for the specified type.
    
    Args:
        embedding_type: "local" or "openai"
        local_model: Sentence-transformers model name (for local only)
    
    Returns:
        Embedding dimension (384 for local all-MiniLM-L6-v2, 1536 for OpenAI)
    """
    if embedding_type == "local":
        model = get_local_embedder(local_model)
        return model.get_sentence_embedding_dimension()
    return 1536  # OpenAI text-embedding-3-small


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

    k = dim // 2
    logger.info("Fitting HOPE model with final dim=%d (k=%d)", dim, k)

    # HOPE via Katz similarity SVD — equivalent to karateclub's HOPE implementation
    A = nx.to_scipy_sparse_array(G, nodelist=range(N), format="csr", dtype=float)
    beta = 0.01
    I = sp.eye(N, format="csr")
    try:
        S = spla.inv((I - beta * A).tocsc()) @ (beta * A)
        U, sigma, Vt = spla.svds(S, k=k)
        sqrt_sigma = np.sqrt(np.maximum(sigma, 0))
        emb = np.hstack([U * sqrt_sigma, Vt.T * sqrt_sigma])
    except Exception as e:
        logger.error("HOPE failed: %s (dim=%d, N=%d, edges=%d)", e, dim, N, len(edges))
        raise

    logger.info("HOPE embedding computed successfully for %d nodes", N)

    return {inv_remap[i]: emb[i] for i in range(N)}


def embed_text_deterministic(text: str, api_key: str = None, model: str = "text-embedding-3-small", embedding_type: str = "openai", local_model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed text using OpenAI or local sentence-transformers.
    
    Args:
        text: Text to embed
        api_key: OpenAI API key (required if embedding_type="openai")
        model: OpenAI embedding model (default: text-embedding-3-small)
        embedding_type: "openai" (1536 dims, paid) or "local" (384 dims, free)
        local_model: Sentence-transformers model name (default: all-MiniLM-L6-v2)
    
    Returns:
        Embedding vector (numpy array, shape depends on embedding_type:
        - openai: (1536,) for text-embedding-3-small
        - local: (384,) for all-MiniLM-L6-v2
    """
    if embedding_type == "local":
        return embed_text_local(text, model_name=local_model)
    
    # OpenAI embedding
    import os
    from openai import OpenAI
    
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    logger.debug("Embedding text (length=%d) with model=%s", len(text), model)
    
    response = client.embeddings.create(
        input=text,
        model=model,
    )
    
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)


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
