"""Entity resolution and key normalization utilities."""

import re
from typing import Dict, Set


def normalize_key(key: str, node_label: str = "") -> str:
    """
    Normalize a node key for consistent entity resolution.
    
    Rules:
    1. Convert to uppercase for ticker symbols and codes
    2. Strip legal suffixes (Inc., Corp., LLC., Ltd., Co., etc.)
    3. Strip whitespace and special characters
    4. Replace spaces with underscores for multi-word keys
    5. For entities, preserve case but remove extra spaces
    
    Args:
        key: The original key/identifier
        node_label: The node label (e.g., "Company", "Ticker") for context-specific normalization
    
    Returns:
        Normalized key suitable for deduplication
    """
    if not key:
        return key
    
    # Strip whitespace
    key = key.strip()
    
    # Legal suffixes to remove (case-insensitive)
    legal_suffixes = [
        r'\b(Inc\.?|Inc|Corporation|Corp\.?|Corp|Company|Co\.?|Co|LLC\.?|LLC|Ltd\.?|Ltd|Limited|L\.L\.C\.?|PLLC|PSC|PA|P\.A\.?)',
        r',\s*(' + '|'.join(['Inc', 'Corp', 'LLC', 'Ltd', 'Co', 'Inc.', 'Corp.', 'LLC.', 'Ltd.', 'Co.']) + r')$',
    ]
    
    for pattern in legal_suffixes:
        key = re.sub(pattern, '', key, flags=re.IGNORECASE)
    
    key = key.strip()
    
    # For ticker symbols and codes (all caps, short), normalize to uppercase
    if node_label.lower() in ["ticker", "company", "exchange"] or (
        len(key) <= 5 and key.isupper() or all(c.isupper() or c.isdigit() or c == "_" for c in key)
    ):
        # Already ticker-like, uppercase
        return key.upper()
    
    # For longer names, normalize whitespace and lowercase
    # Replace multiple spaces with single space
    key = re.sub(r'\s+', '_', key.strip())
    
    # Remove special characters except underscores, hyphens, and alphanumerics
    key = re.sub(r'[^a-zA-Z0-9_\-]', '', key)
    
    # Convert to lowercase for name matching
    return key.lower()


def get_canonical_key(variants: Set[str], node_label: str = "") -> str:
    """
    Given a set of variant keys for the same entity, return the canonical one.
    Prefers: uppercase for tickers, lowercase for names, longest string as tiebreaker.
    
    Args:
        variants: Set of variant keys
        node_label: Node label for context
    
    Returns:
        Canonical key to use
    """
    if not variants:
        return ""
    
    if len(variants) == 1:
        return list(variants)[0]
    
    # Normalize all variants
    normalized = {v: normalize_key(v, node_label) for v in variants}
    
    # Find the original that maps to each normalized key
    # Prefer longest original variant as most complete
    return max(variants, key=len)


# Cypher query to merge duplicate nodes by normalized key
MERGE_DUPLICATES_CYPHER = """
MATCH (n)
WHERE n.key IS NOT NULL
WITH labels(n)[0] as label, n.key as orig_key, 
     apoc.text.upperCase(n.key) as normalized_key,
     collect(n) as nodes,
     count(*) as cnt
WHERE cnt > 1
CALL apoc.refactor.mergeNodes(nodes, {properties: "combine", relationships: "combine"})
YIELD node
RETURN label, orig_key, count(*) as merged
"""

# Alternative: Merge by computing normalized key per node type
MERGE_BY_NORMALIZED_KEY_CYPHER = """
MATCH (n:Entity)
WHERE n.name IS NOT NULL
WITH toLower(trim(n.name)) as normalized_name, 
     collect(n) as nodes, count(*) as cnt
WHERE cnt > 1
CALL apoc.refactor.mergeNodes(nodes, {properties: "combine", relationships: "combine"})
YIELD node
RETURN normalized_name, count(*) as merged
"""
