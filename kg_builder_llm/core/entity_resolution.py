"""Entity resolution and key normalization utilities."""

import re
from typing import Dict, Set


# Canonical ticker symbols for companies frequently mentioned in financial news.
# Maps every plausible variant (name, alias, old ticker) → canonical ticker.
KNOWN_COMPANY_TICKERS: Dict[str, str] = {
    # Nvidia
    "nvidia": "NVDA", "nvidia corporation": "NVDA", "nvda": "NVDA", "nvidia corp": "NVDA",
    # Intel
    "intel": "INTC", "intel corporation": "INTC", "intc": "INTC",
    # Apple
    "apple": "AAPL", "apple inc": "AAPL", "aapl": "AAPL",
    # Microsoft
    "microsoft": "MSFT", "microsoft corporation": "MSFT", "msft": "MSFT",
    # Amazon
    "amazon": "AMZN", "amazon.com": "AMZN", "amzn": "AMZN",
    # Alphabet / Google
    "alphabet": "GOOGL", "google": "GOOGL", "googl": "GOOGL", "goog": "GOOGL",
    "alphabet inc": "GOOGL",
    # Tesla
    "tesla": "TSLA", "tsla": "TSLA", "tesla inc": "TSLA",
    # Meta
    "meta": "META", "facebook": "META", "fb": "META", "meta platforms": "META",
    # AMD
    "advanced micro devices": "AMD", "amd": "AMD",
    # Taiwan Semiconductor
    "taiwan semiconductor": "TSM", "taiwan semiconductor manufacturing": "TSM",
    "tsmc": "TSM", "tsm": "TSM",
    # Qualcomm
    "qualcomm": "QCOM", "qcom": "QCOM",
    # Broadcom
    "broadcom": "AVGO", "avgo": "AVGO",
    # Salesforce
    "salesforce": "CRM", "crm": "CRM",
    # Oracle
    "oracle": "ORCL", "orcl": "ORCL",
    # IBM
    "ibm": "IBM", "international business machines": "IBM",
    # Netflix
    "netflix": "NFLX", "nflx": "NFLX",
    # Adobe
    "adobe": "ADBE", "adbe": "ADBE",
    # ARM
    "arm": "ARM", "arm holdings": "ARM",
    # VMware
    "vmware": "VMW", "vmw": "VMW",
    # Palantir
    "palantir": "PLTR", "pltr": "PLTR",
    # Snowflake
    "snowflake": "SNOW", "snow": "SNOW",
    # TSMC (duplicate alias)
    "semiconductor manufacturing international": "SMIC",
}

# Legal suffixes stripped before name lookup or slug generation.
_LEGAL_SUFFIX_RE = re.compile(
    r'\b(Inc\.?|Inc|Corporation|Corp\.?|Corp|Company|Co\.?|Co|LLC\.?|LLC'
    r'|Ltd\.?|Ltd|Limited|L\.L\.C\.?|PLLC|PSC|PA|P\.A\.?)'
    r'|,\s*(Inc|Corp|LLC|Ltd|Co|Inc\.|Corp\.|LLC\.|Ltd\.|Co\.)$',
    re.IGNORECASE,
)


def _strip_legal_suffixes(text: str) -> str:
    return _LEGAL_SUFFIX_RE.sub("", text).strip()


def normalize_key(key: str, node_label: str = "") -> str:
    """
    Normalize a node key for consistent entity resolution.

    Rules:
    1. Strip legal suffixes (Inc., Corp., LLC., Ltd., Co., etc.)
    2. For Company / Ticker / Exchange nodes: uppercase (ticker-style)
    3. For everything else: lowercase slug (spaces → underscores, strip specials)

    Args:
        key: The original key/identifier
        node_label: The node label (e.g., "Company", "Sector") for context

    Returns:
        Normalized key suitable for Neo4j MERGE.
    """
    if not key:
        return key

    key = key.strip()
    key = _strip_legal_suffixes(key).strip()

    if node_label.lower() in ("ticker", "company", "exchange") or (
        (len(key) <= 5 and key.isupper())
        or all(c.isupper() or c.isdigit() or c == "_" for c in key)
    ):
        return key.upper()

    key = re.sub(r"\s+", "_", key)
    key = re.sub(r"[^a-zA-Z0-9_\-]", "", key)
    return key.lower()


def canonical_key(llm_key: str, name: str, label: str) -> str:
    """
    Derive the canonical graph key for a node, preferring name-based lookup over
    the raw LLM-generated key.

    For Company nodes the lookup order is:
      1. Name → KNOWN_COMPANY_TICKERS
      2. LLM key → KNOWN_COMPANY_TICKERS
      3. normalize_key(llm_key, label)

    For all other labels the key is derived from the name as a lowercase slug
    (more stable than trusting the LLM's free-form key).

    Args:
        llm_key: Key string produced by the LLM extraction.
        name: Value of the node's ``name`` property.
        label: Node label (e.g. "Company", "Sector", "Index").

    Returns:
        Canonical key string.
    """
    if label.lower() == "company":
        return _resolve_company_key(llm_key, name)

    # For non-Company nodes: slug derived from name when available.
    if name:
        slug = name.lower().strip()
        slug = _strip_legal_suffixes(slug).strip()
        slug = re.sub(r"\s+", "_", slug)
        slug = re.sub(r"[^a-z0-9_]", "", slug)
        if slug:
            return slug

    return normalize_key(llm_key, label)


def _resolve_company_key(llm_key: str, name: str) -> str:
    """Return the canonical ticker for a Company node."""
    # 1. Try name-based lookup (clean name → dict)
    if name:
        name_clean = _strip_legal_suffixes(name.lower().strip())
        if name_clean in KNOWN_COMPANY_TICKERS:
            return KNOWN_COMPANY_TICKERS[name_clean]

    # 2. Try LLM-key-based lookup
    key_clean = _strip_legal_suffixes(llm_key.lower().strip())
    if key_clean in KNOWN_COMPANY_TICKERS:
        return KNOWN_COMPANY_TICKERS[key_clean]

    # 3. Fallback: uppercase normalization of whatever the LLM generated
    return normalize_key(llm_key, "Company")


def get_canonical_key(variants: Set[str], node_label: str = "") -> str:
    """
    Given a set of variant keys for the same entity, return the canonical one.
    Prefers uppercase for tickers, lowercase for names, longest string as tiebreaker.

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
    return max(variants, key=len)


# ---------------------------------------------------------------------------
# Deduplication Cypher queries (executed after each pipeline step)
# ---------------------------------------------------------------------------
#
# IMPORTANT: We must NOT use properties:"combine" in mergeNodes — that turns
# scalar properties (like first_seen DATE) into lists when values differ,
# breaking comparisons such as `n.first_seen <= eval_date`.
#
# Strategy:
#   1. Pre-compute the correct merged values (min first_seen, union of tags).
#   2. Use properties:"discard" to keep the surviving node's existing scalars.
#   3. Explicitly overwrite first_seen and candidate_tags with correct values.
# ---------------------------------------------------------------------------

MERGE_DUPLICATES_BY_NAME_CYPHER = """
MATCH (a)
WHERE a.name IS NOT NULL AND labels(a)[0] <> 'Article' AND labels(a)[0] <> 'Day'
WITH labels(a)[0] AS lbl, toLower(trim(a.name)) AS norm_name, collect(a) AS nodes
WHERE size(nodes) > 1
WITH lbl, norm_name, nodes,
     reduce(m = nodes[0].first_seen, n IN nodes |
         CASE
           WHEN m IS NULL THEN n.first_seen
           WHEN n.first_seen IS NULL THEN m
           WHEN n.first_seen < m THEN n.first_seen
           ELSE m
         END
     ) AS min_first_seen,
     reduce(tags = [], n IN nodes |
         tags + [t IN coalesce(n.candidate_tags, []) WHERE NOT t IN tags]
     ) AS merged_tags
CALL apoc.refactor.mergeNodes(nodes, {properties: "discard", mergeRels: true})
YIELD node
SET node.first_seen = min_first_seen,
    node.candidate_tags = merged_tags
RETURN lbl, norm_name, count(*) AS merged
"""

# Secondary pass: merge by uppercased key (catches NVDA vs nvda case drift).
MERGE_DUPLICATES_BY_KEY_CYPHER = """
MATCH (a)
WHERE a.key IS NOT NULL AND labels(a)[0] <> 'Article' AND labels(a)[0] <> 'Day'
WITH labels(a)[0] AS lbl, toUpper(a.key) AS norm_key, collect(a) AS nodes
WHERE size(nodes) > 1
WITH lbl, norm_key, nodes,
     reduce(m = nodes[0].first_seen, n IN nodes |
         CASE
           WHEN m IS NULL THEN n.first_seen
           WHEN n.first_seen IS NULL THEN m
           WHEN n.first_seen < m THEN n.first_seen
           ELSE m
         END
     ) AS min_first_seen,
     reduce(tags = [], n IN nodes |
         tags + [t IN coalesce(n.candidate_tags, []) WHERE NOT t IN tags]
     ) AS merged_tags
CALL apoc.refactor.mergeNodes(nodes, {properties: "discard", mergeRels: true})
YIELD node
SET node.first_seen = min_first_seen,
    node.candidate_tags = merged_tags
RETURN lbl, norm_key, count(*) AS merged
"""
