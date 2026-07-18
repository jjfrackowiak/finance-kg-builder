"""Deterministic identifiers shared across the pipeline."""

import hashlib


def article_text_id(text: str) -> str:
    """Deterministic Article node id derived from the article text.

    Python's built-in ``hash()`` is salted per process (PYTHONHASHSEED), so it
    cannot be used to match Article nodes across runs or processes. All code
    that identifies an Article by its text must go through this function.

    Args:
        text: Full article text

    Returns:
        Hex-encoded SHA-256 digest of the UTF-8 encoded text
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
