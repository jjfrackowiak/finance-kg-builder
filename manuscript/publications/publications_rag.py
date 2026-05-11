#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mcp[cli]>=1.0",
#   "pymupdf>=1.24",
#   "openai>=1.0",
#   "python-dotenv>=1.0",
#   "numpy>=1.24",
#   "rank-bm25>=0.2",
# ]
# ///
"""RAG MCP over manuscript/publications/*.pdf.

Chunking: paragraph-level (~600 chars), one embedding per paragraph.
Storage:  SQLite (text + OpenAI text-embedding-3-small BLOBs).
Search:   cosine similarity; BM25 fallback when no API key.
"""

import json
import logging
import os
import re
import sqlite3
from pathlib import Path

import fitz
import numpy as np
from mcp.server.fastmcp import FastMCP

PUBLICATIONS_DIR = Path(__file__).parent
DB_PATH = PUBLICATIONS_DIR / ".rag_index.db"
LOG_PATH = PUBLICATIONS_DIR / "publications_rag.log"
ENV_PATH = Path(__file__).parent.parent.parent / ".env"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
CHUNK_MAX = 700   # target max chars per chunk
CHUNK_MIN = 120   # discard shorter fragments

# ── logging ──────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    PUBLICATIONS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("publications_rag")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    eh = logging.StreamHandler()
    eh.setLevel(logging.WARNING)
    eh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(eh)
    return logger

log = _setup_logging()

mcp = FastMCP("publications-rag")
log.info("publications-rag MCP server initialised")


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_api_key() -> str | None:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("OPENAI_API_KEY")


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id        TEXT PRIMARY KEY,
            source    TEXT NOT NULL,
            bib_key   TEXT NOT NULL,
            page      INTEGER NOT NULL,
            mtime     REAL NOT NULL,
            text      TEXT NOT NULL,
            embedding BLOB
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)")
    con.commit()
    cols = {r[1] for r in con.execute("PRAGMA table_info(chunks)")}
    if "embedding" not in cols:
        con.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
        con.commit()
    return con


def _bib_key(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    return parts[0] + parts[1] + "".join(parts[2:]) if len(parts) >= 2 else stem


def _split_paragraphs(text: str) -> list[str]:
    """Split page text into paragraph-sized chunks (~CHUNK_MAX chars)."""
    raw = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for para in raw:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= CHUNK_MAX:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # paragraph itself too long — split at sentence boundaries
            if len(para) > CHUNK_MAX:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= CHUNK_MAX:
                        current = (current + " " + sent).strip() if current else sent
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
            else:
                current = para
    if current:
        if len(current) >= CHUNK_MIN:
            chunks.append(current)
        elif chunks:
            chunks[-1] += "\n\n" + current
    return [c for c in chunks if len(c) >= CHUNK_MIN]


def _index_pdf(con: sqlite3.Connection, path: Path) -> int:
    fname = path.name
    mtime = path.stat().st_mtime
    bk = _bib_key(fname)
    log.debug("Indexing %s (bib_key=%s)", fname, bk)
    try:
        doc = fitz.open(str(path))
    except Exception:
        log.exception("Failed to open PDF: %s", path)
        return 0
    rows = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if not text:
            continue
        for ci, chunk in enumerate(_split_paragraphs(text)):
            rows.append((f"{fname}::p{page_num}::c{ci}", fname, bk, page_num, mtime, chunk))
    doc.close()
    if rows:
        con.executemany(
            "INSERT OR REPLACE INTO chunks(id,source,bib_key,page,mtime,text) VALUES(?,?,?,?,?,?)",
            rows,
        )
        con.commit()
    log.info("Indexed %s — %d chunks", fname, len(rows))
    return len(rows)


def _embed_pending(con: sqlite3.Connection, api_key: str) -> int:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    rows = con.execute("SELECT id, text FROM chunks WHERE embedding IS NULL").fetchall()
    if not rows:
        log.debug("No pending chunks to embed")
        return 0
    log.info("Embedding %d chunks in batches of %d", len(rows), BATCH_SIZE)
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]
        try:
            resp = client.embeddings.create(input=texts, model=EMBED_MODEL)
        except Exception:
            log.exception("OpenAI embedding request failed (batch starting at %d)", start)
            raise
        for chunk_id, emb_obj in zip(ids, resp.data):
            vec = np.array(emb_obj.embedding, dtype=np.float32).tobytes()
            con.execute("UPDATE chunks SET embedding=? WHERE id=?", (vec, chunk_id))
        con.commit()
        log.debug("Embedded batch %d–%d", start, start + len(batch) - 1)
    log.info("Finished embedding %d chunks", len(rows))
    return len(rows)


def _cosine_search(con, query_vec: np.ndarray, files: list[str] | None, n: int) -> list[dict]:
    if files:
        ph = ",".join("?" * len(files))
        rows = con.execute(
            f"SELECT source, bib_key, page, text, embedding FROM chunks"
            f" WHERE source IN ({ph}) AND embedding IS NOT NULL",
            files,
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT source, bib_key, page, text, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()
    if not rows:
        return []
    matrix = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sims = (matrix / norms) @ (query_vec / (np.linalg.norm(query_vec) or 1))
    top = np.argsort(sims)[::-1][:n]
    return [
        {"bib_key": rows[i][1], "source": rows[i][0], "page": rows[i][2],
         "score": round(float(sims[i]), 4), "text": rows[i][3]}
        for i in top if sims[i] > 0
    ]


def _bm25_search(con, query: str, files: list[str] | None, n: int) -> list[dict]:
    from rank_bm25 import BM25Okapi
    if files:
        ph = ",".join("?" * len(files))
        rows = con.execute(
            f"SELECT source, bib_key, page, text FROM chunks WHERE source IN ({ph})", files
        ).fetchall()
    else:
        rows = con.execute("SELECT source, bib_key, page, text FROM chunks").fetchall()
    if not rows:
        return []
    tokenize = lambda t: re.findall(r"[a-z0-9]+", t.lower())
    scores = BM25Okapi([tokenize(r[3]) for r in rows]).get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [
        {"bib_key": rows[i][1], "source": rows[i][0], "page": rows[i][2],
         "score": round(float(scores[i]), 3), "text": rows[i][3]}
        for i in top if scores[i] > 0
    ]


# ── MCP tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def sync() -> str:
    """Sync index: index new/modified PDFs, drop removed, embed unembedded chunks via OpenAI."""
    log.info("sync() started")
    try:
        con = _db()
    except Exception:
        log.exception("Failed to open database at %s", DB_PATH)
        raise

    disk = {f.name: f for f in PUBLICATIONS_DIR.glob("*.pdf")}
    indexed = {r[0]: r[1] for r in con.execute(
        "SELECT source, MAX(mtime) FROM chunks GROUP BY source"
    ).fetchall()}
    log.info("Found %d PDFs on disk, %d sources in index", len(disk), len(indexed))

    removed: list[str] = []
    for fname in set(indexed) - set(disk):
        con.execute("DELETE FROM chunks WHERE source=?", (fname,))
        removed.append(fname)
        log.info("Removed stale entry: %s", fname)
    con.commit()

    added: list[dict] = []
    reindexed: list[str] = []
    for fname, fpath in sorted(disk.items()):
        if fname not in indexed:
            pages = _index_pdf(con, fpath)
            added.append({"file": fname, "bib_key": _bib_key(fname), "chunks": pages})
        elif fpath.stat().st_mtime > indexed[fname]:
            log.info("Re-indexing modified file: %s", fname)
            con.execute("DELETE FROM chunks WHERE source=?", (fname,))
            con.commit()
            _index_pdf(con, fpath)
            reindexed.append(fname)

    api_key = _load_api_key()
    if api_key:
        try:
            embedded = _embed_pending(con, api_key)
        except Exception:
            log.exception("Embedding step failed; index text-only")
            embedded = 0
        embed_note = f"{embedded} chunks embedded"
    else:
        log.warning("OPENAI_API_KEY not found — BM25 fallback active")
        embedded = 0
        embed_note = "no OPENAI_API_KEY — BM25 fallback active"

    con.close()
    unchanged = len(disk) - len(added) - len(reindexed)
    log.info("sync() done — added=%d reindexed=%d removed=%d unchanged=%d embedded=%d",
             len(added), len(reindexed), len(removed), unchanged, embedded)
    return json.dumps(
        {"added": added, "reindexed": reindexed, "removed": removed,
         "unchanged": unchanged, "embeddings": embed_note},
        indent=2,
    )


@mcp.tool()
def search(query: str, files: list[str] | None = None, n: int = 10) -> str:
    """Semantic search (cosine) with BM25 fallback. Pass files=[...] to restrict to specific PDFs."""
    log.info("search() query=%r files=%s n=%d", query[:80], files, n)
    con = _db()
    if con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0:
        con.close()
        log.warning("search() called on empty index")
        return "Index is empty — call sync() first."

    api_key = _load_api_key()
    try:
        if api_key and con.execute("SELECT 1 FROM chunks WHERE embedding IS NOT NULL LIMIT 1").fetchone():
            from openai import OpenAI
            resp = OpenAI(api_key=api_key).embeddings.create(input=[query], model=EMBED_MODEL)
            qvec = np.array(resp.data[0].embedding, dtype=np.float32)
            hits = _cosine_search(con, qvec, files, n)
            method = "cosine"
        else:
            hits = _bm25_search(con, query, files, n)
            method = "bm25 (no embeddings)" if not api_key else "bm25 (run sync to embed)"
    except Exception:
        log.exception("search() failed for query=%r", query[:80])
        con.close()
        raise

    con.close()
    log.info("search() returned %d hits via %s", len(hits), method)
    return json.dumps({"method": method, "hits": hits}, indent=2)


@mcp.tool()
def get_page(filename: str, page: int) -> str:
    """Full concatenated text of a specific page — use to verify exact wording before citing."""
    log.debug("get_page() %s p%d", filename, page)
    con = _db()
    rows = con.execute(
        "SELECT text FROM chunks WHERE source=? AND page=? ORDER BY id", (filename, page)
    ).fetchall()
    con.close()
    if not rows:
        log.warning("get_page(): not found — %s p%d", filename, page)
    return "\n\n".join(r[0] for r in rows) if rows else f"Not found: {filename} p{page}"


@mcp.tool()
def list_indexed() -> str:
    """All indexed files with chunk counts, page counts, bib keys, and embedding status."""
    con = _db()
    rows = con.execute("""
        SELECT source, bib_key,
               COUNT(*) AS chunks,
               COUNT(DISTINCT page) AS pages,
               SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded
        FROM chunks GROUP BY source ORDER BY source
    """).fetchall()
    con.close()
    if not rows:
        return "Index is empty — call sync() first."
    return json.dumps(
        [{"file": r[0], "bib_key": r[1], "chunks": r[2], "pages": r[3], "embedded": r[4]}
         for r in rows],
        indent=2,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description="publications-rag CLI")
        sub = parser.add_subparsers(dest="cmd", required=True)

        sub.add_parser("sync", help="Index new/modified PDFs and embed chunks")
        sub.add_parser("list", help="List all indexed files")

        p_search = sub.add_parser("search", help="Search the index")
        p_search.add_argument("query")
        p_search.add_argument("--files", nargs="*", help="Restrict to these filenames")
        p_search.add_argument("--n", type=int, default=10)

        p_page = sub.add_parser("get-page", help="Get full text of a page")
        p_page.add_argument("filename")
        p_page.add_argument("page", type=int)

        args = parser.parse_args()
        if args.cmd == "sync":
            print(sync())
        elif args.cmd == "list":
            print(list_indexed())
        elif args.cmd == "search":
            print(search(args.query, files=args.files, n=args.n))
        elif args.cmd == "get-page":
            print(get_page(args.filename, args.page))
    else:
        mcp.run()
