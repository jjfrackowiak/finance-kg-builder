"""Mark a sweep chunk done in sweep_runs/MANIFEST.md and find the next pending
one. Used by the auto-chain step in .github/workflows/sweep.yml -- kept as a
standalone script (not a workflow heredoc) so it can be tested directly and
so YAML block-scalar indentation never conflicts with Python's own
column-0-at-top-level indentation requirement.

Columns are resolved by HEADER NAME, never by fixed position. They were
hardcoded (status=2, gha=6) against an older 6-column table; once `Ticker`
and `Steps` were inserted the indices silently pointed at `Steps` and
`~Time` instead. Nothing raised -- the status cell read "3", so no row ever
matched "pending", the current chunk was never marked done, and next_run
stayed None. The workflow then reported "no more pending chunks" and exited
0, which is indistinguishable from a finished sweep. The chain would have
stopped dead after the first chunk.

Usage: mark_chunk_and_find_next.py <this_run> <run_url>
Prints the next pending run number to stdout (empty if none).
"""
import sys

MANIFEST = "sweep_runs/MANIFEST.md"


def split_row(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_separator(cols: list) -> bool:
    return all(set(c) <= set("-: ") and c for c in cols)


def find_columns(lines: list) -> tuple:
    """Return (header_index, {name: position}) for the run manifest table."""
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cols = [c.lower() for c in split_row(line)]
        if "run" in cols and "status" in cols:
            return i, {name: cols.index(name) for name in ("run", "status", "gha")
                       if name in cols}
    raise SystemExit("MANIFEST.md: could not find a table header with Run and Status columns")


def main(this_run: str, run_url: str) -> None:
    lines = open(MANIFEST).read().splitlines()
    header_i, idx = find_columns(lines)
    for required in ("run", "status"):
        if required not in idx:
            raise SystemExit(f"MANIFEST.md: no '{required}' column in the table header")
    i_run, i_status = idx["run"], idx["status"]
    i_gha = idx.get("gha")
    width = len(split_row(lines[header_i]))

    next_run = None
    marked_done = False
    for i, line in enumerate(lines):
        if i <= header_i or not line.startswith("|"):
            continue
        cols = split_row(line)
        if len(cols) != width or is_separator(cols):
            continue
        status = cols[i_status]
        if cols[i_run] == this_run and ("pending" in status or "DISPATCHED" in status):
            cols[i_status] = "✅ DONE (auto)"
            if i_gha is not None:
                cols[i_gha] = f"[link]({run_url})"
            lines[i] = "| " + " | ".join(cols) + " |"
            marked_done = True
        elif next_run is None and status == "· pending":
            next_run = cols[i_run]
            cols[i_status] = "⏳ DISPATCHED (auto)"
            lines[i] = "| " + " | ".join(cols) + " |"

    if not marked_done:
        # Loud on stderr, not fatal: the chain should still advance even if the
        # row for this run was edited by hand mid-flight. Silence here is what
        # made the column drift undetectable in the first place.
        print(f"warning: no pending/dispatched row matched run '{this_run}'",
              file=sys.stderr)

    open(MANIFEST, "w").write("\n".join(lines) + "\n")
    print(next_run or "", end="")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
