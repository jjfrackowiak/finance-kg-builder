"""Mark a sweep chunk done in sweep_runs/MANIFEST.md and find the next pending
one. Used by the auto-chain step in .github/workflows/sweep.yml -- kept as a
standalone script (not a workflow heredoc) so it can be tested directly and
so YAML block-scalar indentation never conflicts with Python's own
column-0-at-top-level indentation requirement.

Usage: mark_chunk_and_find_next.py <this_run> <run_url>
Prints the next pending run number to stdout (empty if none).
"""
import sys

MANIFEST = "sweep_runs/MANIFEST.md"


def main(this_run: str, run_url: str) -> None:
    lines = open(MANIFEST).read().splitlines()
    next_run = None
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 7:
            continue
        run_id = cols[0]
        if run_id == this_run and ("pending" in cols[2] or "DISPATCHED" in cols[2]):
            cols[2] = "✅ DONE (auto)"
            cols[6] = f"[link]({run_url})"
            lines[i] = "| " + " | ".join(cols) + " |"
        elif next_run is None and cols[2] == "· pending":
            next_run = run_id
            cols[2] = "⏳ DISPATCHED (auto)"
            lines[i] = "| " + " | ".join(cols) + " |"
    open(MANIFEST, "w").write("\n".join(lines) + "\n")
    print(next_run or "", end="")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
