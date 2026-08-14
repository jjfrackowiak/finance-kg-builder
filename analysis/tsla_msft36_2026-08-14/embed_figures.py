"""Inline every fig_*.png into the two-ticker report as a data URI.

The report is self-contained, so figures live in the HTML rather than beside it.
Placeholders are written as `data:image/png;base64,FIG:<filename>` and replaced here,
which keeps the report editable by hand and makes re-embedding idempotent: run
`make_figures.py` then this, and the report picks up the new plots.

Re-embedding an already-embedded report is a no-op — the placeholders are gone. To
refresh, restore the placeholders (git checkout) or edit the payload directly.
"""

import base64
import re
from pathlib import Path

HERE = Path(__file__).parent
REPORT = HERE.resolve().parents[1] / "analyses" / "tsla_msft_200days_report.html"


def main() -> None:
    html = REPORT.read_text(encoding="utf-8")
    names = re.findall(r"data:image/png;base64,FIG:([\w.\-]+)", html)
    if not names:
        raise SystemExit(f"no FIG: placeholders left in {REPORT.name} — already embedded?")

    for name in names:
        png = HERE / name
        if not png.exists():
            raise SystemExit(f"missing figure: {png}")
        payload = base64.b64encode(png.read_bytes()).decode("ascii")
        html = html.replace(f"data:image/png;base64,FIG:{name}",
                            f"data:image/png;base64,{payload}")
        print(f"embedded {name:28s} {len(payload):>9,} base64 chars")

    REPORT.write_text(html, encoding="utf-8")
    print(f"\nwrote {REPORT}  ({len(html):,} chars, {len(names)} figures)")


if __name__ == "__main__":
    main()
