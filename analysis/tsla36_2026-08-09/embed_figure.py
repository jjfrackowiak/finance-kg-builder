"""Replace one base64 <img> in the report with a PNG from this directory.

The report is self-contained — figures are inlined as data URIs — so updating a plot
means swapping the base64 payload rather than a file path. Figures are addressed by
their 0-based order of appearance.

Usage: python embed_figure.py <figure_index> <png_filename>
"""

import base64
import re
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parents[2] / "analyses" / "tsla_200days_report.html"


def main() -> None:
    index, png = int(sys.argv[1]), Path(__file__).parent / sys.argv[2]
    html = REPORT.read_text(encoding="utf-8")
    spans = [m.span(1) for m in re.finditer(r'<img src="data:image/png;base64,([^"]+)"', html)]
    if not 0 <= index < len(spans):
        raise SystemExit(f"figure {index} out of range — report has {len(spans)}")

    start, end = spans[index]
    payload = base64.b64encode(png.read_bytes()).decode("ascii")
    REPORT.write_text(html[:start] + payload + html[end:], encoding="utf-8")
    print(f"figure {index}: {end - start} -> {len(payload)} base64 chars from {png.name}")


if __name__ == "__main__":
    main()
