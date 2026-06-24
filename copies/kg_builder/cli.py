"""Command-line interface for kg_builder."""

import sys

from copies.kg_builder.main import main

if __name__ == "__main__":
    exit_code = __import__("asyncio").run(main())
    sys.exit(exit_code)
