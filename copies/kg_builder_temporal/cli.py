"""Command-line interface for kg_builder."""

import sys

from kg_builder_temporal.main import main

if __name__ == "__main__":
    exit_code = __import__("asyncio").run(main())
    sys.exit(exit_code)
