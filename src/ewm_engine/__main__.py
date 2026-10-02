"""Package main entrypoint enabling `python -m ewm_engine`."""

from __future__ import annotations

import sys

from ewm_engine.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
