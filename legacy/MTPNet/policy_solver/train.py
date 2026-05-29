"""CLI entrypoint for the path policy solver.

The implementation currently delegates to ``mtpnet.chunk_policy``. Keeping this
module separate gives us a stable public command while the research internals
continue to move.
"""

from __future__ import annotations

from mtpnet.chunk_policy import main


if __name__ == "__main__":
    main()
