"""``python -m evals`` -- delegates to the TREC inspector CLI.

Kept to a delegation so there is one implementation of the CLI:
``python -m evals --topic 2021 --first`` and
``python -m evals.trec --topic 2021 --first`` behave identically.
"""

from __future__ import annotations

from evals.trec import main

if __name__ == "__main__":
    raise SystemExit(main())
