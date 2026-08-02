"""
Shared pytest fixtures/helpers.

This project has no separate test fixtures or mocked data -- every test runs
against the same real, cached artifacts the rest of the project uses (raw
CSVs, graph pickle, feature parquet, model predictions), consistent with how
this whole codebase operates. Tests that need an artifact which hasn't been
generated yet (e.g. a fresh clone before running the pipeline) skip with a
clear message pointing at the command to run, rather than failing opaquely
or silently passing on fabricated data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def skip_if_missing(path: Path, hint: str):
    if not path.exists():
        pytest.skip(f"{path} not found -- run `{hint}` first")
