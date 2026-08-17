from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 10),
    reason="regression targets the minimum supported Python 3.10 runtime",
)
def test_python310_imports_and_parses_repository_config() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = """
from pathlib import Path
from docprune.config import load_config

config = load_config(Path("configs/docprune-m3docvqa.toml"))
assert tuple(sorted(config.page_settings)) == (1, 2, 4)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
