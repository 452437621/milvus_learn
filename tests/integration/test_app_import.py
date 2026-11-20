"""集成：导入 app.gradio_search 不应产生数据库副作用（重构回归护栏）。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


def test_import_does_not_create_database(tmp_path) -> None:
    env = dict(os.environ)
    env.update(
        {
            "MILVUS_DB_DIR": str(tmp_path),
            "MILVUS_URI": "",
            "PYTHONPATH": str(ROOT),
            "GRADIO_ANALYTICS_ENABLED": "False",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", "import app.gradio_search as m; print(sorted(m.__all__))"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "make_callbacks" in result.stdout
    assert not (tmp_path / "runs" / "ui").exists()
