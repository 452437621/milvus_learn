"""端到端夹具：临时数据目录自动清理。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def e2e_root() -> Path:
    path = Path(tempfile.mkdtemp(prefix="milvus_e2e_")).resolve()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
