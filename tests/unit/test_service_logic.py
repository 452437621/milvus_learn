"""service 的纯逻辑与架构护栏（不启动 Milvus / Gradio）。"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from mlcore.service import INDEX_SPECS, build_expression

pytestmark = pytest.mark.unit


def test_build_expression_empty() -> None:
    assert build_expression("全部", 0, "") == ""
    assert build_expression("", 0, "   ") == ""


def test_build_expression_combined() -> None:
    expression = build_expression("美食", 4.5, " year >= 2020 ")
    assert expression == 'category == "美食" and rating >= 4.50 and (year >= 2020)'


def test_build_expression_raw_only() -> None:
    assert build_expression("全部", 0, "is_free == true") == "(is_free == true)"


def test_index_specs_complete() -> None:
    assert set(INDEX_SPECS) == {"COSINE + HNSW", "L2 + FLAT（精确）", "IP + HNSW"}
    for spec in INDEX_SPECS.values():
        assert {
            "suffix",
            "index_type",
            "metric",
            "index_params",
            "search_params",
            "direction",
        } <= set(spec)


def test_service_module_does_not_import_gradio_or_pymilvus() -> None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    code = (
        "import sys; import mlcore.service; "
        "assert 'gradio' not in sys.modules, 'gradio imported'; "
        "assert 'pymilvus' not in sys.modules, 'pymilvus imported'; "
        "print('ok')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = root
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
