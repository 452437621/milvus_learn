"""端到端：逐个运行 11 个 demo，校验退出码与关键输出。"""

from __future__ import annotations

import subprocess
import sys

import pytest

import scripts.run_all as run_all
from tests.support import ROOT, clean, e2e_env

pytestmark = pytest.mark.e2e

EXPECTED = {
    "00_environment_check.py": ["Demo 00：环境自检", "能力矩阵", "BM25 全文检索"],
    "01_hello_milvus.py": ["Demo 01：Hello Milvus", "实体数量", "索引"],
    "02_embeddings_and_similarity.py": [
        "Demo 02：Embedding 与相似度",
        "两次结果一致",
        "暴力 Top-5",
    ],
    "03_crud.py": ["Demo 03：CRUD 与 Schema", "upsert 后数量", "删除后数量"],
    "04_schema_and_index.py": ["Demo 04：索引类型对比", "索引描述", "排名重合度"],
    "05_scalar_filter.py": ["Demo 05：标量过滤", "命中", "命中分类集合"],
    "06_partitions_aliases.py": ["Demo 06：分区与别名", "分区列表", "切换后", "删除别名后"],
    "07_sparse_bm25.py": ["Demo 07：稀疏向量与 BM25", "BM25 检索", "分析器"],
    "08_hybrid_search.py": ["Demo 08：混合检索与融合排序", "观察点"],
    "09_search_advanced.py": ["Demo 09：高级搜索", "迭代总数", "类目是否唯一", "两页 id 是否重叠"],
    "10_mini_rag.py": ["Demo 10：Mini RAG", "chunk 数", "离线抽取式回答"],
}


@pytest.mark.parametrize(
    "filename,extra_args", run_all.DEMOS, ids=[name for name, _ in run_all.DEMOS]
)
def test_demo_runs(filename, extra_args, e2e_root) -> None:
    data_dir = e2e_root / filename.replace(".py", "")
    command = run_all.demo_command(filename, extra_args)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=e2e_env(data_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    output = clean(result.stdout + result.stderr)
    assert result.returncode == 0, output[-3000:]
    assert "Traceback" not in output, output[-3000:]
    for marker in EXPECTED[filename]:
        assert marker in output, f"{filename} 缺少标记 {marker}"


def test_demo_07_mode_marker(e2e_root) -> None:
    result = subprocess.run(
        run_all.demo_command("07_sparse_bm25.py", []),
        cwd=ROOT,
        env=e2e_env(e2e_root / "demo07"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    output = clean(result.stdout + result.stderr)
    assert result.returncode == 0
    assert "native" in output or "manual" in output


def test_demo_08_fusion_marker(e2e_root) -> None:
    result = subprocess.run(
        run_all.demo_command("08_hybrid_search.py", []),
        cwd=ROOT,
        env=e2e_env(e2e_root / "demo08"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    output = clean(result.stdout + result.stderr)
    assert result.returncode == 0
    assert ("RRF 融合" in output) or ("手动 RRF" in output)


def test_script_entry_points_importable() -> None:
    assert sys.version_info >= (3, 10)
    assert len(run_all.DEMOS) == 11
