"""项目一致性：README 链接、文档与 demo 清单、语料条数、忽略规则。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mlcore import corpus

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
DOC_LINK = re.compile(r"\((docs/[^)]+\.md)\)")


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_links_exist() -> None:
    links = DOC_LINK.findall(_readme_text())
    assert links, "README 中应至少引用一个 docs 文档"
    missing = [link for link in links if not (ROOT / link).exists()]
    assert missing == []


def test_every_doc_is_linked() -> None:
    linked = {Path(link).name for link in DOC_LINK.findall(_readme_text())}
    docs = {path.name for path in (ROOT / "docs").glob("*.md")}
    assert docs - linked == set(), "存在未在 README 中引用的文档"


def test_corpus_count_matches_readme() -> None:
    assert len(corpus.load_docs()) == 25
    assert "25 条" in _readme_text()


def test_gitignore_covers_generated_artifacts() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in ("htmlcov/", ".coverage", "data/runs/", ".ruff_cache/", ".pytest_cache/"):
        assert entry in text, entry


def _split_table_cells(line: str) -> list[str]:
    """按 GFM 规则切分表格行：行内代码里的 | 也是分隔符，除非转义成 \\|。"""
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]

    cells: list[str] = []
    buffer = ""
    escaped = False
    for char in row:
        if escaped:
            buffer += char
            escaped = False
        elif char == "\\":
            buffer += char
            escaped = True
        elif char == "|":
            cells.append(buffer)
            buffer = ""
        else:
            buffer += char
    cells.append(buffer)
    return cells


def _broken_tables(path: Path) -> list[str]:
    """返回列数与表头不一致的表格行（跳过代码块）。"""
    problems: list[str] = []
    header_cells: int | None = None
    in_fence = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            count = len(_split_table_cells(stripped))
            if header_cells is None:
                header_cells = count
            elif count != header_cells:
                problems.append(f"{path.name}:{number} 表头 {header_cells} 列，实际 {count} 列")
        else:
            header_cells = None
    return problems


def test_markdown_tables_have_consistent_columns() -> None:
    # 表格里行内代码中的 | 会被当成分隔符，必须写成 \|（如余弦公式）。
    files = [README, *sorted((ROOT / "docs").glob("*.md"))]
    problems = [problem for path in files for problem in _broken_tables(path)]
    assert problems == [], "\n".join(problems)
