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
