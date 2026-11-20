"""corpus 模块：语料加载与校验。"""

from __future__ import annotations

import json

import pytest

from mlcore import corpus

pytestmark = pytest.mark.unit

EXPECTED_CATEGORIES = {"向量数据库", "编程开发", "美食", "旅行", "电影", "健康", "财经"}
REQUIRED_FIELDS = set(corpus.REQUIRED_FIELDS)


def test_load_docs_shape() -> None:
    docs = corpus.load_docs()
    assert len(docs) == 25
    assert [doc["id"] for doc in docs] == list(range(1, 26))
    for doc in docs:
        assert REQUIRED_FIELDS <= set(doc)


def test_categories() -> None:
    assert set(corpus.categories()) == EXPECTED_CATEGORIES
    assert len(corpus.by_category("美食")) == 4


def test_documents_for_embedding_contains_metadata() -> None:
    docs = corpus.load_docs()[:1]
    text = corpus.documents_for_embedding(docs)[0]
    assert docs[0]["title"] in text
    assert docs[0]["text"] in text
    assert docs[0]["tags"][0] in text


def _patch_path(monkeypatch, tmp_path, payload) -> None:
    path = tmp_path / "docs.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(corpus, "DOCS_PATH", path)


def test_malformed_json(monkeypatch, tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(corpus, "DOCS_PATH", path)
    with pytest.raises(json.JSONDecodeError):
        corpus.load_docs()


def test_empty_list(monkeypatch, tmp_path) -> None:
    _patch_path(monkeypatch, tmp_path, [])
    with pytest.raises(ValueError):
        corpus.load_docs()


def test_non_list_payload(monkeypatch, tmp_path) -> None:
    _patch_path(monkeypatch, tmp_path, {"id": 1})
    with pytest.raises(ValueError):
        corpus.load_docs()


def test_missing_field(monkeypatch, tmp_path) -> None:
    doc = dict(corpus.load_docs()[0])
    doc.pop("rating")
    _patch_path(monkeypatch, tmp_path, [doc])
    with pytest.raises(ValueError, match="rating"):
        corpus.load_docs()


def test_duplicate_ids(monkeypatch, tmp_path) -> None:
    docs = [dict(doc) for doc in corpus.load_docs()[:2]]
    docs[1]["id"] = docs[0]["id"]
    _patch_path(monkeypatch, tmp_path, docs)
    with pytest.raises(ValueError, match="重复"):
        corpus.load_docs()


def test_missing_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(corpus, "DOCS_PATH", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError):
        corpus.load_docs()
