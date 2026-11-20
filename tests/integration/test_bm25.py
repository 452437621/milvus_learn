"""集成：BM25 全文检索（原生 / 手动回退）。"""

from __future__ import annotations

import pytest

from mlcore import fulltext

pytestmark = pytest.mark.integration


@pytest.fixture
def auto_collection(client, docs, embedder):
    return fulltext.create_fulltext_collection(client, "fts_auto", docs, embedder, mode="auto")


def test_auto_mode_and_search(client, auto_collection) -> None:
    assert auto_collection.mode in {"native", "manual"}
    assert auto_collection.count == 25

    hits = fulltext.bm25_search(client, auto_collection, "火锅 麻辣", limit=3)[0]
    assert hits
    assert any(hit["entity"]["category"] == "美食" for hit in hits)


def test_native_scores_are_negative(client, auto_collection) -> None:
    if auto_collection.mode != "native":
        pytest.skip("当前环境回退到手动稀疏")
    hits = fulltext.bm25_search(client, auto_collection, "向量 检索", limit=3)[0]
    assert hits
    assert all(hit["distance"] <= 0 for hit in hits)


def test_manual_mode(client, docs, embedder) -> None:
    collection = fulltext.create_fulltext_collection(
        client, "fts_manual", docs, embedder, mode="manual"
    )
    assert collection.mode == "manual"
    assert collection.analyzer == "none"

    hits = fulltext.bm25_search(client, collection, "火锅 麻辣", limit=3)[0]
    assert hits
    data, metric = fulltext.sparse_request_data(collection, "火锅")
    assert isinstance(data[0], dict)
    assert metric == "IP"


def test_native_forced(client, docs, embedder) -> None:
    try:
        collection = fulltext.create_fulltext_collection(
            client, "fts_native", docs, embedder, mode="native"
        )
    except RuntimeError as exc:
        pytest.skip(f"原生 BM25 不可用：{exc}")
    assert collection.mode == "native"
    assert collection.analyzer in {"jieba", "standard"}
    assert fulltext.bm25_search(client, collection, "睡眠 作息", limit=2)[0]
