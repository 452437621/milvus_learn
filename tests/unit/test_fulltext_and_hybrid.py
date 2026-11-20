"""fulltext 行构造 / 查询分支，以及 hybrid.manual_rrf。"""

from __future__ import annotations

import pytest

from mlcore import corpus, fulltext
from mlcore.embedding import PseudoEmbedder
from mlcore.fulltext import FullTextCollection
from mlcore.hybrid import manual_rrf

pytestmark = pytest.mark.unit


def test_bm25_rows_native_has_no_sparse() -> None:
    docs = corpus.load_docs()[:2]
    rows = fulltext._bm25_rows(docs, PseudoEmbedder(dim=8), manual=False)
    assert len(rows) == 2
    assert "sparse" not in rows[0]
    assert rows[0]["text"] == f"{docs[0]['title']}。{docs[0]['text']}"
    assert len(rows[0]["dense"]) == 8


def test_bm25_rows_manual_has_sparse() -> None:
    docs = corpus.load_docs()[:2]
    rows = fulltext._bm25_rows(docs, PseudoEmbedder(dim=8), manual=True)
    assert rows[0]["sparse"]
    assert all(isinstance(key, int) for key in rows[0]["sparse"])


def test_sparse_request_data_native() -> None:
    collection = FullTextCollection(name="x", mode="native", analyzer="jieba", count=1)
    data, metric = fulltext.sparse_request_data(collection, "向量 检索")
    assert data == ["向量 检索"]
    assert metric == "BM25"


def test_sparse_request_data_manual() -> None:
    collection = FullTextCollection(name="x", mode="manual", analyzer="none", count=1)
    data, metric = fulltext.sparse_request_data(collection, "向量 检索")
    assert isinstance(data[0], dict) and data[0]
    assert metric == "IP"


class _RecordingClient:
    def __init__(self):
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [[]]


def test_bm25_search_native_passes_text() -> None:
    client = _RecordingClient()
    collection = FullTextCollection(name="c", mode="native", analyzer="jieba", count=1)
    fulltext.bm25_search(client, collection, "查询词", limit=3)
    call = client.calls[0]
    assert call["data"] == ["查询词"]
    assert call["anns_field"] == "sparse"
    assert "search_params" not in call


def test_bm25_search_manual_passes_dict_and_ip() -> None:
    client = _RecordingClient()
    collection = FullTextCollection(name="c", mode="manual", analyzer="none", count=1)
    fulltext.bm25_search(client, collection, "查询词", limit=3)
    call = client.calls[0]
    assert isinstance(call["data"][0], dict)
    assert call["search_params"]["metric_type"] == "IP"


def _hit(pk: int) -> dict:
    return {"id": pk, "distance": 0.5, "entity": {"title": f"T{pk}"}}


def test_manual_rrf_merges_and_sorts() -> None:
    dense = [_hit(1), _hit(2)]
    sparse = [_hit(2), _hit(3)]
    fused = manual_rrf(dense, sparse, k=60, limit=5)
    assert [hit["id"] for hit in fused] == [2, 1, 3]
    scores = [hit["distance"] for hit in fused]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0]["entity"] == {"title": "T2"}


def test_manual_rrf_limit_and_missing_ids() -> None:
    fused = manual_rrf([_hit(1), _hit(2), {"entity": {}}], [_hit(2)], limit=1)
    assert len(fused) == 1
    assert fused[0]["id"] == 2
    assert fused[0]["distance"] == pytest.approx(1 / 62 + 1 / 61)
