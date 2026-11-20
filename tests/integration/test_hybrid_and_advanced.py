"""集成：混合检索与高级搜索（分页/范围/分组/迭代器）。"""

from __future__ import annotations

import pytest
from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

from mlcore import demo_data, fulltext
from mlcore.hybrid import manual_rrf

pytestmark = pytest.mark.integration

QUERY = "向量 检索 选型"


@pytest.fixture
def hybrid_setup(client, docs, embedder):
    collection = fulltext.create_fulltext_collection(
        client, "hybrid_docs", docs, embedder, mode="auto"
    )
    dense = client.search(
        collection_name=collection.name,
        data=[embedder.embed_query(QUERY)],
        anns_field="dense",
        limit=5,
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        output_fields=["title", "category"],
    )[0]
    sparse = fulltext.bm25_search(client, collection, QUERY, limit=5)[0]
    return collection, dense, sparse, embedder


def _request_dense(embedder) -> AnnSearchRequest:
    return AnnSearchRequest(
        data=[embedder.embed_query(QUERY)],
        anns_field="dense",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=5,
    )


def _request_sparse(collection) -> AnnSearchRequest:
    data, metric = fulltext.sparse_request_data(collection, QUERY)
    return AnnSearchRequest(
        data=data,
        anns_field="sparse",
        param={"metric_type": metric, "params": {}},
        limit=5,
    )


def test_dense_and_sparse_retrieval(client, hybrid_setup) -> None:
    _, dense, sparse, _ = hybrid_setup
    assert dense
    assert sparse
    assert dense[0]["entity"]["category"] == "向量数据库"


def test_rrf_hybrid(client, hybrid_setup) -> None:
    collection, _, _, embedder = hybrid_setup
    hits = client.hybrid_search(
        collection_name=collection.name,
        reqs=[_request_dense(embedder), _request_sparse(collection)],
        ranker=RRFRanker(60),
        limit=5,
        output_fields=["title", "category"],
    )[0]
    assert 0 < len(hits) <= 5
    assert len({hit["id"] for hit in hits}) == len(hits)


def test_weighted_hybrid(client, hybrid_setup) -> None:
    collection, _, _, embedder = hybrid_setup
    hits = client.hybrid_search(
        collection_name=collection.name,
        reqs=[_request_dense(embedder), _request_sparse(collection)],
        ranker=WeightedRanker(0.7, 0.3),
        limit=5,
        output_fields=["title", "category"],
    )[0]
    assert 0 < len(hits) <= 5


def test_manual_rrf_on_real_hits(client, hybrid_setup) -> None:
    _, dense, sparse, _ = hybrid_setup
    fused = manual_rrf(dense, sparse, limit=5)
    assert 0 < len(fused) <= 5
    scores = [hit["distance"] for hit in fused]
    assert scores == sorted(scores, reverse=True)
    assert fused[0]["id"] in {hit["id"] for hit in dense} | {hit["id"] for hit in sparse}


def test_offset_pagination(client, docs_collection, embedder) -> None:
    params = {"metric_type": "COSINE", "params": {"ef": 64}}
    data = [embedder.embed_query("向量检索")]
    page1 = client.search(
        collection_name=docs_collection,
        data=data,
        limit=5,
        offset=0,
        search_params=params,
        output_fields=["title"],
    )[0]
    page2 = client.search(
        collection_name=docs_collection,
        data=data,
        limit=5,
        offset=5,
        search_params=params,
        output_fields=["title"],
    )[0]
    assert len(page1) == len(page2) == 5
    assert {hit["id"] for hit in page1}.isdisjoint({hit["id"] for hit in page2})


def test_range_search(client, docs_collection, embedder) -> None:
    hits = client.search(
        collection_name=docs_collection,
        data=[embedder.embed_query("向量检索")],
        limit=20,
        search_params={"metric_type": "COSINE", "params": {"radius": 0.15, "range_filter": 1.0}},
        output_fields=["title"],
    )[0]
    assert hits
    assert all(0.15 < hit["distance"] <= 1.0 for hit in hits)


def test_group_by(client, docs_collection, embedder) -> None:
    hits = client.search(
        collection_name=docs_collection,
        data=[embedder.embed_query("向量检索")],
        limit=10,
        group_by_field="category",
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        output_fields=["category"],
    )[0]
    categories = [hit["entity"]["category"] for hit in hits]
    assert categories
    assert len(categories) == len(set(categories))


def test_query_iterator(client, docs_collection) -> None:
    iterator = client.query_iterator(
        collection_name=docs_collection,
        batch_size=10,
        filter="",
        output_fields=["id"],
    )
    batches = 0
    total = 0
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            batches += 1
            total += len(batch)
    finally:
        iterator.close()
    assert batches == 3
    assert total == 25


def test_demo_data_search_wrapper(client, docs_collection, embedder) -> None:
    hits = demo_data.search(client, docs_collection, embedder.embed_query("美食"), limit=2)
    assert len(hits[0]) == 2
