"""集成：建表、描述、计数与基础检索。"""

from __future__ import annotations

import pytest

from mlcore import demo_data
from mlcore.milvus_utils import count_entities, ensure_loaded

pytestmark = pytest.mark.integration

SCHEMA_FIELDS = {
    "id",
    "title",
    "category",
    "tags",
    "text",
    "year",
    "rating",
    "price",
    "is_free",
    "author",
    "meta",
    "embedding",
}


def test_collection_created(client, docs_collection) -> None:
    assert count_entities(client, docs_collection) == 25
    assert docs_collection in client.list_collections()
    description = client.describe_collection(collection_name=docs_collection)
    assert {field["name"] for field in description["fields"]} == SCHEMA_FIELDS


def test_index_description(client, docs_collection) -> None:
    info = client.describe_index(collection_name=docs_collection, index_name="embedding")
    assert info["index_type"] == "HNSW"
    assert info["metric_type"] == "COSINE"


def test_reload_is_idempotent(client, docs_collection) -> None:
    ensure_loaded(client, docs_collection)
    ensure_loaded(client, docs_collection)
    assert count_entities(client, docs_collection) == 25


def test_search_returns_expected_fields(client, docs_collection, embedder) -> None:
    hits = demo_data.search(
        client, docs_collection, embedder.embed_query("向量数据库怎么选型"), limit=5
    )[0]
    assert len(hits) == 5
    assert hits[0]["entity"]["title"] == "Milvus 是什么"
    for hit in hits:
        assert {"title", "category", "year", "rating"} <= set(hit["entity"])


def test_meta_and_array_roundtrip(client, docs_collection) -> None:
    rows = client.get(
        collection_name=docs_collection,
        ids=[3],
        output_fields=["tags", "meta", "author"],
    )
    assert rows[0]["tags"] == ["索引", "HNSW", "ANN"]
    assert rows[0]["meta"]["level"] == "进阶"
    assert rows[0]["author"] == "Malkov"
