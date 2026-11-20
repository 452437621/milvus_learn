"""集成：CRUD 与默认值语义。"""

from __future__ import annotations

import pytest

from mlcore import demo_data
from mlcore.milvus_utils import count_entities

pytestmark = pytest.mark.integration


def test_upsert_keeps_count(client, docs, embedder) -> None:
    name = "crud_upsert"
    rows = demo_data.create_docs_collection(client, name, docs, embedder)
    assert count_entities(client, name) == 25

    updated = dict(rows[0])
    updated["title"] = "Milvus 是什么（已更新）"
    updated["rating"] = 5.0
    client.upsert(collection_name=name, data=[updated])
    client.flush(collection_name=name)

    assert count_entities(client, name) == 25
    row = client.get(collection_name=name, ids=[1], output_fields=["title", "rating"])[0]
    assert row["title"] == "Milvus 是什么（已更新）"
    assert row["rating"] == pytest.approx(5.0)


def test_default_value_and_delete(client, docs, embedder) -> None:
    name = "crud_default"
    rows = demo_data.create_docs_collection(client, name, docs, embedder)

    extra = dict(rows[1])
    extra["id"] = 999
    extra.pop("author")
    client.insert(collection_name=name, data=[extra])
    client.flush(collection_name=name)

    assert count_entities(client, name) == 26
    row = client.get(collection_name=name, ids=[999], output_fields=["author"])[0]
    assert row["author"] == "佚名"

    client.delete(collection_name=name, ids=[999])
    client.flush(collection_name=name)
    assert count_entities(client, name) == 25
    assert client.get(collection_name=name, ids=[999], output_fields=["title"]) == []


def test_delete_by_filter(client, docs, embedder) -> None:
    name = "crud_filter"
    demo_data.create_docs_collection(client, name, docs, embedder)
    client.delete(collection_name=name, filter='category == "美食"')
    client.flush(collection_name=name)

    assert count_entities(client, name) == 21
    remaining = client.query(
        collection_name=name,
        filter='category == "美食"',
        output_fields=["id"],
    )
    assert remaining == []


def test_query_projection(client, docs, embedder) -> None:
    name = "crud_query"
    demo_data.create_docs_collection(client, name, docs, embedder)
    rows = client.query(
        collection_name=name,
        filter="year >= 2024",
        output_fields=["title", "year"],
    )
    assert rows
    # pymilvus 总会返回主键；除此之外只应返回请求的字段
    assert all(set(row) == {"id", "title", "year"} for row in rows)
