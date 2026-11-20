"""集成：DemoSession 与计数/加载辅助函数配合真实 Milvus Lite。"""

from __future__ import annotations

import pytest

from mlcore import demo_data
from mlcore.milvus_utils import count_entities, ensure_loaded

pytestmark = pytest.mark.integration


def test_demo_session_end_to_end(demo_session, docs, embedder) -> None:
    client = demo_session.client
    name = demo_session.name("docs")
    demo_data.create_docs_collection(client, name, docs, embedder)

    assert demo_session.local is True
    assert name == "docs"
    assert name in client.list_collections()
    assert count_entities(client, name) == 25
    ensure_loaded(client, name)

    hits = demo_data.search(client, name, embedder.embed_query("火锅"), limit=2)[0]
    assert hits


def test_count_entities_empty_collection(client) -> None:
    name = "stats_only"
    from pymilvus import DataType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
    client.create_collection(collection_name=name, schema=schema)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
    client.create_index(collection_name=name, index_params=index_params)
    client.load_collection(collection_name=name)

    assert count_entities(client, name) == 0


def test_count_entities_unknown_collection(client) -> None:
    assert count_entities(client, "does_not_exist") is None


def test_ensure_loaded_ignores_already_loaded(client, docs_collection) -> None:
    ensure_loaded(client, docs_collection)
    ensure_loaded(client, docs_collection)
