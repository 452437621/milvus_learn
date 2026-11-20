"""集成：显式分区与别名切换。"""

from __future__ import annotations

import pytest
from pymilvus.exceptions import MilvusException

from mlcore import demo_data
from mlcore.milvus_utils import count_entities, drop_collection_quietly

pytestmark = pytest.mark.integration

PARTITIONS = {
    "cat_ai": ("向量数据库", "编程开发"),
    "cat_food": ("美食",),
    "cat_travel": ("旅行",),
}


def _partition_of(category: str) -> str:
    for name, categories in PARTITIONS.items():
        if category in categories:
            return name
    return "_default"


def _ids_in_partition(client, name: str, partition: str) -> list[int]:
    # Milvus Lite 的 count(*) 会忽略 partition_names，这里取回 id 再计数。
    rows = client.query(
        collection_name=name,
        filter="",
        output_fields=["id"],
        partition_names=[partition],
    )
    return [row["id"] for row in rows]


def test_partition_counts(client, docs, embedder) -> None:
    name = "part_docs"
    rows = demo_data.create_docs_collection(client, name, docs, embedder, insert=False, load=False)
    for partition in PARTITIONS:
        client.create_partition(collection_name=name, partition_name=partition)
    assert {"cat_ai", "cat_food", "cat_travel"} <= set(client.list_partitions(collection_name=name))

    grouped: dict[str, list[dict]] = {partition: [] for partition in PARTITIONS}
    grouped["_default"] = []
    for row in rows:
        grouped[_partition_of(row["category"])].append(row)
    for partition, partition_rows in grouped.items():
        if not partition_rows:
            continue
        if partition == "_default":
            client.insert(collection_name=name, data=partition_rows)
        else:
            client.insert(collection_name=name, data=partition_rows, partition_name=partition)
    client.load_collection(collection_name=name)

    counts = {
        partition: len(_ids_in_partition(client, name, partition))
        for partition in client.list_partitions(collection_name=name)
    }
    assert counts == {"_default": 7, "cat_ai": 10, "cat_food": 4, "cat_travel": 4}
    assert sum(counts.values()) == 25
    assert count_entities(client, name) == 25


def test_search_inside_partition(client, docs, embedder) -> None:
    name = "part_search"
    rows = demo_data.create_docs_collection(client, name, docs, embedder, insert=False, load=False)
    client.create_partition(collection_name=name, partition_name="cat_food")
    food = [row for row in rows if row["category"] == "美食"]
    client.insert(collection_name=name, data=food, partition_name="cat_food")
    client.load_collection(collection_name=name)

    hits = demo_data.search(
        client,
        name,
        embedder.embed_query("面条"),
        limit=4,
        partition_names=["cat_food"],
    )[0]
    assert hits
    assert {hit["entity"]["category"] for hit in hits} == {"美食"}


def test_alias_switch(client, docs, embedder) -> None:
    v1 = "alias_v1"
    v2 = "alias_v2"
    alias = "alias_current"
    demo_data.create_docs_collection(client, v1, docs, embedder)

    rows = demo_data.create_docs_collection(client, v2, docs, embedder, insert=False, load=False)
    for row in rows:
        row["title"] = "V2·" + row["title"]
    client.insert(collection_name=v2, data=rows)
    client.load_collection(collection_name=v2)

    client.create_alias(collection_name=v1, alias=alias)
    assert alias in client.list_aliases(collection_name=v1)["aliases"]

    query = embedder.embed_query("Milvus 向量检索")
    before = demo_data.search(client, alias, query, limit=3)[0]
    assert all(not hit["entity"]["title"].startswith("V2·") for hit in before)

    client.alter_alias(collection_name=v2, alias=alias)
    after = demo_data.search(client, alias, query, limit=3)[0]
    assert all(hit["entity"]["title"].startswith("V2·") for hit in after)

    client.drop_alias(alias=alias)
    assert client.list_aliases(collection_name=v1)["aliases"] == []
    with pytest.raises(MilvusException):
        demo_data.search(client, alias, query, limit=1)

    drop_collection_quietly(client, v2)
    drop_collection_quietly(client, v1)
