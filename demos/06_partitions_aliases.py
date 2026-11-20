"""Demo 06：分区、分区路由与别名切换。

- 把 25 条语料按分类写入 3 个显式分区；
- 只在指定分区内搜索；
- 用别名在 v1 / v2 之间做“蓝绿切换”。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlcore import corpus, demo_data, pretty
from mlcore.embedding import get_embedder
from mlcore.milvus_utils import DemoSession, count_entities

PARTITION_MAP = {
    "cat_ai": ("向量数据库", "编程开发"),
    "cat_food": ("美食",),
    "cat_travel": ("旅行",),
}


def partition_of(category: str) -> str:
    for partition, categories in PARTITION_MAP.items():
        if category in categories:
            return partition
    return "_default"


def partition_count(client, name: str, partition: str) -> int:
    # 注意：Milvus Lite 3.2 的 count(*) 目前会忽略 partition_names，
    # 因此这里取回 id 再计数。
    rows = client.query(
        collection_name=name,
        filter="",
        output_fields=["id"],
        partition_names=[partition],
    )
    return len(rows)


def main() -> None:
    pretty.banner("Demo 06：分区与别名")
    docs = corpus.load_docs()
    embedder = get_embedder()

    with DemoSession("06_partition") as session:
        client = session.client
        name = session.name("docs")
        alias = session.name("docs_current")

        pretty.section("1. 建集合（先不插入），再创建显式分区")
        rows = demo_data.create_docs_collection(
            client,
            name,
            docs,
            embedder,
            index_type="HNSW",
            metric_type="COSINE",
            insert=False,
            load=False,
        )
        for partition in PARTITION_MAP:
            client.create_partition(collection_name=name, partition_name=partition)
        pretty.kv("分区列表", client.list_partitions(collection_name=name))

        pretty.section("2. 按分类把数据写入对应分区")
        grouped: dict[str, list[dict]] = {partition: [] for partition in PARTITION_MAP}
        grouped["_default"] = []
        for row in rows:
            grouped[partition_of(row["category"])].append(row)

        for partition, partition_rows in grouped.items():
            if not partition_rows:
                continue
            if partition == "_default":
                client.insert(collection_name=name, data=partition_rows)
            else:
                client.insert(
                    collection_name=name,
                    data=partition_rows,
                    partition_name=partition,
                )
            pretty.kv(partition, f"写入 {len(partition_rows)} 条")

        client.load_collection(collection_name=name)

        pretty.section("3. 分区计数")
        total = 0
        for partition in client.list_partitions(collection_name=name):
            count = partition_count(client, name, partition)
            total += count
            pretty.kv(partition, count)
        pretty.kv("合计", total)
        pretty.kv("collection 总数", count_entities(client, name))
        print("  说明：Lite 3.2 的 count(*) 会忽略 partition_names，这里用取回 id 的方式计数。")

        pretty.section("4. 只在 cat_food 分区内搜索“面条”")
        hits = demo_data.search(
            client,
            name,
            embedder.embed_query("面条"),
            limit=4,
            metric_type="COSINE",
            params={"ef": 64},
            partition_names=["cat_food"],
        )[0]
        pretty.print_hits(hits)
        pretty.kv("命中分类", {hit["entity"]["category"] for hit in hits})

        pretty.section("5. 别名：蓝绿切换")
        v2 = session.name("docs_v2")
        v2_rows = demo_data.create_docs_collection(
            client,
            v2,
            docs,
            embedder,
            index_type="HNSW",
            metric_type="COSINE",
            insert=False,
            load=False,
        )
        for row in v2_rows:
            row["title"] = "V2·" + row["title"]
        client.insert(collection_name=v2, data=v2_rows)
        client.load_collection(collection_name=v2)

        client.create_alias(collection_name=name, alias=alias)
        pretty.kv("别名", client.list_aliases(collection_name=name))

        query_vector = embedder.embed_query("Milvus 向量检索")
        before = demo_data.search(
            client, alias, query_vector, limit=3, metric_type="COSINE", params={"ef": 64}
        )[0]
        pretty.kv("切换前 (→ v1)", [hit["entity"]["title"] for hit in before])

        client.alter_alias(collection_name=v2, alias=alias)
        after = demo_data.search(
            client, alias, query_vector, limit=3, metric_type="COSINE", params={"ef": 64}
        )[0]
        pretty.kv("切换后 (→ v2)", [hit["entity"]["title"] for hit in after])

        client.drop_alias(alias=alias)
        pretty.kv("删除别名后", client.list_aliases(collection_name=name))

        pretty.section("小结")
        print("  - 显式分区适合同类数据隔离、时间分区和批量删除。")
        print("  - 分区内搜索会裁剪候选集，减少计算。")
        print("  - 别名让应用只认一个名字，底层 collection 可蓝绿切换。")
        print("  - Milvus Lite 的 gRPC 适配器不支持分区级 load/release。")


if __name__ == "__main__":
    main()
