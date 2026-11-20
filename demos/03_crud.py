"""Demo 03：CRUD 与 Schema 细节。

覆盖 insert / upsert / get / query / delete / count，
以及 nullable、default_value、ARRAY、JSON 的读写。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlcore import corpus, demo_data, pretty
from mlcore.embedding import get_embedder
from mlcore.milvus_utils import DemoSession, count_entities

NAME = "docs"


def show_count(client, name: str, label: str) -> None:
    pretty.kv(label, count_entities(client, name))


def main() -> None:
    pretty.banner("Demo 03：CRUD 与 Schema")
    docs = corpus.load_docs()
    embedder = get_embedder()

    with DemoSession("03_crud") as session:
        client = session.client
        NAME = session.name("docs")

        pretty.section("1. 建表并批量写入 25 条语料")
        rows = demo_data.create_docs_collection(
            client,
            NAME,
            docs,
            embedder,
            index_type="HNSW",
            metric_type="COSINE",
            scalar_indexes=("category", "year"),
        )
        show_count(client, NAME, "初始数量")
        pretty.kv(
            "第一条",
            pretty.plain_rows(
                client.get(collection_name=NAME, ids=[1], output_fields=["title", "rating"])
            ),
        )

        pretty.section("2. upsert：按主键更新（数量不变）")
        updated = dict(rows[0])
        updated["title"] = "Milvus 是什么（已更新）"
        updated["rating"] = 5.0
        client.upsert(collection_name=NAME, data=[updated])
        show_count(client, NAME, "upsert 后数量")
        pretty.kv(
            "id=1 当前值",
            pretty.plain_rows(
                client.get(collection_name=NAME, ids=[1], output_fields=["title", "rating"])
            ),
        )

        pretty.section("3. 默认值：插入一条缺少 author 的记录")
        extra = dict(rows[1])
        extra["id"] = 999
        extra.pop("author")
        client.insert(collection_name=NAME, data=[extra])
        pretty.kv(
            "id=999 的 author",
            pretty.plain_rows(
                client.get(collection_name=NAME, ids=[999], output_fields=["title", "author"])
            ),
        )
        client.delete(collection_name=NAME, ids=[999])
        show_count(client, NAME, "清理临时记录后")

        pretty.section("4. delete：删除 id=24")
        client.delete(collection_name=NAME, ids=[24])
        show_count(client, NAME, "删除后数量")
        pretty.kv(
            "再查 id=24",
            pretty.plain_rows(client.get(collection_name=NAME, ids=[24], output_fields=["title"])),
        )

        pretty.section("5. query：标量条件与 JSON / ARRAY")
        food = client.query(
            collection_name=NAME,
            filter='category == "美食"',
            output_fields=["id", "title", "year", "rating", "tags"],
        )
        print(f"  美食类文档（{len(food)} 条）：")
        for row in food:
            pretty.kv(row["title"], f"{row['year']} / {row['rating']:.2f} / {row['tags']}")

        advanced = client.query(
            collection_name=NAME,
            filter='meta["level"] == "进阶"',
            output_fields=["title", "meta"],
        )
        print(f"\n  meta.level == 进阶（{len(advanced)} 条）：")
        for row in advanced:
            pretty.kv(row["title"], row["meta"])

        milvus_tags = client.query(
            collection_name=NAME,
            filter='array_contains(tags, "Milvus")',
            output_fields=["title", "tags"],
        )
        print(f"\n  tags 包含 Milvus（{len(milvus_tags)} 条）：")
        for row in milvus_tags:
            pretty.kv(row["title"], row["tags"])

        pretty.section("小结")
        print("  - insert 追加；upsert 按主键覆盖；delete 后要等 compaction 才真正释放空间。")
        print("  - nullable/default_value 影响“缺字段”时的写入行为。")
        print('  - ARRAY 用 array_contains，JSON 用 meta["key"] 过滤。')
        print("  - 删除后立即 get 不到，但磁盘空间不是立刻回收。")


if __name__ == "__main__":
    main()
