"""Demo 01：Milvus 最小闭环。

创建 collection -> 建索引 -> 写入 6 条语料 -> load -> 向量检索 -> 标量查询。
重点是理解“Schema / 索引 / 写入 / 加载 / 搜索”这条链路。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlcore import corpus, demo_data, pretty
from mlcore.embedding import get_embedder
from mlcore.milvus_utils import DemoSession, count_entities


def print_schema(client, name: str) -> None:
    from pymilvus import DataType

    description = client.describe_collection(collection_name=name)
    pretty.kv("collection", description.get("collection_name", name))
    pretty.kv("auto_id", description.get("auto_id"))
    pretty.kv("description", description.get("description", ""))
    for field in description.get("fields", []):
        extra = {}
        for key in ("max_length", "element_type", "max_capacity", "is_primary"):
            if key in field:
                extra[key] = field[key]
        params = field.get("params")
        if params:
            extra.update(params)
        raw_type = field.get("type")
        try:
            type_name = DataType(raw_type).name
        except (ValueError, TypeError):
            type_name = str(raw_type)
        pretty.kv(f"field {field.get('name')}", f"{type_name} {extra}")


def main() -> None:
    pretty.banner("Demo 01：Hello Milvus")
    docs = corpus.load_docs()[:6]
    embedder = get_embedder()

    with DemoSession("01_hello") as session:
        client = session.client
        name = session.name("docs")
        pretty.kv("连接", session.uri)
        pretty.kv("语料条数", len(docs))
        pretty.kv("向量维度", embedder.dim)

        pretty.section("1. 创建 collection（显式 Schema + HNSW/COSINE 索引）")
        demo_data.create_docs_collection(
            client,
            name,
            docs,
            embedder,
            index_type="HNSW",
            metric_type="COSINE",
            scalar_indexes=("category",),
        )
        print_schema(client, name)

        pretty.section("2. 写入与加载")
        pretty.kv("实体数量", count_entities(client, name))
        pretty.kv("索引", client.describe_index(collection_name=name, index_name="embedding"))

        pretty.section("3. 向量检索：query = 向量数据库怎么选型")
        query = "向量数据库怎么选型"
        hits = demo_data.search(
            client,
            name,
            embedder.embed_query(query),
            limit=3,
            metric_type="COSINE",
            params={"ef": 64},
        )[0]
        pretty.print_hits(hits)
        print("\n  分数越大越相似（COSINE）。Top-1 通常应是 Milvus/向量数据库相关文档。")

        pretty.section("4. 标量查询：只看“向量数据库”分类")
        rows = client.query(
            collection_name=name,
            filter='category == "向量数据库"',
            output_fields=["id", "title", "year", "rating"],
        )
        for row in rows:
            pretty.kv(row["title"], f"年份={row['year']} 评分={row['rating']}")

        pretty.section("小结")
        print("  - Schema 定义了字段、主键、向量维度。")
        print("  - 索引决定查询用哪种 ANN 结构；metric 必须与搜索一致。")
        print("  - 写入后需要 load 才能被查询。")
        print("  - search 是向量相似度，query 是标量条件。")


if __name__ == "__main__":
    main()
