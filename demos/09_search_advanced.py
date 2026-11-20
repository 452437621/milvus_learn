"""Demo 09：分页、范围搜索、分组搜索与迭代器。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlcore import corpus, demo_data, pretty
from mlcore.embedding import get_embedder
from mlcore.milvus_utils import DemoSession


def ids_of(hits: list[dict]) -> list[int]:
    return [hit.get("id") for hit in hits]


def main() -> None:
    pretty.banner("Demo 09：高级搜索")
    docs = corpus.load_docs()
    embedder = get_embedder()
    query = "向量检索"
    query_vector = embedder.embed_query(query)

    with DemoSession("09_advanced") as session:
        client = session.client
        name = session.name("docs")
        demo_data.create_docs_collection(
            client,
            name,
            docs,
            embedder,
            index_type="HNSW",
            metric_type="COSINE",
            scalar_indexes=("category",),
        )

        pretty.section("1. 分页：offset + limit")
        page1 = client.search(
            collection_name=name,
            data=[query_vector],
            limit=5,
            offset=0,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["title", "category"],
        )[0]
        page2 = client.search(
            collection_name=name,
            data=[query_vector],
            limit=5,
            offset=5,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["title", "category"],
        )[0]
        pretty.kv("第 1 页", [hit["entity"]["title"] for hit in page1])
        pretty.kv("第 2 页", [hit["entity"]["title"] for hit in page2])
        pretty.kv("两页 id 是否重叠", bool(set(ids_of(page1)) & set(ids_of(page2))))
        print("  注意：offset+limit 有上限（Milvus 常见为 16384），深分页应使用迭代器。")

        pretty.section("2. 范围搜索：只保留相似度 > 0.15 的结果")
        range_hits = client.search(
            collection_name=name,
            data=[query_vector],
            limit=20,
            search_params={
                "metric_type": "COSINE",
                "params": {"radius": 0.15, "range_filter": 1.0},
            },
            output_fields=["title", "category"],
        )[0]
        for hit in range_hits:
            print(
                f"  {hit['distance']:.4f}  《{hit['entity']['title']}》 {hit['entity']['category']}"
            )
        if range_hits:
            print(
                f"  共 {len(range_hits)} 条，分数均 > 0.15（COSINE 区间：radius < score <= range_filter）。"
            )
        else:
            print("  没有结果；可调低 radius 再试。")

        pretty.section("3. 分组搜索：每个分类只取最相关的一条")
        group_hits = client.search(
            collection_name=name,
            data=[query_vector],
            limit=10,
            group_by_field="category",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["title", "category"],
        )[0]
        for hit in group_hits:
            print(f"  《{hit['entity']['title']}》  [{hit['entity']['category']}]")
        categories = [hit["entity"]["category"] for hit in group_hits]
        pretty.kv("类目数", len(categories))
        pretty.kv("类目是否唯一", len(categories) == len(set(categories)))

        pretty.section("4. 查询迭代器：分批遍历全部 25 条")
        iterator = client.query_iterator(
            collection_name=name,
            batch_size=10,
            filter="",
            output_fields=["id", "title"],
        )
        total = 0
        batches = 0
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                batches += 1
                total += len(batch)
                print(f"  batch {batches}: {len(batch)} 条")
        finally:
            iterator.close()
        pretty.kv("迭代总数", total)

        pretty.section("小结")
        print("  - offset 分页简单但有上限，深分页用 query_iterator。")
        print("  - range search 的边界方向取决于度量：L2 与 COSINE/IP 相反。")
        print("  - group_by 适合“每个类目一条”的展示与多样性控制。")
        print("  - 迭代器适合导出/批量处理，不占用一次大查询的内存。")


if __name__ == "__main__":
    main()
