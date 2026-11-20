"""Demo 05：标量过滤表达式。

覆盖比较、集合、前缀匹配、JSON 路径、数组函数、布尔组合，
并演示 search(filter=...) 的“向量 + 过滤”组合。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlcore import corpus, demo_data, pretty
from mlcore.embedding import get_embedder
from mlcore.milvus_utils import DemoSession

FILTERS = [
    ("分类等值", 'category == "美食"'),
    ("年份范围", "year >= 2022"),
    ("评分阈值", "rating > 4.5"),
    ("集合 in", 'category in ["旅行", "电影"]'),
    ("前缀 like", 'author like "美食%"'),
    ("JSON 路径", 'meta["level"] == "进阶"'),
    ("数组包含", 'array_contains(tags, "Milvus")'),
    ("布尔组合", "is_free == true and rating >= 4.5"),
]


def main() -> None:
    pretty.banner("Demo 05：标量过滤")
    docs = corpus.load_docs()
    embedder = get_embedder()

    with DemoSession("05_filter") as session:
        client = session.client
        name = session.name("docs")
        demo_data.create_docs_collection(
            client,
            name,
            docs,
            embedder,
            index_type="HNSW",
            metric_type="COSINE",
            scalar_indexes=("category", "year", "is_free"),
        )

        for label, expression in FILTERS:
            pretty.section(f"{label}: {expression}")
            try:
                rows = client.query(
                    collection_name=name,
                    filter=expression,
                    output_fields=["id", "title", "category", "year", "rating"],
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  执行失败：{exc}")
                continue
            print(f"  命中 {len(rows)} 条")
            for row in rows[:3]:
                print(
                    f"    - {row['title']} [{row['category']}] "
                    f"{row['year']} 评分={row['rating']:.2f}"
                )
            if len(rows) > 3:
                print(f"    … 其余 {len(rows) - 3} 条")

        pretty.section("向量检索 + 过滤：只在“美食”里搜“面条”")
        hits = demo_data.search(
            client,
            name,
            embedder.embed_query("面条"),
            limit=3,
            filter='category == "美食"',
            metric_type="COSINE",
            params={"ef": 64},
        )[0]
        pretty.print_hits(hits)
        categories = {hit["entity"].get("category") for hit in hits}
        pretty.kv("命中分类集合", categories)

        pretty.section("小结")
        print("  - 字符串用双引号；布尔用 true/false；中文值可直接写。")
        print("  - query 只做标量条件；search 才计算向量相似度。")
        print("  - JSON 路径和数组函数很方便，但高选择性过滤要关注召回与性能。")
        print("  - 高频过滤字段建议建 INVERTED 标量索引。")


if __name__ == "__main__":
    main()
