"""Demo 04：FLAT / HNSW / IVF_FLAT 三种索引对比。

同一份 25 条语料，分别用：
- FLAT + COSINE：精确基线；
- HNSW + COSINE：图索引，低延迟；
- IVF_FLAT + L2：聚类分桶，度量不同，分数不可直接比较。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlcore import corpus, demo_data, pretty
from mlcore.embedding import get_embedder
from mlcore.milvus_utils import DemoSession

QUERY = "向量数据库怎么选型"


def titles(hits: list[dict]) -> list[str]:
    return [hit["entity"]["title"] for hit in hits]


def main() -> None:
    pretty.banner("Demo 04：索引类型对比")
    docs = corpus.load_docs()
    embedder = get_embedder()
    query_vector = embedder.embed_query(QUERY)
    pretty.kv("查询", QUERY)

    with DemoSession("04_index") as session:
        client = session.client
        collections = {
            "FLAT + COSINE": ("docs_flat", "FLAT", "COSINE", {}, {}),
            "HNSW + COSINE": (
                "docs_hnsw",
                "HNSW",
                "COSINE",
                {"M": 16, "efConstruction": 200},
                {"ef": 64},
            ),
            "IVF_FLAT + L2": ("docs_ivf", "IVF_FLAT", "L2", {"nlist": 8}, {"nprobe": 4}),
        }

        results = {}
        for label, (suffix, index_type, metric, index_params, search_params) in collections.items():
            name = session.name(suffix)
            pretty.section(f"构建 {label}")
            demo_data.create_docs_collection(
                client,
                name,
                docs,
                embedder,
                index_type=index_type,
                metric_type=metric,
                index_params=index_params,
            )
            info = client.describe_index(collection_name=name, index_name="embedding")
            pretty.kv("索引描述", info)

            hits = demo_data.search(
                client,
                name,
                query_vector,
                limit=5,
                metric_type=metric,
                params=search_params,
            )[0]
            results[label] = hits
            pretty.kv(
                "Top-5",
                titles(hits),
            )
            pretty.print_hits(hits)

        pretty.section("排名重合度：FLAT 是精确基线")
        flat_titles = titles(results["FLAT + COSINE"])
        for label in ("HNSW + COSINE", "IVF_FLAT + L2"):
            other = titles(results[label])
            overlap = len(set(flat_titles) & set(other))
            pretty.kv(label, f"与 FLAT 的 Top-5 重合 {overlap}/5")

        pretty.section("如何读分数")
        print("  COSINE/内积：分数越大越相似；L2：距离越小越相似。")
        print("  因此 IVF_FLAT + L2 的分数不能直接和 COSINE 比较。")
        print("  25 条数据太小，IVF 的 nprobe 调参意义有限；大样本才会体现差异。")

        pretty.section("小结")
        print("  - FLAT 精确但慢，适合小数据与效果基线。")
        print("  - HNSW 低延迟高召回，内存与构建成本更高。")
        print("  - IVF 用 nlist/nprobe 在召回与延迟之间折中。")
        print("  - 同一套 Schema 可以换索引，但 metric 必须与搜索参数一致。")


if __name__ == "__main__":
    main()
