"""Demo 08：混合检索（dense + BM25）与 RRF / 加权融合。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlcore import config, corpus, fulltext, pretty
from mlcore.embedding import get_embedder
from mlcore.hybrid import manual_rrf
from mlcore.milvus_utils import DemoSession

QUERY = "向量 检索 选型"


def show(label: str, hits: list[dict]) -> None:
    pretty.section(label)
    if not hits:
        print("  （无结果）")
        return
    for hit in hits:
        entity = hit.get("entity", {})
        score = hit.get("distance", hit.get("score"))
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        print(f"  {score_text:>9}  《{entity.get('title', '')}》  {entity.get('category', '')}")


def main() -> None:
    pretty.banner("Demo 08：混合检索与融合排序")
    docs = corpus.load_docs()
    embedder = get_embedder()

    with DemoSession("08_hybrid") as session:
        client = session.client
        name = session.name("docs_hybrid")
        collection = fulltext.create_fulltext_collection(
            client, name, docs, embedder, mode=config.BM25_MODE
        )
        pretty.kv("全文检索模式", f"{collection.mode}（analyzer={collection.analyzer}）")
        pretty.kv("查询", QUERY)

        dense_hits = client.search(
            collection_name=name,
            data=[embedder.embed_query(QUERY)],
            anns_field="dense",
            limit=5,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["title", "category"],
        )[0]
        show("1. 稠密向量 Top-5（COSINE，越大越好）", dense_hits)

        sparse_hits = fulltext.bm25_search(client, collection, QUERY, limit=5)[0]
        show("2. BM25 Top-5（分数通常为负，越负越好）", sparse_hits)

        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        dense_data = [embedder.embed_query(QUERY)]
        sparse_data, sparse_metric = fulltext.sparse_request_data(collection, QUERY)
        dense_req = AnnSearchRequest(
            data=dense_data,
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=5,
        )
        sparse_req = AnnSearchRequest(
            data=sparse_data,
            anns_field="sparse",
            param={"metric_type": sparse_metric, "params": {}},
            limit=5,
        )

        try:
            rrf_hits = client.hybrid_search(
                collection_name=name,
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(60),
                limit=5,
                output_fields=["title", "category"],
            )[0]
            show("3. RRF 融合（只看排名，不看原始分数）", rrf_hits)
        except Exception as exc:  # noqa: BLE001
            print(f"  原生 hybrid_search 失败，改用手动 RRF：{exc}")
            rrf_hits = manual_rrf(dense_hits, sparse_hits)
            show("3. 手动 RRF 融合", rrf_hits)

        try:
            weighted_hits = client.hybrid_search(
                collection_name=name,
                reqs=[dense_req, sparse_req],
                ranker=WeightedRanker(0.7, 0.3),
                limit=5,
                output_fields=["title", "category"],
            )[0]
            show("4. 加权融合 WeightedRanker(0.7, 0.3)", weighted_hits)
        except Exception as exc:  # noqa: BLE001
            print(f"  加权融合不可用：{exc}")

        pretty.section("观察点")
        print("  - RRF 只依赖名次，天然规避“不同分数尺度不可比”的问题。")
        print("  - 加权融合需要先归一化分数；权重反映你对各路召回的信任程度。")
        print("  - 混合检索能同时接住“关键词精确匹配”和“语义相近”的查询。")
        print("  - 融合后再按文档去重，避免同一文档的多个 chunk 占满结果。")


if __name__ == "__main__":
    main()
