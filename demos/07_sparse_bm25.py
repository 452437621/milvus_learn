"""Demo 07：稀疏向量与 BM25 全文检索。

优先使用 Milvus 原生 BM25（VARCHAR 分析器 + Function 自动生成稀疏向量）；
如果环境不支持，则自动回退到“手动 TF 稀疏 + IP”，并在输出中明确标注。
同时用稠密向量做对比，观察关键词召回与语义召回的区别。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlcore import config, corpus, fulltext, pretty
from mlcore.embedding import get_embedder
from mlcore.milvus_utils import DemoSession

QUERIES = ["向量 检索", "火锅 麻辣", "睡眠 作息"]


def main() -> None:
    pretty.banner("Demo 07：稀疏向量与 BM25")
    docs = corpus.load_docs()
    embedder = get_embedder()

    with DemoSession("07_bm25") as session:
        client = session.client
        name = session.name("docs_bm25")

        pretty.section("1. 构建全文检索集合")
        collection = fulltext.create_fulltext_collection(
            client,
            name,
            docs,
            embedder,
            mode=config.BM25_MODE,
        )
        pretty.kv("模式", collection.mode)
        pretty.kv("分析器", collection.analyzer)
        pretty.kv("文档数", collection.count)
        if collection.mode == "manual":
            print("  注意：当前是手动 TF 稀疏 + IP，不是真正的 BM25 统计。")
            print("        安装 jieba 并确认 BM25 可用后，可得到原生 BM25。")

        for query in QUERIES:
            pretty.section(f"2. BM25 检索：{query}")
            hits = fulltext.bm25_search(client, collection, query, limit=3)[0]
            pretty.print_hits(hits, max_text=55)
            print("  提示：Milvus 的 BM25 分数通常为负，越小（越负）越相关。")

            pretty.section(f"3. 稠密向量检索（同一查询）：{query}")
            dense_hits = client.search(
                collection_name=name,
                data=[embedder.embed_query(query)],
                anns_field="dense",
                limit=3,
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                output_fields=["title", "category", "text"],
            )[0]
            pretty.print_hits(dense_hits, max_text=55)
            print("\n  对比：BM25 更看关键词是否出现；稠密向量更看整体语义。")

        pretty.section("小结")
        print("  - 稀疏向量只存非零词项，天然适合倒排索引。")
        print("  - BM25 = 词频饱和 + IDF + 文档长度归一化。")
        print("  - 中文必须使用 jieba/中文分析器，否则分词效果很差。")
        print("  - BM25 分数与余弦分数不可直接比较。")


if __name__ == "__main__":
    main()
