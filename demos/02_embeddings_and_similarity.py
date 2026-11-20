"""Demo 02：不理解向量，就无法理解向量数据库。

本脚本完全离线，不连接 Milvus：
1. 观察伪向量的维度、范数与确定性；
2. 对比同主题/跨主题句子的余弦相似度；
3. 在 25 条语料上做暴力 Top-K，作为后续 ANN 检索的“标准答案”；
4. 比较不同维度下的 Top-1 稳定性。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlcore import corpus, pretty
from mlcore.embedding import PseudoEmbedder, cosine


def top_k(query: str, docs: list[dict], embedder, texts: list[str], k: int = 5):
    query_vector = embedder.embed_query(query)
    document_vectors = embedder.embed_documents(texts)
    scored = [(cosine(query_vector, vector), doc) for vector, doc in zip(document_vectors, docs)]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]


def main() -> None:
    pretty.banner("Demo 02：Embedding 与相似度（离线）")

    docs = corpus.load_docs()
    texts = corpus.documents_for_embedding(docs)
    embedder = PseudoEmbedder()

    pretty.section("1. 伪向量的基本性质")
    sample = "Milvus 是一款开源的向量数据库，支持语义检索。"
    vector_a = embedder.embed_query(sample)
    vector_b = embedder.embed_query(sample)
    vector_c = embedder.embed_query("Milvus 是一款开源的向量数据库，支持语义检索。")
    norm = math.sqrt(sum(value * value for value in vector_a))
    pretty.kv("维度", embedder.dim)
    pretty.kv("范数", f"{norm:.6f}")
    pretty.kv("两次结果一致", vector_a == vector_b)
    pretty.kv("等价文本一致", vector_a == vector_c)
    pretty.kv("前 6 个分量", [round(value, 5) for value in vector_a[:6]])

    pretty.section("2. 同主题 vs 跨主题的余弦相似度")
    pairs = [
        ("向量数据库怎么选型", "Milvus 的部署形态与适用场景"),
        ("向量数据库怎么选型", "红烧肉的家常做法"),
        ("周末去哪里旅行", "杭州西湖一日游攻略"),
        ("周末去哪里旅行", "Python 虚拟环境怎么创建"),
        ("火锅和早茶", "四川火锅与广式早茶的区别"),
        ("火锅和早茶", "HNSW 索引参数怎么调"),
    ]
    for left, right in pairs:
        score = cosine(embedder.embed_query(left), embedder.embed_query(right))
        pretty.kv(f"cos({left[:10]}…, {right[:10]}…)", f"{score:.4f}")

    pretty.section("3. 暴力 Top-5（后续与 Milvus ANN 对照）")
    query = "向量数据库怎么选型"
    for rank, (score, doc) in enumerate(top_k(query, docs, embedder, texts), start=1):
        print(
            f"  [{rank}] {score:.4f}  《{doc['title']}》  [{doc['category']}] {doc['text'][:40]}…"
        )

    pretty.section("4. 维度对 Top-1 的影响")
    for dim in (64, 128, 256):
        small = PseudoEmbedder(dim=dim)
        best_score, best_doc = top_k(query, docs, small, texts, k=1)[0]
        pretty.kv(
            f"{dim} 维 Top-1",
            f"{best_doc['title']}（{best_score:.4f}）",
        )

    pretty.section("小结")
    print("  - 向量是文本的数值表示，相似度用余弦/内积/欧氏距离衡量。")
    print("  - 伪向量共享字词与概念特征，因此同主题文本更接近。")
    print("  - 维度越低，哈希冲突越多，排序越容易抖动。")
    print("  - 下一步：把这些向量写入 Milvus，让索引替你完成 Top-K。")


if __name__ == "__main__":
    main()
