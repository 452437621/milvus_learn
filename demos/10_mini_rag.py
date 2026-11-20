"""Demo 10：迷你 RAG。

流程：切块 -> 向量化 -> 写入 Milvus -> 检索 -> 组装 Prompt -> 回答。
默认离线抽取式回答；配置 RAG_LLM=openai + API Key 后可调用大模型。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlcore import corpus, pretty, rag
from mlcore.embedding import get_embedder
from mlcore.milvus_utils import DemoSession
from mlcore.providers import build_chat


def main() -> None:
    parser = argparse.ArgumentParser(description="迷你 RAG 示例")
    parser.add_argument(
        "--question",
        default="Milvus Lite 有哪些限制？",
        help="要提问的问题",
    )
    parser.add_argument("--top-k", type=int, default=4, help="检索片段数")
    parser.add_argument("--category", default="", help="可选的分类过滤，如 向量数据库")
    args = parser.parse_args()

    pretty.banner("Demo 10：Mini RAG")
    docs = corpus.load_docs()
    embedder = get_embedder()

    with DemoSession("10_rag") as session:
        client = session.client
        name = session.name("chunks")

        pretty.section("1. 切块并建立索引")
        start = time.perf_counter()
        chunks = rag.build_rag_collection(client, name, docs, embedder)
        build_seconds = time.perf_counter() - start
        pretty.kv("文档数", len(docs))
        pretty.kv("chunk 数", len(chunks))
        pretty.kv("索引耗时", f"{build_seconds:.3f}s")
        for chunk in chunks[:3]:
            pretty.kv(f"示例 chunk {chunk.chunk_id}", chunk.text[:70] + "…")

        pretty.section(f"2. 检索：{args.question}")
        start = time.perf_counter()
        hits = rag.retrieve(
            client,
            name,
            args.question,
            embedder,
            top_k=args.top_k,
            category=args.category or None,
        )
        retrieve_seconds = time.perf_counter() - start
        pretty.kv("检索耗时", f"{retrieve_seconds * 1000:.1f} ms")
        pretty.print_hits(hits, fields=("chunk_index",), max_text=70)

        pretty.section("3. 组装 Prompt")
        prompt = rag.build_prompt(args.question, hits)
        print(prompt)

        pretty.section("4. 生成回答")
        chat = build_chat()
        start = time.perf_counter()
        answer, mode, _ = rag.answer_question(args.question, hits, chat)
        answer_seconds = time.perf_counter() - start
        pretty.kv("模式", mode)
        print("\n" + answer)

        pretty.section("5. 来源与耗时")
        for row in rag.source_rows(hits):
            print(f"  [{row[0]}] {row[2]} / chunk {row[4]} / score={row[1]}")
        pretty.kv("检索耗时", f"{retrieve_seconds * 1000:.1f} ms")
        pretty.kv("回答耗时", f"{answer_seconds * 1000:.1f} ms")
        pretty.kv("总耗时", f"{(build_seconds + retrieve_seconds + answer_seconds) * 1000:.1f} ms")

        pretty.section("小结")
        print("  - RAG 的质量取决于切块、embedding、检索与 Prompt 四段。")
        print("  - 离线抽取式回答只复述片段，不会生成新内容。")
        print("  - 配置 RAG_LLM=openai 后，问题与检索片段会发送到外部服务。")
        print("  - 伪向量语义能力有限，换成真实 embedding 往往提升明显。")


if __name__ == "__main__":
    main()
