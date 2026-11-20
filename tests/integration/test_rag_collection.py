"""集成：RAG 集合构建、检索去重与问答。"""

from __future__ import annotations

import pytest

from mlcore import rag
from mlcore.milvus_utils import count_entities

pytestmark = pytest.mark.integration


@pytest.fixture
def rag_setup(client, docs, embedder):
    chunks = rag.build_rag_collection(client, "rag_chunks", docs, embedder)
    return chunks, embedder


def test_chunk_count_matches_layout(client, rag_setup, docs) -> None:
    chunks, _ = rag_setup
    expected = sum(len(rag.chunk_text(doc["text"])) for doc in docs)
    assert len(chunks) == expected
    assert count_entities(client, "rag_chunks") == expected


def test_retrieve_respects_top_k_and_per_doc(client, rag_setup) -> None:
    _, embedder = rag_setup
    hits = rag.retrieve(client, "rag_chunks", "Milvus Lite 有哪些限制？", embedder, top_k=4)
    assert 0 < len(hits) <= 4
    doc_ids = [hit["entity"]["doc_id"] for hit in hits]
    for doc_id in set(doc_ids):
        assert doc_ids.count(doc_id) <= 2


def test_retrieve_with_category(client, rag_setup) -> None:
    _, embedder = rag_setup
    hits = rag.retrieve(client, "rag_chunks", "火锅 面条", embedder, top_k=4, category="美食")
    assert hits
    assert {hit["entity"]["category"] for hit in hits} == {"美食"}


def test_retrieve_per_doc_limit_with_long_docs(client, embedder) -> None:
    long_text = "这是一个用于测试切块的句子。" * 40
    synthetic = [
        {"id": 901, "title": "长文一", "category": "测试", "text": long_text},
        {"id": 902, "title": "长文二", "category": "测试", "text": long_text + "尾巴。"},
    ]
    chunks = rag.build_rag_collection(client, "rag_long", synthetic, embedder)
    assert len(chunks) > 4

    hits = rag.retrieve(client, "rag_long", "测试切块", embedder, top_k=4, max_per_doc=2)
    assert 0 < len(hits) <= 4
    doc_ids = [hit["entity"]["doc_id"] for hit in hits]
    for doc_id in set(doc_ids):
        assert doc_ids.count(doc_id) <= 2


def test_answer_and_sources(client, rag_setup) -> None:
    _, embedder = rag_setup
    hits = rag.retrieve(client, "rag_chunks", "Milvus Lite 有哪些限制？", embedder, top_k=3)
    answer, mode, prompt = rag.answer_question("Milvus Lite 有哪些限制？", hits, None)
    assert mode == "离线抽取式回答"
    assert "资料：" in prompt
    assert "Milvus Lite 有哪些限制？" in prompt
    assert answer
    assert len(rag.source_rows(hits)) == len(hits)
