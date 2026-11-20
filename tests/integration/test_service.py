"""集成：SearchService 端到端业务逻辑（不含 Gradio）。"""

from __future__ import annotations

import pytest

from mlcore.service import INDEX_SPECS, SearchService

pytestmark = pytest.mark.integration


@pytest.fixture
def service(unique_stem):
    svc = SearchService.create(unique_stem, reset=True)
    try:
        yield svc
    finally:
        svc.close()
        svc.close()  # 幂等


def test_collections_ready(service) -> None:
    runtime = service.runtime
    assert set(runtime.collection_names) == set(INDEX_SPECS)
    assert runtime.categories[0] == "全部"
    assert len(runtime.categories) == 8
    for name in runtime.collection_names.values():
        assert service.runtime.client.describe_collection(collection_name=name)
    assert service.runtime.client.has_collection(runtime.rag_name)


@pytest.mark.parametrize("label", list(INDEX_SPECS))
def test_search_each_index(service, label) -> None:
    outcome = service.search("向量数据库怎么选型", 5, label, "全部", 0, "")
    assert outcome.error is None
    assert len(outcome.rows) == 5
    assert label in outcome.summary


def test_search_category_filter(service) -> None:
    outcome = service.search("面条", 5, "COSINE + HNSW", "美食", 0, "")
    assert outcome.error is None
    assert outcome.rows
    assert {row[3] for row in outcome.rows} == {"美食"}


def test_search_min_rating(service) -> None:
    outcome = service.search("学习", 5, "COSINE + HNSW", "全部", 4.5, "")
    assert outcome.error is None
    assert all(row[5] >= 4.5 for row in outcome.rows)


def test_search_invalid_filter(service) -> None:
    outcome = service.search("向量", 5, "COSINE + HNSW", "全部", 0, "rating >")
    assert outcome.error is not None
    assert outcome.summary.startswith("**搜索失败**")
    assert outcome.rows == []


def test_search_empty_query(service) -> None:
    outcome = service.search("   ", 5, "COSINE + HNSW", "全部", 0, "")
    assert outcome.error == "请输入查询内容"
    assert outcome.rows == []


def test_search_unknown_index(service) -> None:
    outcome = service.search("向量", 5, "不存在的索引", "全部", 0, "")
    assert outcome.error is not None
    assert "未知索引" in outcome.summary
    assert outcome.rows == []


def test_ask(service) -> None:
    outcome = service.ask("Milvus Lite 有哪些限制？", 4, "全部")
    assert outcome.error is None
    assert "资料：" in outcome.prompt
    assert 0 < len(outcome.rows) <= 4
    assert "模式" in outcome.summary


def test_ask_empty_question(service) -> None:
    outcome = service.ask("  ", 4, "全部")
    assert outcome.error == "请输入问题"
    assert outcome.prompt == ""


def test_recreate_same_stem_reuses_collections(unique_stem) -> None:
    with SearchService.create(unique_stem, reset=True) as first:
        names = dict(first.runtime.collection_names)
        assert first.search("向量", 3, "COSINE + HNSW", "全部", 0, "").error is None

    with SearchService.create(unique_stem, reset=False) as second:
        assert second.runtime.collection_names == names
        outcome = second.search("向量", 3, "COSINE + HNSW", "全部", 0, "")
        assert outcome.error is None
