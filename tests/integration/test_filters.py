"""集成：标量过滤表达式（计数取自 data/sample_docs.json）。"""

from __future__ import annotations

import pytest
from pymilvus.exceptions import MilvusException

from mlcore import demo_data

pytestmark = pytest.mark.integration

CASES = [
    ('category == "美食"', 4),
    ("year >= 2022", 17),
    ("rating > 4.5", 15),
    ('category in ["旅行", "电影"]', 7),
    ('author like "美食%"', 3),
    ('meta["level"] == "进阶"', 7),
    ('array_contains(tags, "Milvus")', 1),
    ("is_free == true and rating >= 4.5", 14),
]


@pytest.mark.parametrize("expression,expected", CASES)
def test_filter_counts(client, docs_collection, expression, expected) -> None:
    rows = client.query(
        collection_name=docs_collection,
        filter=expression,
        output_fields=["id"],
    )
    assert len(rows) == expected


def test_invalid_expression_raises(client, docs_collection) -> None:
    with pytest.raises(MilvusException):
        client.query(collection_name=docs_collection, filter="rating >", output_fields=["id"])


def test_search_with_filter_is_pure(client, docs_collection, embedder) -> None:
    hits = demo_data.search(
        client,
        docs_collection,
        embedder.embed_query("面条"),
        limit=3,
        filter='category == "美食"',
    )[0]
    assert hits
    assert {hit["entity"]["category"] for hit in hits} == {"美食"}

    unfiltered = demo_data.search(client, docs_collection, embedder.embed_query("面条"), limit=3)[0]
    assert len(unfiltered) == 3
