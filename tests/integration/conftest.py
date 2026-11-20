"""集成测试夹具：每个测试一个独立 collection。"""

from __future__ import annotations

import pytest

from mlcore import demo_data


@pytest.fixture
def docs_collection(client, docs, embedder, unique_stem) -> str:
    name = f"docs_{unique_stem}"
    demo_data.create_docs_collection(
        client,
        name,
        docs,
        embedder,
        scalar_indexes=("category", "year", "is_free"),
    )
    client.flush(collection_name=name)
    return name
