"""embedding 模块：伪向量性质与 get_embedder 缓存。"""

from __future__ import annotations

import math

import pytest

from mlcore import config
from mlcore.embedding import PseudoEmbedder, cosine, get_embedder, reset_embedder
from mlcore.sparse import MAX_SPARSE_INDEX, manual_sparse

pytestmark = pytest.mark.unit


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def test_dimension_and_norm() -> None:
    embedder = PseudoEmbedder(dim=64)
    vector = embedder.embed_query("向量数据库")
    assert len(vector) == 64
    assert _norm(vector) == pytest.approx(1.0, abs=1e-9)


def test_deterministic_and_nfkc_equivalent() -> None:
    embedder = PseudoEmbedder(dim=128)
    left = embedder.embed_query("Milvus 向量检索")
    right = embedder.embed_query("Ｍｉｌｖｕｓ 向量检索")
    assert left == right


def test_related_closer_than_unrelated() -> None:
    embedder = PseudoEmbedder()
    base = embedder.embed_query("向量数据库怎么选型")
    related = embedder.embed_query("Milvus 的部署形态与适用场景")
    unrelated = embedder.embed_query("四川火锅和广式早茶")
    assert cosine(base, related) > cosine(base, unrelated)


def test_empty_text_fallback_is_normalised() -> None:
    embedder = PseudoEmbedder(dim=32)
    vector = embedder.embed_query("")
    assert len(vector) == 32
    assert _norm(vector) == pytest.approx(1.0, abs=1e-9)


def test_cosine_edges() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError):
        cosine([1.0], [1.0, 2.0])


def test_get_embedder_cached() -> None:
    assert get_embedder() is get_embedder()


def test_reset_embedder_returns_new_instance() -> None:
    first = get_embedder()
    reset_embedder()
    second = get_embedder()
    assert first is not second
    assert type(first) is type(second)


def test_openai_without_key_falls_back(monkeypatch, capsys) -> None:
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    reset_embedder()
    embedder = get_embedder()
    assert isinstance(embedder, PseudoEmbedder)
    assert "回退离线伪向量" in capsys.readouterr().out


# ---- 手动稀疏向量（mlcore.sparse）----


def test_sparse_empty_text() -> None:
    assert manual_sparse("") == {}
    # 纯标点/空白没有二元组，也不应该产生稀疏项
    assert manual_sparse("。，！ ") == {}


def test_sparse_keys_and_values() -> None:
    vector = manual_sparse("向量数据库 Milvus")
    assert vector
    for key, value in vector.items():
        assert isinstance(key, int)
        assert 0 <= key <= MAX_SPARSE_INDEX
        assert value > 0


def test_sparse_deterministic() -> None:
    assert manual_sparse("向量 检索") == manual_sparse("向量 检索")
    assert manual_sparse("向量 检索") != manual_sparse("火锅 麻辣")


def test_sparse_query_overlaps_document() -> None:
    query = manual_sparse("火锅 麻辣")
    document = manual_sparse("四川火锅以牛油红汤为底，麻辣鲜香")
    assert set(query) & set(document)
