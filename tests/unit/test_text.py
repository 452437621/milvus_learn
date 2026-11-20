"""text 模块：规范化与中文分词。"""

from __future__ import annotations

import pytest

from mlcore.text import normalize, terms_for_sparse, tokenize

pytestmark = pytest.mark.unit


def test_normalize_fullwidth_and_whitespace() -> None:
    assert normalize("Ｍｉｌｖｕｓ   是  向量库") == "milvus 是 向量库"
    assert normalize("") == ""
    assert normalize("   ") == ""


def test_normalize_none_safe() -> None:
    assert normalize(None) == ""


def test_tokenize_cjk_and_ascii() -> None:
    tokens = tokenize("火锅 麻辣 Milvus")
    kinds = {token: kind for token, kind, _ in tokens}
    assert kinds["w:milvus"] == "word"
    assert kinds["b:火锅"] == "bigram"
    assert kinds["c:火"] == "char"
    weights = {token: weight for token, _, weight in tokens}
    assert weights["w:milvus"] == 1.4
    assert weights["b:火锅"] == 1.0
    assert weights["c:火"] == 0.35


def test_stop_char_filtered() -> None:
    tokens = tokenize("的了和")
    # 单字被停用字表过滤，但相邻二元组仍然保留
    kinds = [kind for _, kind, _ in tokens]
    assert "char" not in kinds
    assert kinds == ["bigram", "bigram"]


def test_single_cjk_char_has_higher_weight() -> None:
    weights = {token: weight for token, kind, weight in tokenize("龙")}
    assert weights["c:龙"] == 0.8


def test_tokenize_is_deterministic() -> None:
    assert tokenize("向量 检索 Milvus") == tokenize("向量 检索 Milvus")


def test_terms_for_sparse_excludes_chars() -> None:
    terms = terms_for_sparse("向量数据库 Milvus")
    assert "b:向量" in terms
    assert "w:milvus" in terms
    assert all(not term.startswith("c:") for term in terms)
