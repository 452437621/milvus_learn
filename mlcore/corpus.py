"""加载并校验示例语料 data/sample_docs.json。"""

from __future__ import annotations

import json

from . import config

DOCS_PATH = config.ROOT / "data" / "sample_docs.json"
REQUIRED_FIELDS = (
    "id",
    "title",
    "category",
    "tags",
    "text",
    "year",
    "rating",
    "price",
    "is_free",
    "author",
    "meta",
)


def load_docs() -> list[dict]:
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"找不到语料文件：{DOCS_PATH}")
    with open(DOCS_PATH, encoding="utf-8") as handle:
        docs = json.load(handle)
    if not isinstance(docs, list) or not docs:
        raise ValueError("语料文件应为非空 JSON 数组")

    for doc in docs:
        missing = [field for field in REQUIRED_FIELDS if field not in doc]
        if missing:
            raise ValueError(f"文档 {doc.get('id')} 缺少字段：{', '.join(missing)}")
    ids = [doc["id"] for doc in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("语料中存在重复 id")
    return docs


def categories(docs: list[dict] | None = None) -> list[str]:
    docs = docs if docs is not None else load_docs()
    return sorted({doc["category"] for doc in docs})


def by_category(category: str, docs: list[dict] | None = None) -> list[dict]:
    docs = docs if docs is not None else load_docs()
    return [doc for doc in docs if doc["category"] == category]


def documents_for_embedding(docs: list[dict] | None = None) -> list[str]:
    """标题 + 标签 + 正文，作为文档向量化的输入。"""
    docs = docs if docs is not None else load_docs()
    return [f"{doc['title']}。{' '.join(doc['tags'])}。{doc['text']}" for doc in docs]
