"""BM25 全文检索集合的构建与查询（原生 BM25 / 手动稀疏回退）。"""

from __future__ import annotations

from dataclasses import dataclass

from . import corpus
from .sparse import manual_sparse

try:  # jieba 首次加载会打印“Prefix dict has been built”，这里静音
    import logging

    import jieba

    jieba.setLogLevel(logging.WARNING)
except ImportError:  # pragma: no cover
    pass


@dataclass
class FullTextCollection:
    name: str
    mode: str  # "native" 或 "manual"
    analyzer: str
    count: int


def _bm25_rows(docs: list[dict], embedder, manual: bool) -> list[dict]:
    texts = corpus.documents_for_embedding(docs)
    dense_vectors = embedder.embed_documents(texts)
    rows: list[dict] = []
    for doc, dense in zip(docs, dense_vectors):
        full_text = f"{doc['title']}。{doc['text']}"
        row = {
            "id": int(doc["id"]),
            "title": doc["title"],
            "category": doc["category"],
            "text": full_text,
            "dense": dense,
        }
        if manual:
            row["sparse"] = manual_sparse(full_text)
        rows.append(row)
    return rows


def _create_native(client, name: str, docs: list[dict], embedder, analyzer: str) -> None:
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("title", DataType.VARCHAR, max_length=200)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=4096,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"type": analyzer},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=embedder.dim)
    schema.add_function(
        Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"],
        )
    )
    client.create_collection(collection_name=name, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={},
    )
    index_params.add_index(
        field_name="dense",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_index(collection_name=name, index_params=index_params)
    client.insert(collection_name=name, data=_bm25_rows(docs, embedder, manual=False))
    client.load_collection(collection_name=name)


def _create_manual(client, name: str, docs: list[dict], embedder) -> None:
    from pymilvus import DataType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("title", DataType.VARCHAR, max_length=200)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=embedder.dim)
    client.create_collection(collection_name=name, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={},
    )
    index_params.add_index(
        field_name="dense",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_index(collection_name=name, index_params=index_params)
    client.insert(collection_name=name, data=_bm25_rows(docs, embedder, manual=True))
    client.load_collection(collection_name=name)


def create_fulltext_collection(
    client,
    name: str,
    docs: list[dict],
    embedder,
    mode: str = "auto",
) -> FullTextCollection:
    """mode: auto / native / manual。

    auto 先尝试原生 BM25（jieba -> standard），失败则回退手动稀疏 + IP。
    """
    from .milvus_utils import drop_collection_quietly

    drop_collection_quietly(client, name)

    if mode in ("auto", "native"):
        last_error = None
        for analyzer in ("jieba", "standard"):
            try:
                _create_native(client, name, docs, embedder, analyzer)
                return FullTextCollection(
                    name=name, mode="native", analyzer=analyzer, count=len(docs)
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                drop_collection_quietly(client, name)
        if mode == "native":
            raise RuntimeError(f"原生 BM25 不可用：{last_error}") from last_error
        print(f"[提示] 原生 BM25 不可用，回退手动稀疏：{last_error}")

    _create_manual(client, name, docs, embedder)
    return FullTextCollection(name=name, mode="manual", analyzer="none", count=len(docs))


def bm25_search(client, collection: FullTextCollection, query: str, limit: int = 5):
    if collection.mode == "native":
        return client.search(
            collection_name=collection.name,
            data=[query],
            anns_field="sparse",
            limit=limit,
            output_fields=["title", "category", "text"],
        )
    return client.search(
        collection_name=collection.name,
        data=[manual_sparse(query)],
        anns_field="sparse",
        limit=limit,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["title", "category", "text"],
    )


def sparse_request_data(collection: FullTextCollection, query: str):
    """供 hybrid search 组装 AnnSearchRequest 使用。"""
    if collection.mode == "native":
        return [query], "BM25"
    return [manual_sparse(query)], "IP"
