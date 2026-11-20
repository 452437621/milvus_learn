"""构建 demo 共用的 25 条语料 collection，避免每个脚本重复 Schema 代码。"""

from __future__ import annotations

from collections.abc import Iterable

from . import corpus

DEFAULT_OUTPUT_FIELDS = ("title", "category", "year", "rating", "is_free")


def doc_rows(docs: list[dict], embedder) -> list[dict]:
    """把语料转成 Milvus 行（含 embedding）。"""
    texts = corpus.documents_for_embedding(docs)
    vectors = embedder.embed_documents(texts)
    rows = []
    for doc, vector in zip(docs, vectors):
        rows.append(
            {
                "id": int(doc["id"]),
                "title": doc["title"],
                "category": doc["category"],
                "tags": list(doc["tags"]),
                "text": doc["text"],
                "year": int(doc["year"]),
                "rating": float(doc["rating"]),
                "price": float(doc["price"]),
                "is_free": bool(doc["is_free"]),
                "author": doc["author"],
                "meta": dict(doc["meta"]),
                "embedding": vector,
            }
        )
    return rows


def default_index_params(index_type: str) -> dict:
    index_type = index_type.upper()
    if index_type == "HNSW":
        return {"M": 16, "efConstruction": 200}
    if index_type == "IVF_FLAT":
        return {"nlist": 8}
    return {}


def create_docs_collection(
    client,
    name: str,
    docs: list[dict],
    embedder,
    index_type: str = "HNSW",
    metric_type: str = "COSINE",
    index_params: dict | None = None,
    scalar_indexes: Iterable[str] = (),
    load: bool = True,
    insert: bool = True,
) -> list[dict]:
    """创建带完整标量字段的 collection 并写入 25 条语料。"""
    from pymilvus import DataType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("title", DataType.VARCHAR, max_length=200)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field(
        "tags",
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=8,
        max_length=64,
    )
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("year", DataType.INT32)
    schema.add_field("rating", DataType.FLOAT)
    schema.add_field("price", DataType.FLOAT)
    schema.add_field("is_free", DataType.BOOL)
    schema.add_field(
        "author", DataType.VARCHAR, max_length=100, nullable=True, default_value="佚名"
    )
    schema.add_field("meta", DataType.JSON)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=embedder.dim)

    client.create_collection(collection_name=name, schema=schema)

    index_params_obj = client.prepare_index_params()
    index_params_obj.add_index(
        field_name="embedding",
        index_type=index_type.upper(),
        metric_type=metric_type,
        params=index_params if index_params is not None else default_index_params(index_type),
    )
    for field in scalar_indexes:
        index_params_obj.add_index(field_name=field, index_type="INVERTED")
    client.create_index(collection_name=name, index_params=index_params_obj)

    rows = doc_rows(docs, embedder)
    if insert:
        client.insert(collection_name=name, data=rows)
    if load:
        client.load_collection(collection_name=name)
    return rows


def search(
    client,
    name: str,
    query_vector: list[float],
    limit: int = 5,
    filter: str = "",
    metric_type: str = "COSINE",
    params: dict | None = None,
    output_fields: Iterable[str] | None = None,
    **kwargs,
):
    search_params = {"metric_type": metric_type, "params": params or {}}
    return client.search(
        collection_name=name,
        data=[query_vector],
        limit=limit,
        filter=filter,
        search_params=search_params,
        output_fields=list(output_fields or DEFAULT_OUTPUT_FIELDS),
        **kwargs,
    )
