"""迷你 RAG：切块、建索引、检索、Prompt 与离线回答。"""

from __future__ import annotations

from dataclasses import dataclass

from .pretty import hit_entity, hit_score

SENTENCE_END = "。！？!?；;\n"

SYSTEM_PROMPT = (
    "你是一个严谨的中文学习助手。请只根据用户提供的编号资料回答问题。"
    "如果资料不足以回答，请明确说“资料中没有足够信息”，不要编造。"
    "回答时用 [1]、[2] 标注引用来源。"
)


@dataclass
class Chunk:
    chunk_id: int
    doc_id: int
    title: str
    category: str
    chunk_index: int
    text: str
    embedding_text: str


def chunk_text(text: str, max_chars: int = 180, overlap: int = 30) -> list[str]:
    """按句子边界尽量切块，超长时硬切；相邻块保留少量重叠。"""
    text = " ".join((text or "").split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    overlap = max(0, min(overlap, max_chars // 2))
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window_start = start + max_chars // 2
            window = text[window_start:end]
            cut = max((window.rfind(char) for char in SENTENCE_END), default=-1)
            if cut >= 0:
                end = window_start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(docs: list[dict]) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_id = 1
    for doc in docs:
        for index, piece in enumerate(chunk_text(doc["text"])):
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=int(doc["id"]),
                    title=doc["title"],
                    category=doc["category"],
                    chunk_index=index,
                    text=piece,
                    embedding_text=f"{doc['title']}。{piece}",
                )
            )
            chunk_id += 1
    return chunks


def build_rag_collection(client, name: str, docs: list[dict], embedder) -> list[Chunk]:
    """创建 RAG 用的 chunk collection（HNSW + COSINE），返回 chunk 列表。"""
    from pymilvus import DataType, MilvusClient

    from .milvus_utils import drop_collection_quietly, insert_in_batches

    drop_collection_quietly(client, name)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("doc_id", DataType.INT64)
    schema.add_field("title", DataType.VARCHAR, max_length=200)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field("chunk_index", DataType.INT32)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=embedder.dim)

    client.create_collection(collection_name=name, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_index(collection_name=name, index_params=index_params)

    chunks = build_chunks(docs)
    vectors = embedder.embed_documents([chunk.embedding_text for chunk in chunks])
    rows = [
        {
            "id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "category": chunk.category,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    insert_in_batches(client, name, rows)
    client.load_collection(collection_name=name)
    return chunks


def retrieve(
    client,
    name: str,
    question: str,
    embedder,
    top_k: int = 4,
    category: str | None = None,
    max_per_doc: int = 2,
) -> list[dict]:
    expression = f'category == "{category}"' if category else ""
    results = client.search(
        collection_name=name,
        data=[embedder.embed_query(question)],
        anns_field="embedding",
        limit=max(top_k * 3, top_k),
        filter=expression,
        output_fields=["doc_id", "title", "category", "chunk_index", "text"],
    )
    hits = results[0] if results else []

    kept: list[dict] = []
    per_doc: dict[int, int] = {}
    for hit in hits:
        entity = hit_entity(hit)
        doc_id = entity.get("doc_id")
        if per_doc.get(doc_id, 0) >= max_per_doc:
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        kept.append(hit)
        if len(kept) >= top_k:
            break
    return kept


def build_user_prompt(question: str, hits: list[dict]) -> str:
    lines = ["资料："]
    for index, hit in enumerate(hits, start=1):
        entity = hit_entity(hit)
        lines.append(f"[{index}] 《{entity.get('title', '')}》 分类：{entity.get('category', '')}")
        lines.append(f"内容：{entity.get('text', '')}")
    lines.extend(["", f"问题：{question}"])
    return "\n".join(lines)


def build_prompt(question: str, hits: list[dict]) -> str:
    return f"{SYSTEM_PROMPT}\n\n{build_user_prompt(question, hits)}"


def _first_sentence(text: str, limit: int = 110) -> str:
    text = " ".join(str(text).split())
    for index, char in enumerate(text):
        if char in SENTENCE_END:
            return text[: index + 1]
    return text[:limit] + ("…" if len(text) > limit else "")


def offline_answer(hits: list[dict], max_chars: int = 320) -> str:
    if not hits:
        return "（离线抽取式回答）没有检索到相关片段，无法回答。"

    parts: list[str] = []
    total = 0
    for index, hit in enumerate(hits, start=1):
        entity = hit_entity(hit)
        piece = f"[{index}] {_first_sentence(entity.get('text', ''))}"
        if parts and total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "（离线抽取式回答，未调用大模型）\n" + "\n".join(parts)


def answer_question(question: str, hits: list[dict], chat=None) -> tuple[str, str, str]:
    """返回 (回答, 模式说明, 完整 Prompt)。"""
    prompt = build_prompt(question, hits)
    if chat is not None:
        try:
            reply = chat.generate(SYSTEM_PROMPT, build_user_prompt(question, hits))
        except Exception as exc:  # noqa: BLE001 - 联网失败回退离线
            print(f"[警告] 大模型调用失败，改用离线回答：{exc}")
            reply = None
        if reply:
            return reply, "大模型（OpenAI 兼容接口）", prompt
    return offline_answer(hits), "离线抽取式回答", prompt


def source_rows(hits: list[dict]) -> list[list]:
    rows: list[list] = []
    for index, hit in enumerate(hits, start=1):
        entity = hit_entity(hit)
        text = " ".join(str(entity.get("text", "")).split())
        if len(text) > 80:
            text = text[:80] + "…"
        score = hit_score(hit)
        rows.append(
            [
                index,
                None if score is None else round(score, 4),
                entity.get("title", ""),
                entity.get("category", ""),
                entity.get("chunk_index", ""),
                text,
            ]
        )
    return rows
