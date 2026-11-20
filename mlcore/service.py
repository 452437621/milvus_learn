"""Gradio 之外的业务逻辑：Runtime 与 SearchService。

这个模块**不导入 gradio 和 pymilvus**（pymilvus 延迟到函数内部），
方便单元测试，也保证 `import app.gradio_search` 不会隐式创建数据库。

`SearchService.search/ask` 只返回纯数据（NamedTuple），错误以 `error` 字段
返回而不是抛出；弹窗提示（gr.Warning）由 UI 层负责。
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

from . import config, corpus, demo_data, pretty, rag
from .embedding import get_embedder
from .providers import build_chat

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from pymilvus import MilvusClient

INDEX_SPECS: dict[str, dict[str, Any]] = {
    "COSINE + HNSW": {
        "suffix": "cos_hnsw",
        "index_type": "HNSW",
        "metric": "COSINE",
        "index_params": {"M": 16, "efConstruction": 200},
        "search_params": {"ef": 64},
        "direction": "分数越大越相似",
    },
    "L2 + FLAT（精确）": {
        "suffix": "l2_flat",
        "index_type": "FLAT",
        "metric": "L2",
        "index_params": {},
        "search_params": {},
        "direction": "距离越小越相似",
    },
    "IP + HNSW": {
        "suffix": "ip_hnsw",
        "index_type": "HNSW",
        "metric": "IP",
        "index_params": {"M": 16, "efConstruction": 200},
        "search_params": {"ef": 64},
        "direction": "分数越大越相似（伪向量已归一化，IP 与 COSINE 等价）",
    },
}

OUTPUT_FIELDS = ["title", "category", "year", "rating", "text"]


class SearchOutcome(NamedTuple):
    summary: str
    rows: list[list]
    error: str | None = None


class AskOutcome(NamedTuple):
    summary: str
    rows: list[list]
    prompt: str
    error: str | None = None


def build_expression(category: str, min_rating: float, raw_filter: str) -> str:
    """把 UI 上的零散条件拼成 Milvus 过滤表达式。"""
    parts = []
    if category and category != "全部":
        parts.append(f'category == "{category}"')
    if min_rating and min_rating > 0:
        parts.append(f"rating >= {float(min_rating):.2f}")
    raw = (raw_filter or "").strip()
    if raw:
        parts.append(f"({raw})")
    return " and ".join(parts)


@dataclass
class Runtime:
    """一次 UI 运行所需的全部状态。"""

    target: config.DemoTarget
    client: MilvusClient
    docs: list[dict]
    embedder: Any
    chat: Any
    categories: list[str]
    collection_names: dict[str, str]
    rag_name: str
    closed: bool = field(default=False)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.client.close()
        except Exception:  # noqa: BLE001 - 关闭失败不应掩盖原始异常
            pass


def _create_client(target: config.DemoTarget, reset: bool) -> MilvusClient:
    from pymilvus import MilvusClient

    if target.local and target.run_dir is not None:
        if reset:
            shutil.rmtree(target.run_dir, ignore_errors=True)
        target.run_dir.mkdir(parents=True, exist_ok=True)
    return MilvusClient(uri=target.uri)


def _ensure_collections(
    client: MilvusClient,
    target: config.DemoTarget,
    docs: list[dict],
    embedder: Any,
) -> tuple[dict[str, str], str]:
    names: dict[str, str] = {}
    for label, spec in INDEX_SPECS.items():
        name = target.name(f"docs_{spec['suffix']}_{embedder.dim}")
        if not client.has_collection(name):
            demo_data.create_docs_collection(
                client,
                name,
                docs,
                embedder,
                index_type=spec["index_type"],
                metric_type=spec["metric"],
                index_params=spec["index_params"],
            )
        else:
            client.load_collection(collection_name=name)
        names[label] = name

    rag_name = target.name(f"rag_{embedder.dim}")
    if not client.has_collection(rag_name):
        rag.build_rag_collection(client, rag_name, docs, embedder)
    else:
        client.load_collection(collection_name=rag_name)
    return names, rag_name


def create_runtime(stem: str = "ui", *, reset: bool | None = None) -> Runtime:
    """解析目标、创建客户端并准备 3 个检索集合与 1 个 RAG 集合。"""
    target = config.resolve_demo_target(stem)
    client = _create_client(target, bool(config.UI_RESET) if reset is None else reset)
    try:
        docs = corpus.load_docs()
        embedder = get_embedder()
        names, rag_name = _ensure_collections(client, target, docs, embedder)
    except BaseException:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
        raise
    return Runtime(
        target=target,
        client=client,
        docs=docs,
        embedder=embedder,
        chat=build_chat(),
        categories=["全部"] + corpus.categories(docs),
        collection_names=names,
        rag_name=rag_name,
    )


class SearchService:
    """Gradio 回调背后的业务逻辑。"""

    def __init__(self, runtime: Runtime):
        self._runtime = runtime

    @classmethod
    def create(cls, stem: str = "ui", *, reset: bool | None = None) -> SearchService:
        return cls(create_runtime(stem, reset=reset))

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    def search(
        self,
        query: str,
        top_k: int,
        index_label: str,
        category: str,
        min_rating: float,
        raw_filter: str,
    ) -> SearchOutcome:
        query = (query or "").strip()
        if not query:
            return SearchOutcome("请输入查询内容。", [], "请输入查询内容")

        spec = INDEX_SPECS.get(index_label)
        if spec is None:
            message = f"未知索引：{index_label}"
            return SearchOutcome(f"**搜索失败**：{message}", [], message)

        runtime = self._runtime
        expression = build_expression(category, min_rating, raw_filter)
        started = time.perf_counter()
        try:
            hits = runtime.client.search(
                collection_name=runtime.collection_names[index_label],
                data=[runtime.embedder.embed_query(query)],
                limit=int(top_k),
                filter=expression,
                search_params={"metric_type": spec["metric"], "params": spec["search_params"]},
                output_fields=OUTPUT_FIELDS,
            )[0]
        except Exception as exc:  # noqa: BLE001 - 学习项目：把异常转成 UI 提示
            return SearchOutcome(f"**搜索失败**：{exc}", [], f"搜索失败：{exc}")

        elapsed = (time.perf_counter() - started) * 1000
        rows = pretty.hits_to_rows(
            hits,
            fields=("title", "category", "year", "rating"),
            max_text=80,
        )
        summary = (
            f"**查询**：{query}  \n"
            f"**索引**：{index_label}  \n"
            f"**过滤**：`{expression or '无'}`  \n"
            f"**耗时**：{elapsed:.1f} ms ｜ 命中 {len(hits)} 条  \n"
            f"**分数方向**：{spec['direction']}"
        )
        return SearchOutcome(summary, rows)

    def ask(self, question: str, top_k: int, category: str) -> AskOutcome:
        question = (question or "").strip()
        if not question:
            return AskOutcome("请输入问题。", [], "", "请输入问题")

        runtime = self._runtime
        started = time.perf_counter()
        try:
            hits = rag.retrieve(
                runtime.client,
                runtime.rag_name,
                question,
                runtime.embedder,
                top_k=int(top_k),
                category=None if category == "全部" else category,
            )
            answer, mode, prompt = rag.answer_question(question, hits, runtime.chat)
        except Exception as exc:  # noqa: BLE001
            return AskOutcome(f"**RAG 执行失败**：{exc}", [], "", f"RAG 执行失败：{exc}")

        elapsed = (time.perf_counter() - started) * 1000
        summary = (
            f"**模式**：{mode}  \n"
            f"**检索片段**：{len(hits)} 条 ｜ **耗时**：{elapsed:.1f} ms\n\n"
            f"---\n\n{answer}"
        )
        return AskOutcome(summary, rag.source_rows(hits), prompt)

    def close(self) -> None:
        self._runtime.close()

    def __enter__(self) -> SearchService:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
