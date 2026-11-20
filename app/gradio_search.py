"""Gradio 交互界面：向量检索 + Mini RAG（薄 UI 层）。

业务逻辑在 `mlcore.service`；这里只负责布局和把错误转成 gr.Warning。

启动：
    venv\\Scripts\\python.exe app\\gradio_search.py
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import sys  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from mlcore import config  # noqa: E402
from mlcore.service import (  # noqa: E402
    INDEX_SPECS,
    SearchService,
    build_expression,
)

SEARCH_HEADERS = ["排名", "分数", "标题", "分类", "年份", "评分", "摘要"]
SOURCE_HEADERS = ["排名", "分数", "标题", "分类", "chunk", "摘要"]

__all__ = [
    "INDEX_SPECS",
    "SEARCH_HEADERS",
    "SOURCE_HEADERS",
    "build_app",
    "build_expression",
    "make_callbacks",
    "main",
]


def make_callbacks(service: SearchService) -> dict[str, Callable]:
    """返回可直接绑定到 Gradio 组件的回调。"""

    def do_search(query, top_k, index_label, category, min_rating, raw_filter):
        summary, rows, error = service.search(
            query, top_k, index_label, category, min_rating, raw_filter
        )
        if error:
            gr.Warning(error)
        return summary, rows

    def do_rag(question, top_k, category):
        summary, rows, prompt, error = service.ask(question, top_k, category)
        if error:
            gr.Warning(error)
        return summary, rows, prompt

    return {"search": do_search, "rag": do_rag}


def build_app(service: SearchService) -> gr.Blocks:
    runtime = service.runtime
    callbacks = make_callbacks(service)

    with gr.Blocks(title="Milvus 学习 Demo", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Milvus 学习 Demo\n"
            f"数据：{len(runtime.docs)} 条中文语料 ｜ 向量：`{runtime.embedder.name}`（离线伪向量）"
            f" ｜ 数据目录：`{runtime.target.uri}`\n\n"
            "> 伪向量用于学习 Milvus 的检索流程，不代表真实语义模型。"
        )

        with gr.Tab("向量检索"):
            query = gr.Textbox(
                value="向量数据库怎么选型",
                label="查询",
                placeholder="输入自然语言查询",
            )
            with gr.Row():
                top_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-K")
                index_label = gr.Dropdown(
                    choices=list(INDEX_SPECS),
                    value="COSINE + HNSW",
                    label="索引 / 度量",
                )
                category = gr.Dropdown(
                    choices=runtime.categories,
                    value="全部",
                    label="分类过滤",
                )
            with gr.Row():
                min_rating = gr.Slider(minimum=0, maximum=5, value=0, step=0.1, label="最低评分")
                raw_filter = gr.Textbox(
                    value="",
                    label="原始过滤表达式（可选）",
                    placeholder="例如：year >= 2020 and is_free == true",
                )
            search_button = gr.Button("搜索", variant="primary")
            summary = gr.Markdown("输入查询后点击“搜索”。")
            table = gr.Dataframe(
                headers=SEARCH_HEADERS,
                value=[],
                interactive=False,
                wrap=True,
                max_height=420,
                label="检索结果",
            )
            search_button.click(
                callbacks["search"],
                inputs=[query, top_k, index_label, category, min_rating, raw_filter],
                outputs=[summary, table],
            )

        with gr.Tab("迷你 RAG"):
            question = gr.Textbox(
                value="Milvus Lite 有哪些限制？",
                label="问题",
                lines=2,
            )
            with gr.Row():
                rag_top_k = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="检索片段数")
                rag_category = gr.Dropdown(
                    choices=runtime.categories,
                    value="全部",
                    label="分类过滤",
                )
            rag_button = gr.Button("生成回答", variant="primary")
            rag_summary = gr.Markdown("输入问题后点击“生成回答”。")
            rag_sources = gr.Dataframe(
                headers=SOURCE_HEADERS,
                value=[],
                interactive=False,
                wrap=True,
                max_height=320,
                label="来源片段",
            )
            with gr.Accordion("查看实际发送的 Prompt", open=False):
                prompt_box = gr.Textbox(lines=12, interactive=False, label="Prompt")
            rag_button.click(
                callbacks["rag"],
                inputs=[question, rag_top_k, rag_category],
                outputs=[rag_summary, rag_sources, prompt_box],
            )

        gr.Markdown(
            "离线默认：伪向量 + 抽取式回答。配置 `.env` 中的 "
            "`EMBEDDING_PROVIDER=openai` 或 `RAG_LLM=openai` 可接入真实服务。"
        )

    return demo


def main() -> None:
    settings = config.ui_settings()
    service = SearchService.create("ui")
    try:
        app = build_app(service)
        app.launch(
            server_name="127.0.0.1",
            server_port=settings["port"],
            share=False,
            inbrowser=True,
            show_error=True,
        )
    finally:
        service.close()


if __name__ == "__main__":
    main()
