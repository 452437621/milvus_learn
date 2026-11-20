"""端到端：启动 Gradio 界面并验证 HTTP 200 与回调。"""

from __future__ import annotations

import socket
import time
import urllib.request

import pytest

from app.gradio_search import build_app, make_callbacks
from mlcore.service import SearchService

pytestmark = pytest.mark.e2e


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _launch(app, port: int):
    return app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=False,
        prevent_thread_lock=True,
        show_error=True,
        quiet=True,
    )


def test_ui_serves_and_callbacks_work(unique_stem) -> None:
    service = SearchService.create(unique_stem, reset=True)
    app = None
    last_error: Exception | None = None
    try:
        for _ in range(3):
            port = _free_port()
            try:
                app = build_app(service)
                _launch(app, port)
                break
            except OSError as exc:  # 端口被占用，换一个
                last_error = exc
                app = None
                time.sleep(0.2)
        assert app is not None, last_error

        url = f"http://127.0.0.1:{port}/"
        deadline = time.time() + 30
        status = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    status = response.status
                    body = response.read(200).decode("utf-8", "replace")
                break
            except OSError:
                time.sleep(0.5)
        assert status == 200
        assert "<html" in body.lower()

        callbacks = make_callbacks(service)
        summary, rows = callbacks["search"]("向量数据库怎么选型", 3, "COSINE + HNSW", "全部", 0, "")
        assert len(rows) == 3
        assert "分数方向" in summary

        rag_summary, sources, prompt = callbacks["rag"]("Milvus Lite 有哪些限制？", 3, "全部")
        assert prompt
        assert "资料：" in prompt
        assert sources
        assert "模式" in rag_summary

        empty_summary, empty_rows = callbacks["search"]("  ", 3, "COSINE + HNSW", "全部", 0, "")
        assert empty_rows == []
        assert "请输入查询内容" in empty_summary
    finally:
        if app is not None:
            try:
                app.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响断言
                pass
        service.close()
        time.sleep(0.2)
