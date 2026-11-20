"""providers 模块：OpenAI 兼容接口（全部使用假 urlopen，不联网）。"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from mlcore import config, providers
from mlcore.providers import (
    OfflineChat,
    OpenAICompatibleChat,
    OpenAICompatibleEmbedder,
    ProviderError,
    _endpoint,
    build_chat,
)

pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._data


def _install_fake_urlopen(monkeypatch, calls: list[dict], handler):
    def fake_urlopen(request, timeout=None):
        calls.append(
            {
                "url": request.full_url,
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
                "auth": request.headers.get("Authorization"),
            }
        )
        return handler(len(calls))

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)


def test_missing_api_key() -> None:
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAICompatibleEmbedder(api_key="")
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAICompatibleChat(api_key="")


def test_endpoint_join() -> None:
    assert _endpoint("http://x/v1/", "/embeddings") == "http://x/v1/embeddings"


def test_embedder_batches(monkeypatch) -> None:
    calls: list[dict] = []

    def handler(batch_no: int) -> _FakeResponse:
        batch_size = 64 if batch_no == 1 else 1
        return _FakeResponse({"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in range(batch_size)]})

    _install_fake_urlopen(monkeypatch, calls, handler)
    embedder = OpenAICompatibleEmbedder(base_url="http://fake/v1", api_key="k", model="m", dim=3)
    vectors = embedder.embed_documents([f"text-{i}" for i in range(65)])
    assert len(vectors) == 65
    assert len(calls) == 2
    assert calls[0]["url"] == "http://fake/v1/embeddings"
    assert calls[0]["payload"]["model"] == "m"
    assert len(calls[0]["payload"]["input"]) == 64
    assert len(calls[1]["payload"]["input"]) == 1
    assert calls[0]["auth"] == "Bearer k"
    assert embedder.dim == 3


def test_embed_query(monkeypatch) -> None:
    calls: list[dict] = []
    _install_fake_urlopen(
        monkeypatch,
        calls,
        lambda _n: _FakeResponse({"data": [{"embedding": [1.0, 0.0]}]}),
    )
    embedder = OpenAICompatibleEmbedder(base_url="http://fake/v1", api_key="k", dim=2)
    assert embedder.embed_query("hi") == [1.0, 0.0]


def test_chat_payload_and_strip(monkeypatch) -> None:
    calls: list[dict] = []
    _install_fake_urlopen(
        monkeypatch,
        calls,
        lambda _n: _FakeResponse({"choices": [{"message": {"content": "  回答  "}}]}),
    )
    chat = OpenAICompatibleChat(base_url="http://fake/v1", api_key="k", model="chat", timeout=5)
    assert chat.generate("sys", "user") == "回答"
    payload = calls[0]["payload"]
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["messages"][1] == {"role": "user", "content": "user"}
    assert payload["temperature"] == 0.2
    assert calls[0]["timeout"] == 5


def test_http_error_becomes_provider_error(monkeypatch) -> None:
    def raising(request, timeout=None):
        raise urllib.error.HTTPError("http://fake", 401, "Unauthorized", {}, io.BytesIO(b"denied"))

    monkeypatch.setattr(providers.urllib.request, "urlopen", raising)
    embedder = OpenAICompatibleEmbedder(base_url="http://fake", api_key="k")
    with pytest.raises(ProviderError) as excinfo:
        embedder.embed_query("hi")
    assert "HTTP 401" in str(excinfo.value)


def test_generic_error_becomes_provider_error(monkeypatch) -> None:
    def raising(request, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(providers.urllib.request, "urlopen", raising)
    chat = OpenAICompatibleChat(base_url="http://fake", api_key="k")
    with pytest.raises(ProviderError, match="network down"):
        chat.generate("s", "u")


def test_build_chat_offline() -> None:
    monkeypatch_target = config.RAG_LLM
    try:
        config.RAG_LLM = "offline"
        assert isinstance(build_chat(), OfflineChat)
    finally:
        config.RAG_LLM = monkeypatch_target


def test_build_chat_openai_without_key_falls_back(capsys) -> None:
    original = (config.RAG_LLM, config.OPENAI_API_KEY)
    try:
        config.RAG_LLM = "openai"
        config.OPENAI_API_KEY = ""
        assert isinstance(build_chat(), OfflineChat)
        assert "回退离线回答" in capsys.readouterr().out
    finally:
        config.RAG_LLM, config.OPENAI_API_KEY = original


def test_offline_chat_returns_none() -> None:
    assert OfflineChat().generate("s", "u") is None
