"""可选的 OpenAI 兼容接口（embedding / chat）。

只使用标准库 urllib，避免额外依赖。默认不会联网：只有用户显式设置
EMBEDDING_PROVIDER=openai 或 RAG_LLM=openai 且提供 API Key 时才会调用。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config


class ProviderError(RuntimeError):
    """外部服务调用失败。"""


def _post_json(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        raise ProviderError(f"HTTP {exc.code}: {body}") from exc
    except Exception as exc:  # noqa: BLE001 - 统一转成可读错误
        raise ProviderError(str(exc)) from exc


def _endpoint(base_url: str, suffix: str) -> str:
    return base_url.rstrip("/") + suffix


class OpenAICompatibleEmbedder:
    """调用 /embeddings，批量生成向量。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dim: int | None = None,
    ):
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_EMBED_MODEL
        self.dim = int(dim or config.OPENAI_EMBED_DIM)
        self.name = f"openai-{self.model}"
        if not self.api_key:
            raise ProviderError("未配置 OPENAI_API_KEY")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            payload = {"model": self.model, "input": batch}
            response = _post_json(
                _endpoint(self.base_url, "/embeddings"),
                payload,
                self.api_key,
                config.LLM_TIMEOUT,
            )
            vectors.extend(item["embedding"] for item in response["data"])
        if vectors:
            self.dim = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        if not vectors:
            raise ProviderError("embedding 服务返回空结果")
        return vectors[0]


class OpenAICompatibleChat:
    """调用 /chat/completions，非流式返回文本。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_CHAT_MODEL
        self.timeout = int(timeout or config.LLM_TIMEOUT)
        if not self.api_key:
            raise ProviderError("未配置 OPENAI_API_KEY")

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        response = _post_json(
            _endpoint(self.base_url, "/chat/completions"),
            payload,
            self.api_key,
            self.timeout,
        )
        return response["choices"][0]["message"]["content"].strip()


class OfflineChat:
    """离线模式：不生成，由 RAG 模块走抽取式回答。"""

    def generate(self, system: str, user: str) -> None:  # noqa: ARG002
        return None


def build_chat():
    """按配置返回聊天后端；失败时回退离线。"""
    if config.RAG_LLM == "openai":
        try:
            return OpenAICompatibleChat()
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 大模型初始化失败，回退离线回答：{exc}")
    return OfflineChat()
