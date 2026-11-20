"""Milvus 连接、数据目录隔离与少量兼容适配。"""

from __future__ import annotations

import inspect
import shutil
from collections.abc import Iterator
from typing import Any

from . import config


class MilvusUnavailable(RuntimeError):
    """pymilvus / milvus-lite 不可用。"""


def import_client_class():
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:  # pragma: no cover
        raise MilvusUnavailable(
            "未安装 pymilvus。请运行：venv\\Scripts\\python.exe -m pip install pymilvus milvus-lite"
        ) from exc
    return MilvusClient


def _remove_dir(path) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise RuntimeError(
            f"无法清理数据目录 {path}。请确认没有其他 Python 进程/UI 正在使用它，然后重试。"
        ) from exc


class DemoSession:
    """一个 demo 的隔离会话。

    本地模式：删除并重建 data/runs/<stem>/，使用独立 .db 文件。
    远程模式：连接 MILVUS_URI，并清理带 learn_<stem>_ 前缀的 collection。
    """

    def __init__(self, stem: str):
        self.stem = stem
        self.target = config.resolve_demo_target(stem)
        self.client = None

    def __enter__(self) -> DemoSession:
        MilvusClient = import_client_class()
        if self.target.local and self.target.run_dir is not None:
            _remove_dir(self.target.run_dir)
            self.target.run_dir.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(uri=self.target.uri)
        if not self.target.local:
            self._drop_prefixed()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:  # noqa: BLE001 - 关闭失败不应掩盖原始异常
                pass
            self.client = None

    def name(self, suffix: str) -> str:
        return self.target.name(suffix)

    def _drop_prefixed(self) -> None:
        prefix = self.target.collection_prefix
        try:
            names = self.client.list_collections()
        except Exception:  # noqa: BLE001
            return
        for name in names:
            if name.startswith(prefix):
                drop_collection_quietly(self.client, name)

    # 便捷访问
    @property
    def local(self) -> bool:
        return self.target.local

    @property
    def uri(self) -> str:
        return self.target.uri


def drop_collection_quietly(client, name: str) -> None:
    try:
        client.drop_collection(name)
    except Exception:  # noqa: BLE001 - 不存在时忽略
        pass


def has_method(client, name: str) -> bool:
    return callable(getattr(client, name, None))


def method_signature(client, name: str) -> str:
    method = getattr(client, name, None)
    if not callable(method):
        return f"{name}: 不可用"
    try:
        return f"{name}{inspect.signature(method)}"
    except (TypeError, ValueError):
        return f"{name}: 签名不可读"


def count_entities(client, name: str) -> int | None:
    """优先 count(*)，失败回退 collection stats。"""
    try:
        rows = client.query(
            collection_name=name,
            filter="",
            output_fields=["count(*)"],
        )
        if rows:
            row = rows[0]
            for key in ("count(*)", "count"):
                if key in row:
                    return int(row[key])
    except Exception:  # noqa: BLE001
        pass
    try:
        stats = client.get_collection_stats(collection_name=name)
        value = stats.get("row_count")
        return int(value) if value is not None else None
    except Exception:  # noqa: BLE001
        return None


def ensure_loaded(client, name: str) -> None:
    try:
        client.load_collection(collection_name=name)
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if "loaded" in message or "load" in message:
            return
        raise


def list_collections(client) -> list[str]:
    try:
        return list(client.list_collections())
    except Exception:  # noqa: BLE001
        return []


def iter_batches(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def insert_in_batches(client, name: str, rows: list[dict], size: int = 64) -> int:
    inserted = 0
    for batch in iter_batches(rows, size):
        result = client.insert(collection_name=name, data=batch)
        try:
            inserted += int(result.get("insert_count", len(batch)))
        except (AttributeError, TypeError):
            inserted += len(batch)
    return inserted
