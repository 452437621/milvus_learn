"""milvus_utils：不依赖真实 Milvus 的纯函数与 DemoSession 生命周期。"""

from __future__ import annotations

import pytest

from mlcore import config, milvus_utils
from mlcore.milvus_utils import (
    DemoSession,
    drop_collection_quietly,
    has_method,
    insert_in_batches,
    iter_batches,
    list_collections,
    method_signature,
)

pytestmark = pytest.mark.unit


class _FakeClient:
    def __init__(self, names=None, fail_drop: bool = False):
        self.names = list(names or [])
        self.dropped: list[str] = []
        self.inserted: list[list[dict]] = []
        self.closed = False
        self.fail_drop = fail_drop

    def list_collections(self):
        return list(self.names)

    def drop_collection(self, name):
        if self.fail_drop:
            raise RuntimeError("nope")
        self.dropped.append(name)
        if name in self.names:
            self.names.remove(name)

    def insert(self, collection_name, data):
        self.inserted.append(data)
        return {"insert_count": len(data)}

    def close(self):
        self.closed = True


def test_iter_batches() -> None:
    assert list(iter_batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(iter_batches([], 3)) == []


def test_insert_in_batches() -> None:
    client = _FakeClient()
    rows = [{"id": index} for index in range(5)]
    assert insert_in_batches(client, "c", rows, size=2) == 5
    assert [len(batch) for batch in client.inserted] == [2, 2, 1]


def test_insert_in_batches_without_count() -> None:
    class _NoCount(_FakeClient):
        def insert(self, collection_name, data):
            self.inserted.append(data)
            return {}

    client = _NoCount()
    assert insert_in_batches(client, "c", [{"id": 1}], size=10) == 1


def test_has_method_and_signature() -> None:
    assert has_method(_FakeClient(), "insert") is True
    assert has_method(_FakeClient(), "missing") is False
    signature = method_signature(_FakeClient, "insert")
    assert "collection_name" in signature
    assert "不可用" in method_signature(_FakeClient, "missing")


def test_drop_and_list_helpers_are_lenient() -> None:
    client = _FakeClient(names=["a", "b"], fail_drop=True)
    drop_collection_quietly(client, "a")  # 不抛异常
    assert list_collections(client) == ["a", "b"]

    broken = _FakeClient()
    broken.list_collections = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    assert list_collections(broken) == []


def test_demo_session_local_lifecycle(monkeypatch, unique_stem) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(milvus_utils, "import_client_class", lambda: lambda uri: fake)
    monkeypatch.setattr(config, "MILVUS_URI", "")

    with DemoSession(unique_stem) as session:
        assert session.local is True
        assert session.uri.endswith(f"runs/{unique_stem}/milvus.db")
        assert session.name("docs") == "docs"
        assert session.target.run_dir.exists()
    assert fake.closed is True


def test_demo_session_remote_cleans_prefix(monkeypatch, unique_stem) -> None:
    fake = _FakeClient(names=[f"learn_{unique_stem}_docs", "other"])
    monkeypatch.setattr(milvus_utils, "import_client_class", lambda: lambda uri: fake)
    monkeypatch.setattr(config, "MILVUS_URI", "http://127.0.0.1:19530")

    with DemoSession(unique_stem) as session:
        assert session.local is False
        assert session.name("docs") == f"learn_{unique_stem}_docs"
    assert fake.dropped == [f"learn_{unique_stem}_docs"]
    assert fake.closed is True


def test_demo_session_raises_when_dir_locked(monkeypatch, unique_stem, tmp_path) -> None:
    run_dir = tmp_path / "runs" / unique_stem
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "MILVUS_URI", "")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        milvus_utils.shutil,
        "rmtree",
        lambda *a, **k: (_ for _ in ()).throw(OSError("locked")),
    )
    with pytest.raises(RuntimeError, match="无法清理"):
        DemoSession(unique_stem).__enter__()
