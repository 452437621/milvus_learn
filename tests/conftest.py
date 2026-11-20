"""全局测试配置：环境隔离、配置快照、Milvus Lite 独立数据目录。

关键：所有环境变量必须在导入 mlcore 之前设置。
`MILVUS_URI` 等设为空字符串而不是删除，因为 config 用
`load_dotenv(..., override=False)`：变量“已存在”时 .env 不会覆盖它。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="milvus_learn_tests_")).resolve()

_ENV_DEFAULTS = {
    "MILVUS_DB_DIR": str(_TEST_ROOT),
    "MILVUS_URI": "",
    "EMBEDDING_PROVIDER": "pseudo",
    "BM25_MODE": "auto",
    "RAG_LLM": "offline",
    "OPENAI_API_KEY": "",
    "UI_RESET": "0",
    "UI_PORT": "7860",
    "PSEUDO_EMBED_DIM": "256",
    "GRADIO_ANALYTICS_ENABLED": "False",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}
os.environ.update(_ENV_DEFAULTS)

import pytest  # noqa: E402

from mlcore import config, corpus  # noqa: E402
from mlcore.embedding import PseudoEmbedder, reset_embedder  # noqa: E402

CONFIG_KEYS = [key for key in vars(config) if key.isupper()]
_CONFIG_SNAPSHOT = {key: getattr(config, key) for key in CONFIG_KEYS}
TEST_ROOT = _TEST_ROOT


def _rmtree_retry(path: Path, attempts: int = 3) -> None:
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == attempts - 1:
                shutil.rmtree(path, ignore_errors=True)
            else:
                time.sleep(0.1)


@pytest.fixture(scope="session", autouse=True)
def _session_data_dir():
    yield
    _rmtree_retry(_TEST_ROOT)


@pytest.fixture(autouse=True)
def _restore_config(_session_data_dir):
    """每个测试后恢复 config 全局与 embedder 缓存。"""
    yield
    for key, value in _CONFIG_SNAPSHOT.items():
        setattr(config, key, value)
    reset_embedder()


@pytest.fixture
def unique_stem() -> str:
    return f"t_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def docs() -> list[dict]:
    return corpus.load_docs()


@pytest.fixture
def embedder() -> PseudoEmbedder:
    return PseudoEmbedder()


@pytest.fixture
def client(unique_stem):
    from pymilvus import MilvusClient

    run_dir = _TEST_ROOT / "runs" / unique_stem
    run_dir.mkdir(parents=True, exist_ok=True)
    milvus_dir = run_dir / "milvus.db"
    instance = MilvusClient(uri=milvus_dir.as_posix())
    try:
        yield instance
    finally:
        instance.close()
        _rmtree_retry(run_dir)


@pytest.fixture
def demo_session(unique_stem):
    from mlcore.milvus_utils import DemoSession

    session = DemoSession(unique_stem)
    session.__enter__()
    run_dir = session.target.run_dir
    try:
        yield session
    finally:
        session.__exit__(None, None, None)
        if run_dir is not None:
            _rmtree_retry(run_dir)


@pytest.fixture
def config_env(monkeypatch):
    """monkeypatch 环境变量 + reload；teardown 先撤销 monkeypatch 再还原配置。"""

    def apply(**values: str) -> dict:
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        return config.reload_settings()

    yield apply
    monkeypatch.undo()
    config.reload_settings()
