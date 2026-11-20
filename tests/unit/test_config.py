"""config 模块：默认值、环境变量解析、路径回退与 DemoTarget。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlcore import config

pytestmark = pytest.mark.unit


def test_conftest_isolated_data_dir() -> None:
    assert config.DATA_DIR.name.startswith("milvus_learn_tests_")
    assert config.MILVUS_URI == ""
    assert config.DATA_DIR_FALLBACK is False


def test_defaults() -> None:
    assert config.PSEUDO_EMBED_DIM == 256
    assert config.EMBEDDING_PROVIDER == "pseudo"
    assert config.BM25_MODE == "auto"
    assert config.RAG_LLM == "offline"
    assert config.UI_PORT == 7860
    assert config.UI_RESET == 0


def test_reload_reads_new_values(config_env) -> None:
    values = config_env(PSEUDO_EMBED_DIM="64", UI_PORT="8123")
    assert values["PSEUDO_EMBED_DIM"] == 64
    assert config.PSEUDO_EMBED_DIM == 64
    assert config.UI_PORT == 8123


def test_int_clamping(config_env, capsys) -> None:
    values = config_env(PSEUDO_EMBED_DIM="999999", UI_PORT="80")
    assert values["PSEUDO_EMBED_DIM"] == 4096
    assert values["UI_PORT"] == 1024

    config_env(PSEUDO_EMBED_DIM="abc")
    assert config.PSEUDO_EMBED_DIM == 256
    assert "不是整数" in capsys.readouterr().out


def test_case_normalisation(config_env) -> None:
    values = config_env(EMBEDDING_PROVIDER="OPENAI", BM25_MODE="MANUAL")
    assert values["EMBEDDING_PROVIDER"] == "openai"
    assert values["BM25_MODE"] == "manual"

    values = config_env(EMBEDDING_PROVIDER="  ")
    assert values["EMBEDDING_PROVIDER"] == "pseudo"


def test_data_dir_env_relative_to_root(config_env) -> None:
    values = config_env(MILVUS_DB_DIR="tmp_db")
    assert values["DATA_DIR"] == (config.ROOT / "tmp_db").resolve()
    assert values["DATA_DIR_FALLBACK"] is False


def test_non_ascii_root_falls_back_to_home(monkeypatch, config_env) -> None:
    monkeypatch.setattr(config, "ROOT", Path("E:/学习笔记/fake") / "repo")
    values = config_env(MILVUS_DB_DIR="")
    assert values["DATA_DIR"] == (Path.home() / ".milvus_learn_data").resolve()
    assert values["DATA_DIR_FALLBACK"] is True


def test_resolve_demo_target_local() -> None:
    target = config.resolve_demo_target("demo_x")
    assert target.local is True
    assert target.collection_prefix == ""
    assert target.run_dir == config.DATA_DIR / "runs" / "demo_x"
    assert target.uri.endswith("runs/demo_x/milvus.db")
    assert target.name("docs") == "docs"


def test_resolve_demo_target_remote(config_env) -> None:
    config_env(MILVUS_URI="http://127.0.0.1:19530")
    target = config.resolve_demo_target("demo_x")
    assert target.local is False
    assert target.uri == "http://127.0.0.1:19530"
    assert target.collection_prefix == "learn_demo_x_"
    assert target.run_dir is None
    assert target.name("docs") == "learn_demo_x_docs"


def test_describe_contains_key_sections() -> None:
    text = config.describe()
    assert "目标:" in text
    assert "伪向量维度:" in text
    assert "BM25 模式:" in text


def test_settings_dicts() -> None:
    embedding = config.embedding_settings()
    assert embedding["provider"] == config.EMBEDDING_PROVIDER
    assert embedding["pseudo_dim"] == config.PSEUDO_EMBED_DIM
    assert isinstance(embedding["dim"], int)

    llm = config.llm_settings()
    assert llm["mode"] == config.RAG_LLM
    assert isinstance(llm["timeout"], int)

    ui = config.ui_settings()
    assert ui == {"port": config.UI_PORT, "reset": bool(config.UI_RESET)}
