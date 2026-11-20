"""集中读取环境变量与路径配置。

约定：
- 本地模式下，每个 demo 使用独立的 data/runs/<stem>/milvus.db，
  避免 Milvus Lite “一个 data_dir 一个进程”的文件锁冲突。
- 设置 MILVUS_URI 后，所有 demo 连接同一个远程服务，并用
  collection 前缀隔离。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv 随 pymilvus 一起安装，缺失时也能运行
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = _env(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        print(f"[配置警告] {name}={raw!r} 不是整数，使用默认值 {default}")
        value = default
    return max(minimum, min(maximum, value))


def _resolve_dir(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _contains_non_ascii(path: Path) -> bool:
    return any(ord(char) > 127 for char in str(path))


def _compute_data_dir() -> tuple[Path, bool]:
    """返回 (数据目录, 是否因非 ASCII 路径自动回退)。

    1. 显式设置 MILVUS_DB_DIR 时按配置走；
    2. 仓库路径包含中文等非 ASCII 字符时回退到用户主目录，
       因为 FAISS 的 C++ 文件写入在 Windows 上无法打开非 ASCII 路径；
    3. 否则使用仓库下的 data/。
    """
    data_dir_env = _env("MILVUS_DB_DIR")
    if data_dir_env:
        return _resolve_dir(data_dir_env), False
    if _contains_non_ascii(ROOT):
        return (Path.home() / ".milvus_learn_data").resolve(), True
    return (ROOT / "data").resolve(), False


def _compute_settings() -> dict:
    data_dir, fallback = _compute_data_dir()
    return {
        "DATA_DIR": data_dir,
        "DATA_DIR_FALLBACK": fallback,
        "MILVUS_URI": _env("MILVUS_URI"),
        "PSEUDO_EMBED_DIM": _env_int("PSEUDO_EMBED_DIM", 256, 16, 4096),
        "EMBEDDING_PROVIDER": _env("EMBEDDING_PROVIDER", "pseudo").lower() or "pseudo",
        "BM25_MODE": _env("BM25_MODE", "auto").lower() or "auto",
        "OPENAI_BASE_URL": _env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "OPENAI_API_KEY": _env("OPENAI_API_KEY"),
        "OPENAI_EMBED_MODEL": _env("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        "OPENAI_EMBED_DIM": _env_int("OPENAI_EMBED_DIM", 1536, 1, 8192),
        "OPENAI_CHAT_MODEL": _env("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "RAG_LLM": _env("RAG_LLM", "offline").lower() or "offline",
        "LLM_TIMEOUT": _env_int("LLM_TIMEOUT", 30, 1, 600),
        "UI_PORT": _env_int("UI_PORT", 7860, 1024, 65535),
        "UI_RESET": _env_int("UI_RESET", 0, 0, 1),
    }


# 模块级配置（由 reload_settings() 填充）。
DATA_DIR: Path
DATA_DIR_FALLBACK: bool
MILVUS_URI: str
PSEUDO_EMBED_DIM: int
EMBEDDING_PROVIDER: str
BM25_MODE: str
OPENAI_BASE_URL: str
OPENAI_API_KEY: str
OPENAI_EMBED_MODEL: str
OPENAI_EMBED_DIM: int
OPENAI_CHAT_MODEL: str
RAG_LLM: str
LLM_TIMEOUT: int
UI_PORT: int
UI_RESET: int


def reload_settings() -> dict:
    """重新读取环境变量并刷新模块级配置（测试用）。"""
    global DATA_DIR, DATA_DIR_FALLBACK, MILVUS_URI
    global PSEUDO_EMBED_DIM, EMBEDDING_PROVIDER, BM25_MODE
    global OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_EMBED_MODEL, OPENAI_EMBED_DIM
    global OPENAI_CHAT_MODEL, RAG_LLM, LLM_TIMEOUT, UI_PORT, UI_RESET

    values = _compute_settings()
    DATA_DIR = values["DATA_DIR"]
    DATA_DIR_FALLBACK = values["DATA_DIR_FALLBACK"]
    MILVUS_URI = values["MILVUS_URI"]
    PSEUDO_EMBED_DIM = values["PSEUDO_EMBED_DIM"]
    EMBEDDING_PROVIDER = values["EMBEDDING_PROVIDER"]
    BM25_MODE = values["BM25_MODE"]
    OPENAI_BASE_URL = values["OPENAI_BASE_URL"]
    OPENAI_API_KEY = values["OPENAI_API_KEY"]
    OPENAI_EMBED_MODEL = values["OPENAI_EMBED_MODEL"]
    OPENAI_EMBED_DIM = values["OPENAI_EMBED_DIM"]
    OPENAI_CHAT_MODEL = values["OPENAI_CHAT_MODEL"]
    RAG_LLM = values["RAG_LLM"]
    LLM_TIMEOUT = values["LLM_TIMEOUT"]
    UI_PORT = values["UI_PORT"]
    UI_RESET = values["UI_RESET"]
    return values


# 允许通过 .env 覆盖；override=False 保证真实环境变量优先。
if load_dotenv is not None:
    load_dotenv(ROOT / ".env", override=False)

reload_settings()


@dataclass(frozen=True)
class DemoTarget:
    """一个 demo 的运行目标（本地文件或远程服务）。"""

    stem: str
    uri: str
    local: bool
    collection_prefix: str
    run_dir: Path | None

    def name(self, suffix: str) -> str:
        """本地模式直接用 suffix；远程模式加前缀，便于清理。"""
        return f"{self.collection_prefix}{suffix}" if self.collection_prefix else suffix


def resolve_demo_target(stem: str) -> DemoTarget:
    if MILVUS_URI:
        return DemoTarget(
            stem=stem,
            uri=MILVUS_URI,
            local=False,
            collection_prefix=f"learn_{stem}_",
            run_dir=None,
        )
    run_dir = DATA_DIR / "runs" / stem
    return DemoTarget(
        stem=stem,
        uri=(run_dir / "milvus.db").as_posix(),
        local=True,
        collection_prefix="",
        run_dir=run_dir,
    )


def embedding_settings() -> dict:
    return {
        "provider": EMBEDDING_PROVIDER,
        "pseudo_dim": PSEUDO_EMBED_DIM,
        "base_url": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_EMBED_MODEL,
        "dim": OPENAI_EMBED_DIM,
    }


def llm_settings() -> dict:
    return {
        "mode": RAG_LLM,
        "base_url": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_CHAT_MODEL,
        "timeout": LLM_TIMEOUT,
    }


def ui_settings() -> dict:
    return {"port": UI_PORT, "reset": bool(UI_RESET)}


def describe() -> str:
    mode = f"远程 {MILVUS_URI}" if MILVUS_URI else f"本地 {DATA_DIR / 'runs'}"
    lines = [
        f"目标: {mode}",
        f"伪向量维度: {PSEUDO_EMBED_DIM}",
        f"Embedding: {EMBEDDING_PROVIDER}",
        f"BM25 模式: {BM25_MODE}",
        f"RAG 生成: {RAG_LLM}",
    ]
    if DATA_DIR_FALLBACK:
        lines.append(
            "数据目录说明: 仓库路径含非 ASCII 字符，本地数据已自动放到 "
            f"{DATA_DIR}（FAISS 无法写中文路径）；可用 MILVUS_DB_DIR 覆盖。"
        )
    return "\n".join(lines)
