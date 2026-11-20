"""测试共享工具：路径、子进程环境与输出清理。"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def e2e_env(data_dir: Path) -> dict[str, str]:
    """为子进程准备隔离的离线环境。"""
    env = dict(os.environ)
    env.update(
        {
            "MILVUS_DB_DIR": str(data_dir),
            "MILVUS_URI": "",
            "EMBEDDING_PROVIDER": "pseudo",
            "BM25_MODE": "auto",
            "RAG_LLM": "offline",
            "OPENAI_API_KEY": "",
            "UI_RESET": "0",
            "GRADIO_ANALYTICS_ENABLED": "False",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONPATH": str(ROOT),
        }
    )
    return env


def clean(text: str) -> str:
    return ANSI.sub("", text or "")
