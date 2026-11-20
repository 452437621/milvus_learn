"""Milvus 学习项目的共享库。

统一处理控制台编码，避免 Windows 下中文输出乱码。
"""

from __future__ import annotations

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # 被重定向或非标准流时忽略
        pass

__all__ = ["config", "text", "embedding", "sparse", "corpus", "pretty"]
