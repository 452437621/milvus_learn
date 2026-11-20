"""按顺序运行全部 demo，用于冒烟检查。

用法：
    venv/Scripts/python.exe scripts/run_all.py
    venv/Scripts/python.exe scripts/run_all.py --no-deep
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DEMOS: list[tuple[str, list[str]]] = [
    ("00_environment_check.py", ["--deep"]),
    ("01_hello_milvus.py", []),
    ("02_embeddings_and_similarity.py", []),
    ("03_crud.py", []),
    ("04_schema_and_index.py", []),
    ("05_scalar_filter.py", []),
    ("06_partitions_aliases.py", []),
    ("07_sparse_bm25.py", []),
    ("08_hybrid_search.py", []),
    ("09_search_advanced.py", []),
    ("10_mini_rag.py", ["--question", "Milvus Lite 有哪些限制？"]),
]


def demo_command(
    filename: str,
    extra_args: list[str] | None = None,
    *,
    no_deep: bool = False,
) -> list[str]:
    """构造运行某个 demo 的命令行。"""
    args = list(extra_args or [])
    if no_deep and filename.startswith("00"):
        args = [arg for arg in args if arg != "--deep"]
    return [sys.executable, str(ROOT / "demos" / filename), *args]


def run(
    *,
    no_deep: bool = False,
    stop_on_failure: bool = True,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> list[str]:
    """顺序运行 demo，返回失败的文件名列表。"""
    run_env = dict(os.environ)
    run_env.setdefault("PYTHONIOENCODING", "utf-8")
    run_env.setdefault("PYTHONUTF8", "1")
    if env:
        run_env.update(env)

    failures: list[str] = []
    for filename, extra_args in DEMOS:
        command = demo_command(filename, extra_args, no_deep=no_deep)
        print("\n" + "=" * 78)
        print(f"运行 {filename}")
        print("=" * 78)
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=run_env,
                timeout=timeout,
                capture_output=capture_output,
                text=capture_output,
                encoding="utf-8" if capture_output else None,
                errors="replace" if capture_output else None,
            )
        except subprocess.TimeoutExpired:
            failures.append(filename)
            print(f"[超时] {filename}（超过 {timeout}s）")
            if stop_on_failure:
                break
            continue

        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            failures.append(filename)
            print(f"[失败] {filename}（退出码 {result.returncode}，{elapsed:.1f}s）")
            if stop_on_failure:
                break
        else:
            print(f"[通过] {filename}（{elapsed:.1f}s）")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="顺序运行 Milvus 学习 demo")
    parser.add_argument("--no-deep", action="store_true", help="00 不执行深度探测")
    args = parser.parse_args()

    started = time.perf_counter()
    failures = run(no_deep=args.no_deep)

    print("\n" + "=" * 78)
    if failures:
        print(f"冒烟检查失败：{', '.join(failures)}")
        return 1
    print(f"全部 {len(DEMOS)} 个 demo 通过（总耗时 {time.perf_counter() - started:.1f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
