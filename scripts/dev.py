"""本地开发/CI 统一入口（不依赖 Make、不依赖任何云 CI）。

用法：
    venv/Scripts/python.exe scripts/dev.py <命令>

命令：
    install       安装运行依赖
    install-dev   安装运行 + 开发依赖，并安装 pre-commit 钩子
    lint          ruff check
    format        ruff format + ruff check --fix
    test-unit     只跑单元测试
    test          跑除 slow 外的全部测试
    test-all      跑全部测试（含 slow）
    coverage      测试 + 覆盖率（阈值 80%）
    smoke         顺序运行 11 个 demo
    clean         清理缓存与覆盖率产物（--data 同时清理本地数据库）
    ci            lint -> format check -> coverage -> smoke
    precommit     pre-commit run --all-files
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PYTHON = sys.executable

CLEAN_TARGETS = [
    ".pytest_cache",
    ".ruff_cache",
    ".coverage",
    "coverage.xml",
    "coverage.json",
    "htmlcov",
]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    return env


def _run(args: list[str], *, check: bool = True) -> int:
    print(f"\n$ {' '.join(args)}", flush=True)
    result = subprocess.run(args, cwd=ROOT, env=_env())
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def _module(*args: str) -> int:
    return _run([PYTHON, *args])


def install() -> None:
    _module("-m", "pip", "install", "-U", "pip")
    _module("-m", "pip", "install", "-r", "requirements.txt")


def install_dev() -> None:
    _module("-m", "pip", "install", "-U", "pip")
    _module("-m", "pip", "install", "-r", "requirements-dev.txt")
    _module("-m", "pre_commit", "install")


def lint() -> None:
    _module("-m", "ruff", "check", ".")


def format_code() -> None:
    _module("-m", "ruff", "format", ".")
    _module("-m", "ruff", "check", "--fix", ".")


def test_unit(extra: list[str] | None = None) -> None:
    _module("-m", "pytest", "-m", "unit", "-q", *(extra or []))


def test(extra: list[str] | None = None) -> None:
    _module("-m", "pytest", "-q", *(extra or []))


def test_all(extra: list[str] | None = None) -> None:
    _module("-m", "pytest", "-m", "unit or integration or e2e", "-q", *(extra or []))


def coverage(extra: list[str] | None = None) -> None:
    _module(
        "-m",
        "pytest",
        "--cov=mlcore",
        "--cov=app",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "-q",
        *(extra or []),
    )


def smoke() -> None:
    _module("scripts/run_all.py", "--no-deep")


def clean(include_data: bool = False) -> None:
    removed: list[str] = []
    for name in CLEAN_TARGETS:
        path = ROOT / name
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(name)
        elif path.exists():
            path.unlink(missing_ok=True)
            removed.append(name)
    for path in ROOT.rglob("__pycache__"):
        if "venv" in path.parts or ".venv" in path.parts[0]:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(str(path.relative_to(ROOT)))

    if include_data:
        sys.path.insert(0, str(ROOT))
        from mlcore import config

        runs = config.DATA_DIR / "runs"
        if runs.exists():
            shutil.rmtree(runs, ignore_errors=True)
            removed.append(str(runs))

    if removed:
        print("已清理：")
        for name in removed:
            print(f"  - {name}")
    else:
        print("没有需要清理的产物。")


def precommit() -> None:
    _module("-m", "pre_commit", "run", "--all-files")


def ci() -> None:
    """本地 CI 流水线：任一阶段失败即退出。"""
    steps = [
        ("lint", lambda: _module("-m", "ruff", "check", ".")),
        ("format check", lambda: _module("-m", "ruff", "format", "--check", ".")),
        ("coverage", coverage),
        ("smoke", smoke),
    ]
    for label, step in steps:
        print(f"\n===== CI: {label} =====")
        step()
    print("\n===== CI 全部通过 =====")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus 学习项目开发脚本")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "install",
        "install-dev",
        "lint",
        "format",
        "test-unit",
        "test",
        "test-all",
        "coverage",
        "smoke",
        "ci",
        "precommit",
    ):
        sub.add_parser(name)
    clean_parser = sub.add_parser("clean")
    clean_parser.add_argument("--data", action="store_true", help="同时删除本地 Milvus 运行数据")

    # 用 parse_known_args 透传额外参数，例如 `dev.py test -k config`
    args, extra = parser.parse_known_args(argv)
    extra = [item for item in extra if item != "--"]

    commands = {
        "install": install,
        "install-dev": install_dev,
        "lint": lint,
        "format": format_code,
        "test-unit": lambda: test_unit(extra),
        "test": lambda: test(extra),
        "test-all": lambda: test_all(extra),
        "coverage": lambda: coverage(extra),
        "smoke": smoke,
        "ci": ci,
        "precommit": precommit,
        "clean": lambda: clean(args.data),
    }
    try:
        commands[args.command]()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
