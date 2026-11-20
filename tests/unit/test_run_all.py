"""scripts/run_all.py：命令构造与失败处理。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_all as run_all

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_demo_list_matches_disk() -> None:
    files = {path.name for path in (ROOT / "demos").glob("*.py")}
    assert {name for name, _ in run_all.DEMOS} == files
    assert len(run_all.DEMOS) == 11


def test_demo_command_deep_flags() -> None:
    with_deep = run_all.demo_command("00_environment_check.py", ["--deep"])
    assert "--deep" in with_deep
    without_deep = run_all.demo_command("00_environment_check.py", ["--deep"], no_deep=True)
    assert "--deep" not in without_deep
    other = run_all.demo_command("01_hello_milvus.py", [], no_deep=True)
    assert other[-1].endswith("01_hello_milvus.py")


def test_run_all_success(monkeypatch) -> None:
    monkeypatch.setattr(run_all.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert run_all.run() == []


def test_run_all_failure_stops(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    failures = run_all.run()
    assert failures == [run_all.DEMOS[0][0]]
    assert len(calls) == 1


def test_run_all_continue_on_failure(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    failures = run_all.run(stop_on_failure=False)
    assert len(failures) == len(run_all.DEMOS)
    assert len(calls) == len(run_all.DEMOS)


def test_run_all_timeout(monkeypatch, capsys) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    failures = run_all.run(timeout=1)
    assert failures == [run_all.DEMOS[0][0]]
    assert "[超时]" in capsys.readouterr().out


def test_main_exit_code(monkeypatch) -> None:
    monkeypatch.setattr(run_all, "run", lambda **kwargs: [])
    monkeypatch.setattr("sys.argv", ["run_all.py", "--no-deep"])
    assert run_all.main() == 0

    monkeypatch.setattr(run_all, "run", lambda **kwargs: ["x.py"])
    assert run_all.main() == 1
