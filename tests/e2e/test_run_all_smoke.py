"""端到端（slow）：完整重放 scripts/run_all.py。"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.support import ROOT, clean, e2e_env

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def test_run_all_script(e2e_root) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_all.py"), "--no-deep"],
        cwd=ROOT,
        env=e2e_env(e2e_root / "run_all"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    output = clean(result.stdout + result.stderr)
    assert result.returncode == 0, output[-3000:]
    assert "全部 11 个 demo 通过" in output
    assert "[失败]" not in output
