# 可选入口：所有目标都转发到 scripts/dev.py（Windows 上 make 可能未安装，dev.py 为准）。
PY ?= venv/Scripts/python.exe

.PHONY: install install-dev lint format test-unit test test-all coverage smoke clean ci precommit

install:
	$(PY) scripts/dev.py install

install-dev:
	$(PY) scripts/dev.py install-dev

lint:
	$(PY) scripts/dev.py lint

format:
	$(PY) scripts/dev.py format

test-unit:
	$(PY) scripts/dev.py test-unit

test:
	$(PY) scripts/dev.py test

test-all:
	$(PY) scripts/dev.py test-all

coverage:
	$(PY) scripts/dev.py coverage

smoke:
	$(PY) scripts/dev.py smoke

clean:
	$(PY) scripts/dev.py clean

ci:
	$(PY) scripts/dev.py ci

precommit:
	$(PY) scripts/dev.py precommit
