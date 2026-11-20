# 13 测试与 DevOps 实践

## 1. 为什么一个学习项目也要写测试

前 12 章证明了“Milvus 能跑通”，但那是人工验证：每次改动后要挨个运行 11 个 demo，靠肉眼判断输出。

测试把这种验证变成**可重复、可自动执行**的检查：

- 改共享库时，立刻知道哪个 demo 的行为变了；
- 升级 `pymilvus` / `milvus-lite` 时，能快速发现 API 差异；
- 重构界面（例如把业务逻辑从 Gradio 拆出来）时，有回归护栏；
- 别人（或未来的你）接手时，`scripts/dev.py test` 一条命令就能确认环境正常。

## 2. 测试分层

| 层 | 标记 | 内容 | 是否启动 Milvus | 耗时 |
|---|---|---|---|---|
| 单元 | `unit` | 分词、伪向量、切块、Prompt、配置、结果格式化、RRF、run_all 逻辑 | 否 | 约 8 秒 |
| 集成 | `integration` | CRUD、8 种过滤、分区/别名、BM25、混合检索、高级搜索、RAG、SearchService | 是（Milvus Lite） | 1–2 分钟 |
| 端到端 | `e2e` | 11 个 demo 子进程、Gradio 启动 + HTTP 200 + 回调 | 是（子进程） | 约 1 分钟 |
| 慢速 | `slow` | 完整重放 `scripts/run_all.py` | 是 | 1–2 分钟 |

默认命令排除 `slow`：

```bash
venv/Scripts/python.exe -m pytest -q          # unit + integration + e2e
venv/Scripts/python.exe -m pytest -m unit -q  # 只跑单元
venv/Scripts/python.exe -m pytest -m slow -q  # 显式跑慢速
```

## 3. 环境隔离：最容易踩的坑

### 3.1 配置在导入时读取

`mlcore/config.py` 在 import 时读取 `.env` 和 `os.environ`。因此 `tests/conftest.py` 必须在**导入 mlcore 之前**设置环境变量：

```python
os.environ["MILVUS_DB_DIR"] = <临时目录>
os.environ["MILVUS_URI"] = ""        # 注意：是空字符串，不是删除
import mlcore  # 之后才导入
```

### 3.2 为什么用空字符串而不是 `del`

`config.py` 使用 `load_dotenv(ROOT / ".env", override=False)`：

- 变量**不存在**时，`.env` 里的值会被加载进来；
- 变量**已存在**（哪怕是空字符串），`override=False` 就不会覆盖。

如果测试里 `del os.environ["MILVUS_URI"]`，而开发者本地 `.env` 写了一台远程 Milvus，测试就会悄悄连到远程机器。设为 `""` 可以彻底锁死。

### 3.3 配置是全局状态

`reload_settings()` 会改写模块级变量。测试用 autouse fixture 在结束后恢复快照，并清理 `get_embedder()` 的缓存：

```python
def _restore_config(...):
    yield
    for key, value in _CONFIG_SNAPSHOT.items():
        setattr(config, key, value)
    reset_embedder()
```

因此**不要**在多线程/多进程中并行修改配置（本项目也不启用 pytest-xdist）。

## 4. Milvus Lite 的文件锁

Lite 对一个数据目录加进程级文件锁。测试策略：

1. 每个测试用 `unique_stem` 生成独立的 `runs/<stem>/milvus.db`；
2. client 在 `finally` 中 `close()`；
3. 删除目录带重试（Windows 上可能短暂占用 `.db` 文件）；
4. 不并行跑集成测试。

```python
def _rmtree_retry(path, attempts=3):
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
```

## 5. 单元测试示例

纯函数最容易测。比如伪向量的确定性：

```python
def test_deterministic_and_nfkc_equivalent():
    embedder = PseudoEmbedder(dim=128)
    assert embedder.embed_query("Milvus 向量检索") == embedder.embed_query("Ｍｉｌｖｕｓ 向量检索")
```

外部依赖用 `monkeypatch` 替换，不发真实请求：

```python
def test_chat_payload_and_strip(monkeypatch):
    _install_fake_urlopen(monkeypatch, calls, lambda n: _FakeResponse({...}))
    chat = OpenAICompatibleChat(base_url="http://fake/v1", api_key="k")
    assert chat.generate("sys", "user") == "回答"
```

语料校验测试还会故意构造坏 JSON，确认 `load_docs()` 给出明确报错。

## 6. 集成测试示例

过滤计数直接从语料统计得出（`data/sample_docs.json` 当前 25 条）：

```python
@pytest.mark.parametrize(
    "expression,expected",
    [
        ('category == "美食"', 4),
        ("year >= 2022", 17),
        ("rating > 4.5", 15),
    ],
)
def test_filter_counts(client, docs_collection, expression, expected):
    rows = client.query(collection_name=docs_collection, filter=expression, output_fields=["id"])
    assert len(rows) == expected
```

两个必须记住的 Milvus Lite 事实：

- `count(*)` 会**忽略** `partition_names`，按分区计数要取回 `id` 再数；
- BM25 原生模式返回**负分**，越负越相关（手动回退模式用 IP，正分）。

## 7. 端到端测试

### 7.1 demo 子进程

每个 demo 用独立数据目录运行，并检查关键输出：

```python
result = subprocess.run(run_all.demo_command(filename, extra_args), env=e2e_env(data_dir), ...)
assert result.returncode == 0
assert "Traceback" not in output
assert "迭代总数" in output
```

注意 Windows 默认控制台编码是 cp936，必须显式传 `encoding="utf-8", errors="replace"`，并设置 `PYTHONUTF8=1`。

### 7.2 Gradio

```python
app = build_app(service)
app.launch(server_name="127.0.0.1", server_port=<空闲端口>,
           share=False, inbrowser=False, prevent_thread_lock=True)
# 轮询 http://127.0.0.1:<port>/ 直到 HTTP 200
```

不在 UI 上模拟点击，而是直接调用 `make_callbacks(service)`，把 Gradio 的请求上下文排除在测试之外。

## 8. 覆盖率

```bash
venv/Scripts/python.exe scripts/dev.py coverage
```

- 统计范围：`mlcore` + `app`；
- 阈值 `fail_under = 80`；
- 报告：终端（含缺失行）、`htmlcov/index.html`、`coverage.xml`。

**盲区**：11 个 demo 在子进程中运行，不会被计入覆盖率。这是有意的——demo 用端到端断言保证行为，覆盖率衡量的是共享库与 UI 逻辑。当前约 93%。

## 9. 代码质量

### 9.1 ruff

```bash
venv/Scripts/python.exe -m ruff check .          # 静态检查
venv/Scripts/python.exe -m ruff format .         # 格式化
```

配置在 `pyproject.toml`：`select = ["E", "F", "W", "I", "UP", "B"]`。

几个约定：

- `demos/`、`app/`、`scripts/` 放行 `E402`（入口脚本需要在 `sys.path.insert` 之后导入）；
- `B905`（`zip` 缺 `strict`）暂不启用，避免改变行为；
- ruff 0.17 也会格式化 Markdown 代码块，所以文档里的 Python 片段也是统一风格。

### 9.2 pre-commit

```bash
venv/Scripts/python.exe scripts/dev.py install-dev   # 安装并在 git 中注册钩子
```

`.pre-commit-config.yaml` 包含：

- `ruff check --fix`；
- `ruff format`；
- 单元测试（约 8 秒），保证提交前快速反馈。

三个钩子都用 `language: system` 指向项目 venv，**首次运行不需要联网**。
Windows 上相对路径必须用反斜杠（正斜杠会让 `CreateProcess` 报 WinError 2），
而 YAML 里要写成双反斜杠，shlex 解析后才得到单个反斜杠：

```yaml
entry: 'venv\\Scripts\\python.exe -m ruff check --fix'
```

另外 pre-commit 只检查 **git 已跟踪** 的文件，新文件要先 `git add` 才会被钩子看到。

## 10. 命令速查

| 命令 | 作用 |
|---|---|
| `scripts/dev.py install` | 安装运行依赖 |
| `scripts/dev.py install-dev` | 安装开发依赖 + 注册 pre-commit |
| `scripts/dev.py lint` | ruff check |
| `scripts/dev.py format` | ruff format + 自动修复 |
| `scripts/dev.py test-unit` | 单元测试 |
| `scripts/dev.py test` | 除 slow 外全部测试 |
| `scripts/dev.py test-all` | 全部测试（含 slow） |
| `scripts/dev.py coverage` | 测试 + 覆盖率 + HTML/XML 报告 |
| `scripts/dev.py smoke` | 顺序运行 11 个 demo |
| `scripts/dev.py clean` | 清理缓存与覆盖率产物（`--data` 同时清数据库） |
| `scripts/dev.py ci` | lint → format check → coverage → smoke |
| `scripts/dev.py precommit` | 对全部文件跑 pre-commit |

额外参数会透传给底层命令，例如只跑配置相关测试：

```bash
venv/Scripts/python.exe scripts/dev.py test -k config
```

`Makefile` 只是把同样的命令包一层；Windows 上没有 `make` 时直接用 `dev.py`。

`dev.py ci` 就是本项目的“CI 流水线”：不依赖 GitHub Actions/GitLab CI，任何机器、任何 CI 系统都可以直接调用。

## 11. Windows 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 集成测试报数据库被锁 | 上一个进程未关闭 client | 确认 `finally` 中 close；重试删除目录 |
| 子进程输出乱码 | cp936 编码 | 子进程设置 `PYTHONUTF8=1`，capture 用 UTF-8 |
| Gradio 启动失败 | 端口被占用 | 测试使用随机空闲端口并重试 |
| 覆盖率低于 80 | 新增代码没有测试 | 补测试，不要下调阈值 |
| `ruff format` 改了文档代码块 | 0.17 支持 Markdown | 正常，提交即可 |
| pre-commit 报“Executable 未找到” | Windows 相对路径 + 单反斜杠被 shlex 吞掉 | entry 写成双反斜杠 |
| `pre-commit run --all-files` 显示 no files | 文件还没被 git 跟踪 | 先 `git add` |

## 12. 练习

1. 新增一个 `demos/11_xxx.py`，并完成三件事：加入 `run_all.DEMOS`、在 `test_demos.py` 的 `EXPECTED` 里加标记、更新 README 学习路线（一致性测试会校验）。
2. 写一个测试：把 `EMBEDDING_PROVIDER` 设为 `openai` 但不给 Key，断言回退到伪向量而不是抛异常。
3. 让 `SearchService.search` 支持按年份范围过滤，并补一条集成测试。
