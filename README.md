# Milvus 学习项目

一套可以**完全离线运行**的 Milvus 中文学习材料：系统笔记 + 分步 demo + Gradio 交互检索 + 迷你 RAG。

- 运行环境：Windows + Python 3.10+，使用 **Milvus Lite**（本地 `.db` 文件），**不需要 Docker**。
- 文本向量化：内置**离线确定性伪向量**，不需要下载模型、不需要 API Key。
- 所有 demo 都使用同一份 25 条中文样例语料，便于对照理解。
- 真实 embedding / 大模型接口是**可选项**，通过 `.env` 显式开启；默认不联网。

> 伪向量不是真正的语义模型。它用“分词 + 特征哈希 + 少量概念词表”模拟向量，
> 目的是让你专注理解 Milvus 的数据模型、索引、过滤与检索流程。
> 生产环境请替换成 BGE、OpenAI Embedding 等真实模型，接入点见 `mlcore/embedding.py`。

---

## 快速开始

在仓库根目录打开终端（Git Bash / PowerShell 均可）：

```bash
# 1. 安装依赖（Windows 上必须显式安装 milvus-lite）
venv/Scripts/python.exe -m pip install -U pip
venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. 环境自检（--deep 会实测 Milvus Lite 的检索与 BM25 能力）
venv/Scripts/python.exe demos/00_environment_check.py --deep

# 3. 第一个 demo
venv/Scripts/python.exe demos/01_hello_milvus.py

# 4. 交互式 Web UI（浏览器自动打开 http://127.0.0.1:7860）
venv/Scripts/python.exe app/gradio_search.py

# 5. 安装开发依赖并跑测试
venv/Scripts/python.exe -m pip install -r requirements-dev.txt
venv/Scripts/python.exe scripts/dev.py test
```

PyCharm：把项目解释器指向 `.\venv\Scripts\python.exe`，直接右键运行 `demos\` 下的脚本即可。

如果 Python 3.14 安装依赖失败，可用本机 Python 3.12 新建备用环境：

```bash
py -3.12 -m venv .venv312
.venv312/Scripts/python.exe -m pip install -r requirements.txt
```

---

## 学习路线

建议按顺序阅读笔记并运行对应 demo，每节约 20–40 分钟。

| 阶段 | 笔记 | Demo | 你会学到 |
|---|---|---|---|
| 0 | [00 学习路线与环境准备](docs/00_学习路线与环境准备.md) | `00_environment_check.py` | 安装、配置、数据目录隔离 |
| 1 | [01 向量检索与 Embedding 基础](docs/01_向量检索与Embedding基础.md) | `02_embeddings_and_similarity.py` | 向量、度量、ANN 直觉、伪向量 |
| 1 | [02 Milvus 架构与部署形态](docs/02_Milvus架构与部署形态.md) | — | 分层架构、写入/查询路径、四种形态 |
| 2 | [03 数据模型与 Schema 设计](docs/03_数据模型与Schema设计.md) | `01_hello_milvus.py`、`03_crud.py` | Collection/Schema/CRUD |
| 3 | [04 索引类型与参数调优](docs/04_索引类型与参数调优.md) | `04_schema_and_index.py` | FLAT/HNSW/IVF 与参数 |
| 3 | [05 相似度度量与搜索参数](docs/05_相似度度量与搜索参数.md) | `01`、`09_search_advanced.py` | COSINE/L2/IP、分页、范围搜索 |
| 3 | [06 标量过滤与查询表达式](docs/06_标量过滤与查询表达式.md) | `05_scalar_filter.py` | 布尔表达式、JSON、数组 |
| 4 | [07 分区、分区键与别名](docs/07_分区_分区键与别名.md) | `06_partitions_aliases.py` | 分区裁剪、别名切换 |
| 4 | [08 稀疏向量与 BM25 全文检索](docs/08_稀疏向量与BM25全文检索.md) | `07_sparse_bm25.py` | 稀疏向量、BM25、分词 |
| 5 | [09 混合检索与重排序](docs/09_混合检索与重排序.md) | `08_hybrid_search.py` | 多路召回、RRF、加权融合 |
| 5 | [10 RAG 检索增强生成实战](docs/10_RAG检索增强生成实战.md) | `10_mini_rag.py`、Web UI | 切块、检索、Prompt、离线回答 |
| 6 | [11 最佳实践与性能调优](docs/11_最佳实践与性能调优.md) | — | 批量写入、参数选择、容量估算 |
| 6 | [12 Lite / Standalone / Distributed / Zilliz Cloud](docs/12_Lite_Standalone_Distributed与ZillizCloud.md) | — | 部署选型与迁移 |
| 7 | [13 测试与 DevOps 实践](docs/13_测试与DevOps实践.md) | `scripts/dev.py` | 单元/集成/E2E、覆盖率、lint、pre-commit |

`scripts/run_all.py` 可以按顺序跑完所有 demo，用于冒烟检查。

---

## 测试与开发

```bash
venv/Scripts/python.exe -m pip install -r requirements-dev.txt

venv/Scripts/python.exe scripts/dev.py test-unit     # 约 8 秒
venv/Scripts/python.exe scripts/dev.py test          # 除 slow 外全部（约 4 分钟）
venv/Scripts/python.exe scripts/dev.py coverage      # 覆盖率报告（阈值 80%）
venv/Scripts/python.exe scripts/dev.py lint          # ruff
venv/Scripts/python.exe scripts/dev.py smoke         # 顺序跑 11 个 demo
venv/Scripts/python.exe scripts/dev.py ci            # lint + format check + coverage + smoke
```

测试分三层：`unit`（无 Milvus）、`integration`（Milvus Lite）、`e2e`（demo 子进程 + Gradio HTTP）。
当前 176 个用例、共享库与 UI 覆盖率约 93%。没有云 CI 配置，`dev.py ci` 就是本地流水线，
可直接接入任意 CI 系统。详见 [docs/13](docs/13_测试与DevOps实践.md)。

---

## 目录结构

```text
milvus_learn/
├─ README.md
├─ requirements.txt / requirements-dev.txt
├─ pyproject.toml            # pytest / ruff / coverage 配置
├─ Makefile                  # 可选入口，转发到 scripts/dev.py
├─ .pre-commit-config.yaml
├─ .env.example              # 复制为 .env 后修改；不复制也能运行
├─ docs/                     # 14 篇中文笔记
├─ mlcore/                   # 配置、分词、伪向量、语料、BM25、service、RAG
├─ demos/                    # 00–10 分步脚本
├─ app/gradio_search.py      # 向量检索 + Mini RAG 两个标签页（薄 UI 层）
├─ tests/                    # unit / integration / e2e
├─ data/sample_docs.json     # 25 条中文样例语料
├─ data/runs/                # 运行时数据库（已忽略；中文路径下自动放到 ~/.milvus_learn_data）
└─ scripts/                  # run_all.py 冒烟、dev.py 开发入口
```

每个 demo 使用独立的 `<MILVUS_DB_DIR>/runs/<demo>/milvus.db`，因此可以反复运行、互不干扰，
也避免 Milvus Lite “一个数据目录只能有一个进程”的文件锁问题。

---

## 配置（可选）

复制 `.env.example` 为 `.env`。常用项：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MILVUS_URI` | 空 | 空 = 本地 Milvus Lite；填 `http://host:19530` = 连接远程 Milvus |
| `MILVUS_DB_DIR` | 空 | 本地数据根目录；留空且仓库路径含中文时自动用 `~/.milvus_learn_data` |
| `PSEUDO_EMBED_DIM` | `256` | 伪向量维度 |
| `EMBEDDING_PROVIDER` | `pseudo` | `pseudo` 或 `openai` |
| `BM25_MODE` | `auto` | `auto` / `native` / `manual` |
| `OPENAI_BASE_URL` | OpenAI | 兼容接口地址（也可填 DashScope 等兼容服务） |
| `OPENAI_API_KEY` | 空 | 只有显式开启真实 embedding / LLM 时才需要 |
| `RAG_LLM` | `offline` | `offline` 抽取式回答；`openai` 调用兼容接口 |
| `UI_PORT` | `7860` | Gradio 端口 |

> 隐私提示：默认完全离线。只有当你设置 `EMBEDDING_PROVIDER=openai` 或
> `RAG_LLM=openai` 并填写 Key 时，文本才会发送到你配置的外部服务。

---

## 常见问题

**1. 提示数据目录被锁定 / 无法删除 `data/runs/...`**
Milvus Lite 按数据目录加文件锁。关闭正在运行的 demo / UI / 残留 Python 进程后重试；
不同 demo 之间不要共用同一个 `.db` 文件。

**2. 仓库路径含中文，Milvus Lite 打开失败**
FAISS 在 Windows 上无法写入非 ASCII 路径，项目会自动把本地数据放到
`C:\Users\<用户名>\.milvus_learn_data`。如需指定位置，在 `.env` 设置
`MILVUS_DB_DIR=C:/Users/<你的用户名>/milvus_data`。

**3. 安装依赖时缺少 wheel**
先升级 pip，再使用 `--only-binary=:all:` 快速确认；若 Python 3.14 不兼容，
按上文用 Python 3.12 创建 `.venv312`。

**4. 控制台中文乱码**
`mlcore/__init__.py` 会把 stdout/stderr 设为 UTF-8；如仍有问题，在 PyCharm 的
Run Configuration 中设置 `PYTHONIOENCODING=utf-8`。

**5. 伪向量检索效果“不够聪明”**
这是预期行为：它只共享字词特征和少量概念词，不具备真实模型的同义泛化能力。
换成真实 embedding 后，效果会显著提升。

**6. Milvus Lite 和正式 Milvus 有什么区别？**
Lite 适合学习和单机原型：单进程、无鉴权、无 PQ/多种量化向量类型，部分 gRPC 能力受限。
迁移到 Standalone/Distributed/Zilliz Cloud 时，主要修改连接 URI 与少量参数，
核心 `MilvusClient` 代码基本不变。详见 `docs/12`。
