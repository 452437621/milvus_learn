# 02 Milvus 架构与部署形态

## 1. 为什么向量数据库不是“加个索引的 MySQL”

向量检索有三个和传统数据库很不一样的特点：

1. **数据量大**：一条 768 维 float32 向量约 3 KB，一千万条就是 30 GB 原始向量；
2. **索引构建重**：HNSW/IVF 的索引可能和原始数据一样大，构建需要大量 CPU/内存；
3. **查询是相似度排序**：不是精确等值匹配，而是返回 Top-K，允许近似。

因此 Milvus 把系统拆成“接入、协调、计算、存储”几层，让写入、建索引、查询可以独立扩展。

## 2. 云原生分层架构（Standalone / Distributed）

```text
             SDK / REST / gRPC
                    │
              ┌─────▼─────┐
              │   Proxy   │  接入层：鉴权、路由、结果归并
              └─────┬─────┘
        ┌───────────┼───────────┐
        │           │           │
   ┌────▼────┐ ┌────▼────┐ ┌────▼─────┐
   │ Root    │ │ Query   │ │ Data     │  协调层：元数据、调度、负载均衡
   │ Coord   │ │ Coord   │ │ Coord    │
   └────┬────┘ └────┬────┘ └────┬─────┘
        │           │           │
   ┌────▼────┐ ┌────▼────┐ ┌────▼─────┐
   │ Query   │ │ Index   │ │ Data     │  计算层：查询、建索引、写入
   │ Nodes   │ │ Nodes   │ │ Nodes    │
   └────┬────┘ └────┬────┘ └────┬─────┘
        └───────────┼───────────┘
              ┌─────▼──────────────────────┐
              │ etcd   MinIO/S3   Pulsar/Kafka │  存储层
              └────────────────────────────┘
```

- **etcd**：保存 collection、segment、索引等元数据。
- **对象存储（MinIO / S3 / OSS）**：保存 segment 数据文件与索引文件。
- **消息队列（Pulsar / Kafka）**：作为 WAL，保证写入可靠与节点解耦。

Standalone 把这些组件部署在一台机器上（通常用 Docker Compose），Distributed 则把各角色拆成独立服务。

## 3. 写入路径

1. 客户端调用 `insert` / `upsert`，请求到达 Proxy；
2. Proxy 把数据写入消息队列（WAL），立即返回成功；
3. Data Node 消费 WAL，写入内存中的 **growing segment**；
4. 达到阈值或收到 flush 请求后，segment 落盘为对象存储中的 **sealed segment**；
5. Index Node 为 sealed segment 构建向量索引和标量索引；
6. Query Node 加载 segment（原始数据 + 索引）以提供查询。

**Segment 是数据组织与调度的基本单位**。理解它有助于解释：

- 为什么刚写入的数据可能短暂查不到（取决于一致性级别）；
- 为什么索引构建是异步的；
- 为什么大量小批量写入会产生很多小 segment，需要 compaction。

## 4. 查询路径

1. Proxy 把查询广播给相关 Query Node；
2. Query Node 在已加载的 segment 上执行向量检索 + 标量过滤；
3. 各节点返回局部 Top-K；
4. Proxy 归并、去重、排序，返回全局 Top-K。

过滤与向量检索的执行顺序会影响性能：先过滤可以缩小候选集，但过滤条件的选择性、索引类型、
以及是否使用分区裁剪都会影响最终延迟。

## 5. 一致性级别

Milvus 提供四种一致性级别（collection 级或查询级）：

| 级别 | 含义 | 适用 |
|---|---|---|
| Strong | 读一定能看到之前所有已确认的写 | 对正确性要求极高、能接受更高延迟 |
| Bounded staleness | 允许落后一个可控时间窗口 | 大多数在线业务 |
| Session | 同一客户端能看到自己的写 | 常见的“写完立即查”场景 |
| Eventually | 不保证时序，只保证最终一致 | 离线分析、吞吐优先 |

> 本项目使用的 Milvus Lite 是单进程本地引擎，读写在同一进程内，通常表现为强一致；
> demo 不需要显式设置一致性级别。

## 6. 四种部署形态

| 形态 | 形态说明 | 规模 | 高可用 | 适用 |
|---|---|---|---|---|
| **Milvus Lite** | Python 库，数据在单个 `.db` 文件 | 单机、百万级以内 | 无 | 学习、Notebook、原型、边缘小应用 |
| **Standalone** | 单机 Docker/二进制，组件齐全 | 单机、千万级 | 有限（进程重启） | 中小型生产、内网服务 |
| **Distributed** | K8s 上多副本、多角色 | 亿级及以上 | 有 | 大规模生产 |
| **Zilliz Cloud** | 托管服务 | 按需 | 由服务商保证 | 不想自运维的团队 |

## 7. 本项目的 Milvus Lite 3.2 有什么特别

`milvus-lite 3.2.2` 是**纯 Python 重写**的本地引擎，不是把 C++ Milvus 打包进 wheel：

- 存储：WAL + 内存表 + 不可变 Parquet segment + manifest；
- 索引：FAISS 向量索引 + 标量/稀疏倒排索引；
- 能力：稠密/稀疏检索、BM25 全文、混合检索、分区、别名、JSON/ARRAY、动态字段等；
- 限制：单进程文件锁、无鉴权/RBAC/TLS、无 PQ/BINARY/FLOAT16/BF16/INT8、
  gRPC 适配器不支持分区级 load/release。

它暴露与 `pymilvus` 兼容的 API，所以学习时写的代码迁移到 Standalone 主要是改连接 URI。

## 8. 迁移到远程 Milvus

把 `.env` 改成：

```dotenv
MILVUS_URI=http://127.0.0.1:19530
```

`DemoSession` 会自动：

- 连接远程服务；
- 给 collection 加 `learn_<demo>_` 前缀；
- 启动时清理同前缀的旧 collection。

需要注意的差异：

- 远程 Milvus 的索引构建是异步的，`load_collection` 可能等待更久；
- 鉴权、TLS、数据库（database）需要额外配置；
- 部分 Lite 不支持的功能（如分区级 load/release）在远程可用。

## 9. 思考题

1. 为什么 Milvus 要把 WAL 和对象存储分开？
2. growing segment 和 sealed segment 对查询一致性有什么影响？
3. 什么场景应该选 Standalone 而不是 Distributed？
4. 从 Lite 迁移到 Standalone，哪些代码需要改？哪些不需要？
