# 12 Lite / Standalone / Distributed / Zilliz Cloud

## 1. 四种形态对比

| 维度 | Milvus Lite | Standalone | Distributed | Zilliz Cloud |
|---|---|---|---|---|
| 部署 | Python 库，单 `.db` 文件 | 单机 Docker/二进制 | Kubernetes 多角色 | 托管服务 |
| 数据规模 | 单机、百万级以内 | 单机、千万级 | 亿级及以上 | 按套餐/按需 |
| 高可用 | 无 | 有限（进程级） | 多副本 | 服务商 SLA |
| 扩展性 | 无 | 纵向扩展 | 水平扩展 | 自动/按需 |
| 鉴权/RBAC | 无 | 支持 | 支持 | 支持 |
| 运维成本 | 最低 | 中 | 高 | 最低（托管） |
| 适用 | 学习、Notebook、原型、边缘 | 中小生产、内网 | 大规模生产 | 不想自运维 |

## 2. Milvus Lite

本项目使用的形态。

**优点**

- 零服务、零 Docker，`pip install` 即用；
- 数据就是一个 `.db` 文件，便于备份/拷贝/实验；
- API 与 Standalone 基本一致，迁移成本低；
- 支持稠密/稀疏检索、BM25、混合检索、分区、别名、JSON/ARRAY 等常用能力。

**限制**

- 单进程、文件锁，不能多进程共享；
- 无鉴权/RBAC/TLS；
- 不支持 PQ、BINARY、FLOAT16/BF16、INT8 向量；
- gRPC 适配器不支持分区级 load/release；
- 适合本地和小规模，不是分布式生产引擎；
- 3.2 起是纯 Python 重写，旧版 `.db` 格式不兼容，需要重新导入。

**适用**

- 学习、Notebook、CI 测试；
- 桌面/边缘小应用；
- 原型验证（之后再迁移）。

## 3. Standalone

单机部署，组件齐全（etcd、MinIO、消息队列、各角色进程）。

- 优点：功能完整、部署相对简单、可暴露网络服务；
- 缺点：单机瓶颈，高可用有限；
- 适用：中小型生产、内网服务、团队共享实例。

典型启动方式（Docker Compose）：

```bash
# 仅示意，本项目不要求 Docker
docker compose up -d
```

## 4. Distributed

Kubernetes 上按角色拆分、多副本部署。

- 优点：水平扩展、高可用、支持大规模数据；
- 缺点：组件多、运维复杂、成本高；
- 适用：亿级向量、高并发、多租户平台。

核心挑战：

- 容量规划与分片；
- 索引构建的资源隔离；
- 查询节点的缓存与负载均衡；
- 监控、升级、故障恢复。

## 5. Zilliz Cloud

Milvus 的托管服务：

- 免运维、自动扩缩容、SLA；
- 内置鉴权、备份、监控；
- 按套餐/用量计费；
- 适合不想维护基础设施的团队。

注意网络、数据合规与成本。

## 6. 迁移路径

同一套 `MilvusClient` 代码，迁移主要是配置变化：

| 项目 | Lite | 远程 |
|---|---|---|
| 连接 | `MilvusClient("./x.db")` | `MilvusClient(uri="http://host:19530", token=...)` |
| 鉴权 | 无 | token / 用户名密码 / TLS |
| 索引 | 本地同步构建 | 异步构建，需等待 |
| 分区加载 | 整 collection | 支持分区级 |
| 备份 | 复制 `.db` | 快照/对象存储备份 |

本项目已经把这层抽象进 `DemoSession`：设置 `MILVUS_URI` 即可切换。

```dotenv
MILVUS_URI=http://127.0.0.1:19530
```

## 7. 选型建议

```text
只是学习/演示            -> Milvus Lite
单机、团队共享、有网络服务 -> Standalone
亿级数据、高可用、多租户  -> Distributed 或 Zilliz Cloud
不想运维                 -> Zilliz Cloud
```

不要为了“以后可能变大”一上来就上 Distributed；先用 Lite/Standalone 验证业务，
再根据真实数据量与 SLA 决定。

## 8. 思考题

1. Lite 的哪些限制会在生产环境变成硬伤？
2. 为什么远程 Milvus 的索引构建是异步的？
3. 从 Lite 迁移到 Standalone，哪些代码需要改？
4. 什么情况下托管服务比自建更划算？
