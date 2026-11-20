# 03 数据模型与 Schema 设计

## 1. 层级关系

```text
Database
└─ Collection          # 类似表
   ├─ Partition        # 逻辑分片（可选）
   └─ Segment          # 物理存储单位（系统管理）
      └─ Entity        # 一行数据
```

- **Database**：命名空间，默认 `default`。
- **Collection**：一组同构实体，必须定义 Schema。
- **Partition**：collection 内的逻辑划分，可按租户/时间/类别隔离数据。
- **Segment**：系统内部的存储与调度单位，用户一般不直接操作。
- **Entity**：一条记录，由主键 + 若干字段组成。

## 2. 主键

每个 collection 必须有且只有一个主键字段：

- `INT64` 主键 + `auto_id=True`：由 Milvus 生成递增 ID；
- `INT64` 主键 + `auto_id=False`：客户端提供；
- `VARCHAR` 主键：适合业务 ID（如 `doc_001`），需设置 `max_length`。

```python
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
# 或
schema.add_field("doc_id", DataType.VARCHAR, is_primary=True, max_length=64)
```

> 唯一性：`insert` 是追加写入；需要“按主键覆盖”时使用 `upsert`。
> 业务上要求主键唯一时，应由写入方保证，或统一走 upsert。

## 3. 字段类型

| 类型 | 说明 | 注意 |
|---|---|---|
| `BOOL` | 布尔 | |
| `INT8/16/32/64` | 整数 | 按范围选择 |
| `FLOAT` / `DOUBLE` | 浮点 | 评分、价格 |
| `VARCHAR` | 字符串 | 必须设置 `max_length` |
| `JSON` | 半结构化对象 | 支持路径过滤 |
| `ARRAY` | 同类型数组 | 支持 `ARRAY_CONTAINS` |
| `FLOAT_VECTOR` | 稠密向量 | 必须指定 `dim` |
| `SPARSE_FLOAT_VECTOR` | 稀疏向量 | 用于 BM25/关键词 |
| `GEOMETRY` / `TIMESTAMPTZ` | 地理/时间 | 本项目不展开 |

本项目语料字段：

```python
id        INT64        主键
title     VARCHAR(200)
category  VARCHAR(64)
tags      ARRAY<VARCHAR>
text      VARCHAR(4096)
year      INT32
rating    FLOAT
price     FLOAT
is_free   BOOL
author    VARCHAR(100)
meta      JSON
embedding FLOAT_VECTOR(dim)
```

## 4. 动态字段与 Function

- **动态字段（dynamic field）**：Schema 未声明的字段可以写入 `$meta`，
  适合原型阶段快速迭代；生产环境建议显式建模，便于索引和类型校验。
- **Function**：由 Milvus 在写入时自动生成字段，例如把 `text` 分词后生成 BM25 稀疏向量：

```python
schema.add_field(
    "text", DataType.VARCHAR, max_length=65535, enable_analyzer=True, enable_match=True
)
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
schema.add_function(
    Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse"],
    )
)
```

使用 Function 后，**不要**手动写 `sparse` 字段，也不要给查询传预计算向量；
直接把原始文本传给搜索接口，由 Milvus 完成分析。

## 5. nullable、默认值与动态字段

- `nullable=True`：允许显式写入 null；
- `default_value=...`：未提供该字段时使用默认值；
- 动态字段：未在 Schema 中声明的键进入 `$meta`。

设计建议：

- 生产 Schema 尽量显式声明所有过滤字段；
- 经常过滤的字段使用合适的标量类型，必要时建 `INVERTED` 索引；
- 不常过滤的扩展信息放 JSON，避免 Schema 膨胀；
- `max_length` 不要盲目设成超大值，会影响内存与校验成本。

## 6. CRUD 语义

| 操作 | 行为 |
|---|---|
| `insert` | 追加新实体 |
| `upsert` | 按主键插入或覆盖 |
| `get` | 按主键取回 |
| `query` | 按标量表达式取回 |
| `search` | 向量相似度 Top-K，可带过滤 |
| `delete` | 按主键或过滤表达式删除 |
| `truncate` | 清空 collection 数据（保留结构） |

注意：

- 删除是“标记删除 + 后台 compaction”，不是立即释放磁盘；
- 频繁 upsert 会产生旧版本数据，需要 compaction 回收；
- 大批量导入应使用批量 insert，而不是逐条写。

## 7. Schema 设计原则

1. **先确定查询模式**：按什么过滤、按什么排序、返回哪些字段；
2. **主键稳定**：不要用会变化的值做 VARCHAR 主键；
3. **向量维度与模型绑定**：换 embedding 模型通常意味着重建 collection；
4. **过滤字段单独建模**：不要把所有元数据都塞进 JSON；
5. **控制单条大小**：大文本拆块后入 RAG 集合，而不是直接塞进主表；
6. **为索引留余地**：标量过滤字段考虑 `INVERTED`，高基数字段谨慎建索引。

## 8. 对应 Demo：`01_hello_milvus.py` 与 `03_crud.py`

`01` 用显式 Schema 建一个小 collection，演示：

- `create_schema` / `add_field` / `prepare_index_params`；
- `create_collection` / `insert` / `load_collection`；
- `search` / `query(count(*))` / `describe_collection`。

`03_crud.py` 在 25 条语料上演示：

- 批量 insert；
- upsert 同主键后的数量变化；
- `get` / `query` / `delete`；
- nullable、default、JSON 字段的读写。

运行：

```bash
venv\Scripts\python.exe demos\01_hello_milvus.py
venv\Scripts\python.exe demos\03_crud.py
```

## 9. 思考题

1. 为什么 `VARCHAR` 必须指定 `max_length`？
2. `insert` 和 `upsert` 在数据量、索引、查询结果上分别有什么影响？
3. 什么字段适合放 JSON，什么字段必须单独建模？
4. BM25 Function 生成的稀疏字段为什么不能手动写入？
