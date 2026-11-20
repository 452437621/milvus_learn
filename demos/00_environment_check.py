"""Demo 00：环境自检与能力探测。

默认只做静态检查（版本、签名、伪向量）。
加 --deep 会真正创建 Milvus Lite 数据库，实测 CRUD、别名、分区、group-by、
BM25 与混合检索是否可用，后续 demo 以本脚本的结论为准。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlcore import config, pretty
from mlcore.embedding import PseudoEmbedder, cosine


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "未安装"


def static_checks() -> None:
    pretty.banner("Demo 00：环境自检")
    pretty.section("1. 版本")
    pretty.kv("Python", sys.version.split()[0])
    pretty.kv("pymilvus", package_version("pymilvus"))
    pretty.kv("milvus-lite", package_version("milvus-lite"))
    pretty.kv("gradio", package_version("gradio"))
    pretty.kv("python-dotenv", package_version("python-dotenv"))

    pretty.section("2. 配置")
    print(config.describe())
    if config.DATA_DIR_FALLBACK:
        print(
            "\n  [说明] 仓库路径包含非 ASCII 字符，本地数据已自动改存到 "
            f"{config.DATA_DIR}。\n"
            "         FAISS 在 Windows 上无法写入中文路径；可用 MILVUS_DB_DIR 覆盖。"
        )

    pretty.section("3. 伪向量自检")
    embedder = PseudoEmbedder()
    left = embedder.embed_query("向量数据库与相似度检索")
    right = embedder.embed_query("向量数据库与相似度检索")
    norm = math.sqrt(sum(value * value for value in left))
    related = cosine(left, embedder.embed_query("向量检索和语义搜索"))
    unrelated = cosine(left, embedder.embed_query("四川火锅和广式早茶"))
    pretty.kv("维度", embedder.dim)
    pretty.kv("范数", f"{norm:.6f}")
    pretty.kv("确定性", "PASS" if left == right else "FAIL")
    pretty.kv("同主题相似度", f"{related:.4f}")
    pretty.kv("跨主题相似度", f"{unrelated:.4f}")
    pretty.kv("区分度", "PASS" if related > unrelated else "WARN")

    pretty.section("4. 关键 API 签名（用于核对当前 pymilvus 版本）")
    try:
        from pymilvus import MilvusClient

        from mlcore.milvus_utils import method_signature

        for name in (
            "create_collection",
            "prepare_index_params",
            "insert",
            "upsert",
            "search",
            "query",
            "delete",
            "hybrid_search",
            "create_partition",
            "create_alias",
            "alter_alias",
            "query_iterator",
        ):
            print(f"  {method_signature(MilvusClient, name)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  无法读取签名：{exc}")


def _drop(client, name: str) -> None:
    from mlcore.milvus_utils import drop_collection_quietly

    drop_collection_quietly(client, name)


def _probe_basic(client, session) -> tuple[bool, str]:
    from pymilvus import DataType, MilvusClient

    name = session.name("probe_basic")
    try:
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=4)
        schema.add_field("category", DataType.VARCHAR, max_length=32)
        schema.add_field("text", DataType.VARCHAR, max_length=200)
        client.create_collection(collection_name=name, schema=schema)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 8, "efConstruction": 100},
        )
        client.create_index(collection_name=name, index_params=index_params)

        client.insert(
            collection_name=name,
            data=[
                {"id": 1, "vector": [1.0, 0.0, 0.0, 0.0], "category": "a", "text": "向量检索"},
                {"id": 2, "vector": [0.0, 1.0, 0.0, 0.0], "category": "b", "text": "美食旅行"},
            ],
        )
        client.load_collection(collection_name=name)
        hits = client.search(
            collection_name=name,
            data=[[1.0, 0.0, 0.0, 0.0]],
            limit=2,
            output_fields=["text"],
        )
        if not hits or not hits[0]:
            return False, "search 返回空结果"
        return True, f"可创建/写入/检索，命中 {len(hits[0])} 条"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    finally:
        _drop(client, name)


def _probe_alias(client, session) -> tuple[bool, str]:
    from pymilvus import DataType, MilvusClient

    name = session.name("probe_alias")
    alias = session.name("probe_alias_ref")
    try:
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
        client.create_collection(collection_name=name, schema=schema)
        client.create_alias(collection_name=name, alias=alias)
        aliases = client.list_aliases(collection_name=name)
        client.drop_alias(alias=alias)
        return True, f"create/list/drop alias 可用：{aliases}"
    except Exception as exc:  # noqa: BLE001
        _drop(client, name)
        return False, str(exc)


def _probe_partition(client, session) -> tuple[bool, str]:
    from pymilvus import DataType, MilvusClient

    name = session.name("probe_partition")
    try:
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
        client.create_collection(collection_name=name, schema=schema)
        client.create_partition(collection_name=name, partition_name="p1")
        client.insert(
            collection_name=name,
            data=[{"id": 1, "vector": [0.1, 0.2]}],
            partition_name="p1",
        )
        partitions = client.list_partitions(collection_name=name)
        return True, f"显式分区可用：{partitions}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    finally:
        _drop(client, name)


def _probe_group_by(client, session) -> tuple[bool, str]:
    from pymilvus import DataType, MilvusClient

    name = session.name("probe_group")
    try:
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
        schema.add_field("category", DataType.VARCHAR, max_length=16)
        client.create_collection(collection_name=name, schema=schema)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        client.create_index(collection_name=name, index_params=index_params)
        client.insert(
            collection_name=name,
            data=[
                {"id": 1, "vector": [1.0, 0.0], "category": "a"},
                {"id": 2, "vector": [0.9, 0.1], "category": "a"},
                {"id": 3, "vector": [0.8, 0.2], "category": "b"},
            ],
        )
        client.load_collection(collection_name=name)
        hits = client.search(
            collection_name=name,
            data=[[1.0, 0.0]],
            limit=2,
            group_by_field="category",
            output_fields=["category"],
        )
        categories = [hit["entity"].get("category") for hit in (hits[0] if hits else [])]
        return True, f"group_by_field 可用，返回类目：{categories}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    finally:
        _drop(client, name)


def _probe_bm25_and_hybrid(client, session) -> tuple[bool, str, bool, str]:
    """创建一个带 BM25 Function 和稠密向量的集合，测试全文与混合检索。"""
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    name = session.name("probe_bm25")
    analyzer_used = "standard"
    created = False

    for analyzer in ("jieba", "standard"):
        _drop(client, name)
        try:
            schema = MilvusClient.create_schema(auto_id=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field(
                "text",
                DataType.VARCHAR,
                max_length=2000,
                enable_analyzer=True,
                enable_match=True,
                analyzer_params={"type": analyzer},
            )
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
            schema.add_function(
                Function(
                    name="bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["sparse"],
                )
            )
            client.create_collection(collection_name=name, schema=schema)
            analyzer_used = analyzer
            created = True
            break
        except Exception:  # noqa: BLE001
            continue

    if not created:
        return False, "无法创建带 BM25 Function 的集合", False, "因 BM25 失败未测试 hybrid"

    try:
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={},
        )
        index_params.add_index(
            field_name="dense",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 8, "efConstruction": 100},
        )
        client.create_index(collection_name=name, index_params=index_params)

        client.insert(
            collection_name=name,
            data=[
                {"text": "Milvus 支持向量检索与全文检索", "dense": [1.0, 0.0, 0.0, 0.0]},
                {"text": "四川火锅麻辣鲜香", "dense": [0.0, 1.0, 0.0, 0.0]},
                {"text": "HNSW 是近邻图索引", "dense": [0.5, 0.5, 0.0, 0.0]},
            ],
        )
        client.load_collection(collection_name=name)

        bm25_hits = client.search(
            collection_name=name,
            data=["向量 检索"],
            anns_field="sparse",
            limit=2,
            output_fields=["text"],
        )
        bm25_ok = bool(bm25_hits and bm25_hits[0])

        hybrid_ok = False
        hybrid_detail = "hybrid_search 不可用"
        if hasattr(client, "hybrid_search"):
            try:
                from pymilvus import AnnSearchRequest, RRFRanker

                dense_req = AnnSearchRequest(
                    data=[[1.0, 0.0, 0.0, 0.0]],
                    anns_field="dense",
                    param={"metric_type": "COSINE", "params": {"ef": 32}},
                    limit=3,
                )
                sparse_req = AnnSearchRequest(
                    data=["向量 检索"],
                    anns_field="sparse",
                    param={"metric_type": "BM25", "params": {}},
                    limit=3,
                )
                hybrid_hits = client.hybrid_search(
                    collection_name=name,
                    reqs=[dense_req, sparse_req],
                    ranker=RRFRanker(60),
                    limit=3,
                    output_fields=["text"],
                )
                hybrid_ok = bool(hybrid_hits and hybrid_hits[0])
                hybrid_detail = (
                    f"RRF 混合检索可用，命中 {len(hybrid_hits[0]) if hybrid_hits else 0} 条"
                )
            except Exception as exc:  # noqa: BLE001
                hybrid_detail = str(exc)

        return (
            bm25_ok,
            f"BM25 可用（analyzer={analyzer_used}）" if bm25_ok else "BM25 查询失败",
            hybrid_ok,
            hybrid_detail,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"BM25 不可用：{exc}", False, "因 BM25 失败未测试 hybrid"
    finally:
        _drop(client, name)


def deep_checks() -> None:
    from mlcore.milvus_utils import DemoSession

    pretty.section("5. Milvus Lite 深度探测（--deep）")
    results: list[tuple[str, bool, str]] = []
    target_dir = config.DATA_DIR / "runs" / "00_probe"

    with DemoSession("00_probe") as session:
        client = session.client
        pretty.kv("连接 URI", session.uri)

        ok, detail = _probe_basic(client, session)
        results.append(("基础 CRUD / 向量检索", ok, detail))
        ok, detail = _probe_alias(client, session)
        results.append(("别名 Alias", ok, detail))
        ok, detail = _probe_partition(client, session)
        results.append(("显式分区 Partition", ok, detail))
        ok, detail = _probe_group_by(client, session)
        results.append(("分组搜索 group_by", ok, detail))
        bm25_ok, bm25_detail, hybrid_ok, hybrid_detail = _probe_bm25_and_hybrid(client, session)
        results.append(("BM25 全文检索", bm25_ok, bm25_detail))
        results.append(("混合检索 hybrid_search", hybrid_ok, hybrid_detail))

    pretty.section("6. 能力矩阵")
    for label, ok, detail in results:
        status = "PASS" if ok else "WARN"
        print(f"  [{status}] {label}: {detail}")

    print("\n  结论：")
    print("  - PASS 的能力会被后续 demo 直接使用。")
    print("  - WARN 不代表环境坏了，demo 会自动降级（如手动稀疏 + 手动 RRF）。")

    shutil.rmtree(target_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milvus 学习项目环境自检")
    parser.add_argument("--deep", action="store_true", help="实测 Milvus Lite 能力")
    args = parser.parse_args()

    static_checks()
    if args.deep:
        try:
            deep_checks()
        except Exception as exc:  # noqa: BLE001
            print(f"\n[错误] 深度探测失败：{exc}")
            print("请检查依赖安装、数据目录权限与文件锁。")
            raise SystemExit(1) from exc
    else:
        print("\n  （未运行深度探测；加 --deep 可实测 Milvus Lite 能力。）")


if __name__ == "__main__":
    main()
