"""离线确定性伪向量（学习用，不代表真实语义模型）。

算法：
1. 规范化 + 分词（中文单字/二元组，ASCII 词）。
2. 词频加权后，用 SHA-256 把每个 token 映射到固定维度，符号取摘要比特。
3. 注入少量“概念特征”（检索、部署、美食、旅行等），让同义表达产生重叠。
4. L2 归一化。

同一段文本在任何机器、任何时间都会得到相同向量；没有任何模型下载，
也不具备真实 embedding 的同义泛化能力。生产环境请替换为真实模型。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from functools import lru_cache

from . import config
from .text import normalize, tokenize

# 概念词表：只要文本包含任一关键词，就注入一个高权重概念特征。
# 这是伪向量“看起来懂语义”的主要来源，也是它的主要局限。
CONCEPT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("concept:vector", ("向量", "vector", "embedding", "嵌入", "稠密")),
    ("concept:search", ("检索", "搜索", "查询", "相似", "召回", "search")),
    ("concept:database", ("数据库", "database", "milvus", "存储")),
    ("concept:index", ("索引", "index", "hnsw", "ivf", "ann")),
    ("concept:deploy", ("部署", "集群", "容器", "docker", "standalone", "分布式", "运维")),
    ("concept:fulltext", ("全文", "bm25", "分词", "关键词")),
    ("concept:hybrid", ("混合", "重排", "融合", "rrf", "rerank")),
    ("concept:partition", ("分区", "租户", "partition")),
    ("concept:rag", ("rag", "检索增强", "大模型", "提示词", "prompt")),
    ("concept:food", ("美食", "面条", "火锅", "早茶", "咖啡", "吃", "餐")),
    ("concept:travel", ("旅行", "旅游", "西湖", "成都", "京都", "露营", "景点", "攻略")),
    ("concept:movie", ("电影", "动画", "科幻", "悬疑", "导演", "影片")),
    ("concept:health", ("健康", "睡眠", "健身", "训练", "作息", "运动")),
    ("concept:finance", ("基金", "投资", "理财", "复利", "资产", "收益")),
    ("concept:programming", ("python", "git", "sql", "编程", "代码", "开发")),
)


def _hash_feature(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


def _add_feature(vector: list[float], token: str, weight: float) -> None:
    index, sign = _hash_feature(token, len(vector))
    vector[index] += sign * weight


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _fallback_vector(text: str, dim: int) -> list[float]:
    """空文本或极端情况下，用整段文本的摘要生成确定性向量。"""
    digest = hashlib.sha256((text or "empty").encode("utf-8")).digest()
    repeated = (digest * (dim // len(digest) + 1))[:dim]
    return _l2_normalize([byte / 255.0 - 0.5 for byte in repeated])


class PseudoEmbedder:
    """确定性的离线伪嵌入器。"""

    def __init__(self, dim: int | None = None):
        self.dim = int(dim or config.PSEUDO_EMBED_DIM)
        self.name = f"pseudo-{self.dim}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        counts: Counter[str] = Counter()
        kinds: dict[str, str] = {}
        for token, kind, _ in tokenize(text):
            counts[token] += 1
            kinds[token] = kind

        for token, count in counts.items():
            base = {"char": 0.35, "bigram": 1.0, "word": 1.4}.get(kinds.get(token, ""), 1.0)
            weight = base * (1.0 + math.log1p(count))
            _add_feature(vector, token, weight)

        normalized = normalize(text)
        for concept, keywords in CONCEPT_KEYWORDS:
            if any(keyword in normalized for keyword in keywords):
                _add_feature(vector, concept, 2.5)

        vector = _l2_normalize(vector)
        if all(value == 0.0 for value in vector):
            return _fallback_vector(normalized, self.dim)
        return vector


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("向量维度不一致")
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


@lru_cache(maxsize=1)
def get_embedder():
    """按配置返回嵌入器；OpenAI 配置失败时回退伪向量。"""
    if config.EMBEDDING_PROVIDER == "openai":
        try:
            from .providers import OpenAICompatibleEmbedder

            return OpenAICompatibleEmbedder(
                base_url=config.OPENAI_BASE_URL,
                api_key=config.OPENAI_API_KEY,
                model=config.OPENAI_EMBED_MODEL,
                dim=config.OPENAI_EMBED_DIM,
            )
        except Exception as exc:  # noqa: BLE001 - 学习项目需要优雅回退
            print(f"[警告] 真实 embedding 初始化失败，回退离线伪向量：{exc}")
    return PseudoEmbedder()


def reset_embedder() -> None:
    """清空 get_embedder 缓存（切换配置的测试使用）。"""
    get_embedder.cache_clear()
