"""手动稀疏向量（BM25 不可用时的教学回退）。

注意：这不是 BM25。它只做词频加权和长度归一，然后用向量内积（IP）
计算相似度。把它放在这里，是为了在 Milvus Lite 的原生 BM25 不可用时，
demo 仍然能演示“稀疏向量 + 倒排索引”的基本用法。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from .text import terms_for_sparse

# Milvus 的稀疏索引要求维度是 uint32，且最大值为 2^32 - 2。
MAX_SPARSE_INDEX = 2**32 - 2


def manual_sparse(text: str) -> dict[int, float]:
    terms = terms_for_sparse(text)
    if not terms:
        return {}

    counts = Counter(terms)
    length_norm = math.sqrt(sum(counts.values()))
    vector: dict[int, float] = {}
    for term, tf in counts.items():
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % (MAX_SPARSE_INDEX + 1)
        weight = (1.0 + math.log1p(tf)) / length_norm
        vector[index] = vector.get(index, 0.0) + weight
    return vector
