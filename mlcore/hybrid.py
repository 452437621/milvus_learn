"""混合检索的融合排序工具（不依赖 Milvus）。

`manual_rrf` 是 RRF 的教学实现：当 `client.hybrid_search` 不可用时，
可以用它对多路召回结果手动融合，便于理解“只看名次”的排序思想。
"""

from __future__ import annotations

from typing import Any


def manual_rrf(
    *result_sets: list[dict],
    k: int = 60,
    limit: int = 5,
) -> list[dict]:
    """对多路命中结果做 Reciprocal Rank Fusion。

    RRF(d) = Σ 1 / (k + rank_i(d))，rank 从 1 开始。
    """
    scores: dict[Any, float] = {}
    entities: dict[Any, dict] = {}
    for hits in result_sets:
        for rank, hit in enumerate(hits, start=1):
            pk = hit.get("id")
            if pk is None:
                continue
            scores[pk] = scores.get(pk, 0.0) + 1.0 / (k + rank)
            if pk not in entities:
                entities[pk] = hit.get("entity", {})

    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]
    return [{"id": pk, "distance": score, "entity": entities[pk]} for pk, score in ordered]
