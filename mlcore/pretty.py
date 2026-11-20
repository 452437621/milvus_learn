"""控制台输出与检索结果格式化。

中文在等宽字体中的宽度与英文不同，这里用块状输出而不是表格对齐。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

WIDTH = 78


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def section(title: str) -> None:
    print()
    print("-" * WIDTH)
    print(f"  {title}")
    print("-" * WIDTH)


def kv(label: str, value: Any) -> None:
    print(f"  {label}: {value}")


def plain_rows(rows: Any) -> list[dict]:
    """把 pymilvus 的 ExtraList 转成普通 list[dict]，避免打印出额外包装信息。"""
    if not rows:
        return []
    return [dict(row) for row in rows]


def separator() -> None:
    print("-" * WIDTH)


def _entity(hit: Any) -> dict:
    if not isinstance(hit, dict):
        return {}
    entity = hit.get("entity")
    if isinstance(entity, dict):
        return entity
    # 某些返回结构把字段直接放在 hit 顶层
    return {key: value for key, value in hit.items() if key not in ("id", "distance", "score")}


def hit_entity(hit: Any) -> dict:
    """公开的结果实体访问器（Gradio / RAG 模块复用）。"""
    return _entity(hit)


def hit_score(hit: Any) -> float | None:
    if not isinstance(hit, dict):
        return None
    for key in ("distance", "score", "similarity"):
        value = hit.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def hit_id(hit: Any) -> Any:
    if isinstance(hit, dict):
        if "id" in hit:
            return hit["id"]
        return _entity(hit).get("id")
    return None


def hit_title(hit: Any) -> str:
    entity = _entity(hit)
    title = entity.get("title")
    if title:
        return str(title)
    text = entity.get("text")
    if text:
        return str(text)[:32]
    return "（无标题）"


def print_hits(
    hits: Iterable[Any],
    fields: tuple[str, ...] = (),
    text_field: str = "text",
    max_text: int = 60,
) -> None:
    hits = list(hits)
    if not hits:
        print("  （没有命中结果）")
        return
    for rank, hit in enumerate(hits, start=1):
        score = hit_score(hit)
        score_text = f"{score:.4f}" if score is not None else "-"
        entity = _entity(hit)
        category = entity.get("category", "")
        print(f"  [{rank}] score={score_text:<9} 《{hit_title(hit)}》  {category}")
        for field in fields:
            if field in entity:
                print(f"        {field}: {entity[field]}")
        text = entity.get(text_field)
        if text:
            compact = " ".join(str(text).split())
            if len(compact) > max_text:
                compact = compact[:max_text] + "…"
            print(f"        {compact}")


def hits_to_rows(
    hits: Iterable[Any],
    fields: tuple[str, ...] = ("title", "category", "year"),
    text_field: str = "text",
    max_text: int = 80,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for rank, hit in enumerate(hits, start=1):
        entity = _entity(hit)
        score = hit_score(hit)
        text = " ".join(str(entity.get(text_field, "")).split())
        if len(text) > max_text:
            text = text[:max_text] + "…"
        row = [rank, None if score is None else round(score, 4)]
        row.extend(entity.get(field, "") for field in fields)
        row.append(text)
        rows.append(row)
    return rows
