"""pretty 模块：打印与控制台格式化。"""

from __future__ import annotations

import pytest

from mlcore import pretty

pytestmark = pytest.mark.unit


def test_banner_and_section(capsys) -> None:
    pretty.banner("标题")
    pretty.section("小节")
    pretty.separator()
    out = capsys.readouterr().out
    assert "标题" in out
    assert "小节" in out
    assert "=" * 78 in out
    assert "-" * 78 in out


def test_kv(capsys) -> None:
    pretty.kv("维度", 256)
    assert "维度: 256" in capsys.readouterr().out


def test_hit_entity_variants() -> None:
    assert pretty.hit_entity({"entity": {"a": 1}}) == {"a": 1}
    assert pretty.hit_entity({"id": 1, "title": "T", "distance": 0.5}) == {"title": "T"}
    assert pretty.hit_entity("not a dict") == {}
    assert pretty.hit_entity(None) == {}


@pytest.mark.parametrize(
    "hit, expected",
    [
        ({"distance": 0.5}, 0.5),
        ({"score": 1}, 1.0),
        ({"similarity": 2.5}, 2.5),
        ({"distance": "x"}, None),
        ({}, None),
    ],
)
def test_hit_score(hit, expected) -> None:
    assert pretty.hit_score(hit) == expected


def test_hit_id_and_title() -> None:
    assert pretty.hit_id({"id": 7, "entity": {}}) == 7
    assert pretty.hit_id({"entity": {"id": 8}}) == 8
    assert pretty.hit_title({"entity": {"title": "标题"}}) == "标题"
    assert pretty.hit_title({"entity": {"text": "x" * 100}}) == "x" * 32
    assert pretty.hit_title({}) == "（无标题）"


def test_print_hits(capsys) -> None:
    pretty.print_hits([])
    assert "没有命中结果" in capsys.readouterr().out

    hits = [{"id": 1, "distance": 0.9, "entity": {"title": "T", "category": "c", "text": "正文"}}]
    pretty.print_hits(hits, fields=("category",))
    out = capsys.readouterr().out
    assert "[1]" in out
    assert "0.9000" in out
    assert "《T》" in out
    assert "category: c" in out
    assert "正文" in out


def test_hits_to_rows() -> None:
    hits = [
        {
            "id": 2,
            "distance": 0.123456,
            "entity": {"title": "B", "category": "c", "year": 2024, "text": "y" * 200},
        },
        {"id": 1, "score": 0.2, "entity": {"title": "A", "category": "d"}},
    ]
    rows = pretty.hits_to_rows(hits, fields=("title", "category", "year"))
    assert [row[0] for row in rows] == [1, 2]
    assert rows[0][1] == 0.1235
    assert rows[0][2:5] == ["B", "c", 2024]
    assert rows[0][-1].endswith("…")
    assert rows[1][1] == 0.2
    assert rows[1][-1] == ""


def test_plain_rows() -> None:
    assert pretty.plain_rows(None) == []
    assert pretty.plain_rows([{"a": 1}]) == [{"a": 1}]
