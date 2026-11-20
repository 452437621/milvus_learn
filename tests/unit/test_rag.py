"""rag 模块：切块、Prompt、离线回答与来源表格。"""

from __future__ import annotations

import pytest

from mlcore import corpus, rag

pytestmark = pytest.mark.unit


def _hit(doc_id: int, text: str, title: str = "标题", score: float = 0.5) -> dict:
    return {
        "id": doc_id,
        "distance": score,
        "entity": {
            "doc_id": doc_id,
            "title": title,
            "category": "向量数据库",
            "chunk_index": 0,
            "text": text,
        },
    }


def test_chunk_text_empty_and_short() -> None:
    assert rag.chunk_text("") == []
    assert rag.chunk_text("   ") == []
    assert rag.chunk_text("短句。") == ["短句。"]


def test_chunk_text_long_has_overlap_and_limit() -> None:
    text = "x" * 400
    chunks = rag.chunk_text(text, max_chars=100, overlap=20)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert chunks[1][:20] == chunks[0][-20:]
    assert "".join(chunks[0][0:100]) == text[:100]


def test_chunk_text_chinese_sentence_boundary() -> None:
    text = "第一句话。" * 30
    chunks = rag.chunk_text(text, max_chars=50, overlap=5)
    assert chunks
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert all(chunk.endswith("。") for chunk in chunks)


def test_build_chunks_metadata() -> None:
    docs = corpus.load_docs()[:2]
    chunks = rag.build_chunks(docs)
    assert [chunk.chunk_id for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert chunks[0].doc_id == docs[0]["id"]
    assert chunks[0].title == docs[0]["title"]
    assert docs[0]["title"] in chunks[0].embedding_text
    assert chunks[0].text in chunks[0].embedding_text


def test_prompt_numbering() -> None:
    hits = [_hit(1, "内容一。"), _hit(2, "内容二。")]
    prompt = rag.build_prompt("问题？", hits)
    assert prompt.startswith(rag.SYSTEM_PROMPT)
    assert "[1] 《标题》" in prompt
    assert "[2] 《标题》" in prompt
    assert prompt.rstrip().endswith("问题：问题？")


def test_offline_answer_no_hits() -> None:
    answer = rag.offline_answer([])
    assert "没有检索到" in answer


def test_offline_answer_numbers_and_truncates() -> None:
    hits = [_hit(1, "第一段。" * 20), _hit(2, "第二段。" * 20)]
    answer = rag.offline_answer(hits, max_chars=10)
    assert "[1] 第一段。" in answer
    assert "[2]" not in answer  # 超出预算就停止


class _FakeChat:
    def __init__(self, reply: str | None = None, error: Exception | None = None):
        self.reply = reply
        self.error = error

    def generate(self, system: str, user: str):
        if self.error:
            raise self.error
        return self.reply


def test_answer_question_with_chat() -> None:
    hits = [_hit(1, "内容。")]
    answer, mode, prompt = rag.answer_question("问题？", hits, _FakeChat("模型回答"))
    assert answer == "模型回答"
    assert "大模型" in mode
    assert "问题？" in prompt


def test_answer_question_chat_error_falls_back(capsys) -> None:
    hits = [_hit(1, "内容。")]
    answer, mode, _ = rag.answer_question("问题？", hits, _FakeChat(error=RuntimeError("boom")))
    assert "离线抽取式回答" in mode
    assert answer.startswith("（离线抽取式回答")
    assert "boom" in capsys.readouterr().out


def test_answer_question_offline_without_chat() -> None:
    hits = [_hit(1, "内容。")]
    _, mode, _ = rag.answer_question("问题？", hits, None)
    assert mode == "离线抽取式回答"


def test_source_rows() -> None:
    hits = [_hit(1, "长文本" * 40, score=0.123456)]
    rows = rag.source_rows(hits)
    assert len(rows) == len(hits)
    assert len(rows[0]) == 6
    assert rows[0][1] == 0.1235
    assert rows[0][-1].endswith("…")
