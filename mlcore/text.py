"""文本规范化与轻量中文分词。

Milvus 本身不负责把文本变成向量；本项目用“单字 + 二元组 + ASCII 词”
的确定性分词，为离线伪向量和稀疏回退提供一致的特征。
"""

from __future__ import annotations

import re
import unicodedata

_CJK = r"[一-鿿]"
_TOKEN_RE = re.compile(rf"{_CJK}+|[A-Za-z0-9]+(?:[._+#'-][A-Za-z0-9]+)*")

# 常见虚词/代词在单字层面噪声较大，跳过单字特征。
_STOP_CHARS = set(
    "的了和是在有我你他她它们这那就都很也还与及或而之其所以为对从到把被让给"
    "一个不没有中上下里外时年月日"
)

# 二元组仍会保留这些字，避免“数据”“数据库”之类的词根被切断。


def normalize(text: str) -> str:
    """NFKC 规范化、转小写并压缩空白，保证相同输入得到相同向量。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[tuple[str, str, float]]:
    """返回 (token, kind, weight)。

    kind 取值：char / bigram / word。
    - 中文单字：权重 0.35（太短，信息量低）
    - 中文二元组：权重 1.0（主要语义单元）
    - ASCII 词/数字：权重 1.4
    - 单个汉字单独成段时，若不在停用字表，权重提高到 0.8
    """
    tokens: list[tuple[str, str, float]] = []
    for match in _TOKEN_RE.finditer(normalize(text)):
        chunk = match.group(0)
        if re.fullmatch(rf"{_CJK}+", chunk):
            if len(chunk) == 1:
                if chunk not in _STOP_CHARS:
                    tokens.append(("c:" + chunk, "char", 0.8))
                continue
            for char in chunk:
                if char not in _STOP_CHARS:
                    tokens.append(("c:" + char, "char", 0.35))
            for left, right in zip(chunk, chunk[1:]):
                tokens.append(("b:" + left + right, "bigram", 1.0))
        else:
            tokens.append(("w:" + chunk, "word", 1.4))
    return tokens


def terms_for_sparse(text: str) -> list[str]:
    """稀疏回退只用二元组和 ASCII 词，单字噪声太大。"""
    return [token for token, kind, _ in tokenize(text) if kind != "char"]
