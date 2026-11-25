"""Tokenizer cho dữ liệu văn bản IMDb.

Yêu cầu (docs/TASKS.md §3 D1 & docs/phases/MODEL.md):
- Lowercase (vì GloVe 6B uncased).
- Giữ các token phủ định quan trọng như 'not', "n't".
- Giữ dấu câu cảm thán và hỏi: '!', '?'.
- Không bỏ stopwords (để giữ tín hiệu chuỗi và cấu trúc phủ định).
"""
from __future__ import annotations

import re

# Tách riêng n't trước khi token hóa
_NT_RE = re.compile(r"\b(\w+)n't\b", flags=re.IGNORECASE)
# Token: chuỗi ký tự từ vựng, hoặc n't, hoặc các dấu câu riêng biệt
_TOKEN_RE = re.compile(r"n't|\w+|[^\w\s]", flags=re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize văn bản thành danh sách token chữ thường, giữ n't, not, !, ?."""
    if not text:
        return []

    text_lower = text.lower()
    # Tách don't -> do n't, can't -> ca n't / can n't, wasn't -> was n't
    text_separated = _NT_RE.sub(r"\1 n't", text_lower)
    tokens = _TOKEN_RE.findall(text_separated)
    return tokens
