"""Xây dựng và quản lý từ vựng (Vocabulary).

Yêu cầu (docs/TASKS.md §3 D1 & docs/phases/MODEL.md):
- Xây dựng chỉ từ tập train.
- <pad> = 0, <unk> = 1.
- Hỗ trợ min_freq và max_size.
- Lưu / tải file vocab.json.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable, Sequence

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


class Vocab:
    def __init__(
        self,
        token_to_idx: dict[str, int] | None = None,
        pad_token: str = PAD_TOKEN,
        unk_token: str = UNK_TOKEN,
    ) -> None:
        if token_to_idx is not None:
            self.token_to_idx = dict(token_to_idx)
            self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        else:
            self.token_to_idx = {pad_token: PAD_IDX, unk_token: UNK_IDX}
            self.idx_to_token = {PAD_IDX: pad_token, UNK_IDX: unk_token}

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_idx = self.token_to_idx[pad_token]
        self.unk_idx = self.token_to_idx[unk_token]

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_idx

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx.get(token, self.unk_idx)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        """Chuyển danh sách token thành danh sách index."""
        unk = self.unk_idx
        stoi = self.token_to_idx
        return [stoi.get(t, unk) for t in tokens]

    def decode(self, indices: Sequence[int]) -> list[str]:
        """Chuyển danh sách index thành danh sách token."""
        itos = self.idx_to_token
        unk = self.unk_token
        return [itos.get(idx, unk) for idx in indices]

    @classmethod
    def build_from_tokens(
        cls,
        tokenized_corpus: Iterable[list[str]],
        min_freq: int = 1,
        max_size: int | None = None,
    ) -> Vocab:
        """Xây dựng từ vựng từ tập văn bản đã tokenize (chỉ lấy từ train)."""
        counter = Counter()
        for tokens in tokenized_corpus:
            counter.update(tokens)

        # Bắt đầu với pad và unk
        token_to_idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        current_idx = 2

        # Lọc theo min_freq và sắp xếp giảm dần theo tần suất
        sorted_tokens = [
            (tok, freq) for tok, freq in counter.most_common() if freq >= min_freq
        ]

        if max_size is not None:
            # Dành 2 vị trí cho <pad> và <unk>
            allowed_tokens = max(0, max_size - 2)
            sorted_tokens = sorted_tokens[:allowed_tokens]

        for tok, _ in sorted_tokens:
            if tok not in token_to_idx:
                token_to_idx[tok] = current_idx
                current_idx += 1

        return cls(token_to_idx)

    def save(self, path: Path | str) -> Path:
        """Lưu từ vựng ra file JSON."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "token_to_idx": self.token_to_idx,
        }
        target_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return target_path

    @classmethod
    def load(cls, path: Path | str) -> Vocab:
        """Đọc từ vựng từ file JSON."""
        source_path = Path(path)
        data = json.loads(source_path.read_text(encoding="utf-8"))
        vocab = cls(
            token_to_idx=data["token_to_idx"],
            pad_token=data.get("pad_token", PAD_TOKEN),
            unk_token=data.get("unk_token", UNK_TOKEN),
        )
        return vocab
