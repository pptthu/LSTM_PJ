"""PyTorch Dataset & DataLoader cho dữ liệu phân loại cảm xúc IMDb.

Yêu cầu (docs/TASKS.md §3 D1, D7 & docs/phases/MODEL.md):
- Trả về tensor input_ids, label và độ dài thật (lengths) để dùng với pack_padded_sequence.
- Hỗ trợ cắt ngắn (truncation: 'post', 'pre', 'head_tail') và đệm động (padding_idx=0).
- Theo dõi thông tin phục vụ phân tích lỗi: was_truncated, n_tokens, n_unk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.tokenizer import tokenize
from src.data.vocab import PAD_IDX, UNK_IDX, Vocab
from src.utils.paths import data_dir

VALID_TRUNCATIONS = ("post", "pre", "head_tail")


def truncate_tokens(tokens: list[str], max_len: int, truncation: str = "post") -> list[str]:
    """Cắt danh sách token theo chiến lược:
    - 'post': giữ `max_len` token đầu (cắt phần đuôi).
    - 'pre': giữ `max_len` token cuối (cắt phần đầu).
    - 'head_tail': giữ nửa đầu + nửa cuối (bảo toàn mở bài và câu kết luận).
    """
    if len(tokens) <= max_len:
        return tokens
    if max_len <= 0:
        return []
    mode = truncation.lower()
    if mode == "post":
        return tokens[:max_len]
    if mode == "pre":
        return tokens[-max_len:]
    if mode == "head_tail":
        head_len = max_len // 2
        tail_len = max_len - head_len
        if head_len == 0:
            return tokens[-tail_len:]
        return tokens[:head_len] + tokens[-tail_len:]
    raise ValueError(f"truncation phải thuộc {VALID_TRUNCATIONS}, nhận được {truncation!r}")


class IMDbDataset(Dataset):
    """Dataset quản lý dữ liệu review IMDb."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int] | None = None,
        uids: Sequence[str] | None = None,
        ratings: Sequence[int] | None = None,
        vocab: Vocab | None = None,
        max_len: int = 250,
        tokenizer: Callable[[str], list[str]] = tokenize,
        truncation: str = "post",
    ) -> None:
        if truncation.lower() not in VALID_TRUNCATIONS:
            raise ValueError(f"truncation phải thuộc {VALID_TRUNCATIONS}, nhận được {truncation!r}")
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.uids = list(uids) if uids is not None else [f"sample_{i}" for i in range(len(self.texts))]
        self.ratings = list(ratings) if ratings is not None else [0] * len(self.texts)
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.truncation = truncation.lower()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.texts[idx]
        tokens = self.tokenizer(text)
        n_tokens = len(tokens)

        was_truncated = n_tokens > self.max_len
        if was_truncated:
            tokens = truncate_tokens(tokens, self.max_len, self.truncation)

        if self.vocab is not None:
            input_ids = self.vocab.encode(tokens)
            n_unk = sum(1 for tid in input_ids if tid == self.vocab.unk_idx)
        else:
            input_ids = []
            n_unk = 0

        # Đảm bảo độ dài thật tối thiểu là 1 để pack_padded_sequence không bị lỗi
        length = max(1, len(input_ids))
        if len(input_ids) == 0:
            input_ids = [self.vocab.unk_idx if self.vocab else UNK_IDX]

        item: dict[str, Any] = {
            "uid": self.uids[idx],
            "text": text,
            "input_ids": input_ids,
            "length": length,
            "n_tokens": n_tokens,
            "n_unk": n_unk,
            "was_truncated": was_truncated,
            "rating": int(self.ratings[idx]),
        }

        if self.labels is not None:
            item["label"] = float(self.labels[idx])

        return item


def collate_imdb_batch(
    batch: list[dict[str, Any]],
    pad_idx: int = PAD_IDX,
    fixed_len: int | None = None,
) -> dict[str, Any]:
    """Hàm gộp batch, đệm động đến độ dài lớn nhất trong batch (hoặc fixed_len)."""
    batch_size = len(batch)
    lengths = [item["length"] for item in batch]
    max_len = fixed_len if fixed_len is not None else max(lengths)

    padded_inputs = torch.full((batch_size, max_len), fill_value=pad_idx, dtype=torch.long)
    actual_lengths: list[int] = []
    for i, item in enumerate(batch):
        seq = item["input_ids"][:max_len]
        actual_lengths.append(max(1, len(seq)))
        padded_inputs[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    lengths_tensor = torch.tensor(actual_lengths, dtype=torch.long)

    res: dict[str, Any] = {
        "input_ids": padded_inputs,
        "lengths": lengths_tensor,
        "uids": [item["uid"] for item in batch],
        "texts": [item["text"] for item in batch],
        "ratings": [item["rating"] for item in batch],
        "n_tokens": [item["n_tokens"] for item in batch],
        "n_unk": [item["n_unk"] for item in batch],
        "was_truncated": [item["was_truncated"] for item in batch],
    }

    if "label" in batch[0]:
        res["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    return res


class CollateBatch:
    """Callable class để tránh lỗi pickle lambda trên Windows."""

    def __init__(self, pad_idx: int = PAD_IDX, fixed_len: int | None = None) -> None:
        self.pad_idx = pad_idx
        self.fixed_len = fixed_len

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_imdb_batch(batch, pad_idx=self.pad_idx, fixed_len=self.fixed_len)


def create_dataloader(
    dataset: IMDbDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pad_idx: int = PAD_IDX,
    fixed_len: int | None = None,
) -> DataLoader:
    """Tạo DataLoader tương thích đa luồng và Windows."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=CollateBatch(pad_idx=pad_idx, fixed_len=fixed_len),
    )


def load_imdb_split_data(
    parquet_path: str | Path | None = None,
    split: str = "train",
) -> pd.DataFrame:
    """Đọc dữ liệu theo split từ file processed parquet."""
    p = Path(parquet_path) if parquet_path is not None else (data_dir("processed") / "reviews.parquet")
    if not p.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu đã xử lý tại {p}. Chạy 'python -m src.data.prepare' trước (B1)."
        )
    df = pd.read_parquet(p)
    return df[df["split"] == split].reset_index(drop=True)
