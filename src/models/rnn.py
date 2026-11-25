"""Mô hình phân loại cảm xúc SentimentRNN (LSTM, GRU, BiLSTM).

Theo đặc tả trong docs/MODELS.md §2 và docs/phases/MODEL.md:
- Embedding (100d, padding_idx=0)
- pack_padded_sequence với độ dài thật
- nn.LSTM hoặc nn.GRU (128d, 1 hoặc 2 chiều)
- Trích xuất hidden state cuối (BiLSTM nối h_n[-2] và h_n[-1] -> 256d)
- nn.Dropout -> nn.Linear -> logit
- Hỗ trợ tuỳ chọn pooling='attention' cho phần mở rộng E1.
- Hỗ trợ nạp ma trận embedding GloVe và đóng băng (freeze).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionPooling(nn.Module):
    """Attention pooling layer tính trọng số cho từng vị trí chuỗi (Mở rộng E1)."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1),
        )

    def forward(
        self, rnn_out: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        rnn_out: [B, L, H]
        lengths: [B]
        Returns: pooled: [B, H], weights: [B, L]
        """
        batch_size, max_len, _ = rnn_out.size()
        scores = self.proj(rnn_out).squeeze(-1)  # [B, L]

        device = rnn_out.device
        idx = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, max_len)
        mask = idx < lengths.to(device).unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=-1)  # [B, L]
        pooled = torch.bmm(weights.unsqueeze(1), rnn_out).squeeze(1)  # [B, H]
        return pooled, weights


class SentimentRNN(nn.Module):
    """Mô hình phân loại cảm xúc tổng quát hóa cho LSTM, GRU, BiLSTM và GloVe."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        num_layers: int = 1,
        dropout: float = 0.5,
        padding_idx: int = 0,
        pretrained_embeddings: torch.Tensor | np.ndarray | None = None,
        freeze: bool = False,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.padding_idx = padding_idx
        self.freeze = freeze
        self.pooling = pooling.lower()

        # 1. Embedding layer
        if pretrained_embeddings is not None:
            if isinstance(pretrained_embeddings, np.ndarray):
                tensor_emb = torch.from_numpy(pretrained_embeddings).float()
            else:
                tensor_emb = pretrained_embeddings.float()
            self.embedding = nn.Embedding.from_pretrained(
                tensor_emb,
                freeze=freeze,
                padding_idx=padding_idx,
            )
            self.vocab_size, self.embed_dim = int(tensor_emb.shape[0]), int(tensor_emb.shape[1])
        else:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=padding_idx,
            )
            self.embedding.weight.requires_grad = not freeze

        # 2. Recurrent layer
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Không hỗ trợ rnn_type={rnn_type!r}. Chỉ dùng 'lstm' hoặc 'gru'.")

        self.num_directions = 2 if bidirectional else 1
        self.rnn_out_dim = hidden_dim * self.num_directions

        # 3. Pooling & Classifier
        if self.pooling == "attention":
            self.attn = AttentionPooling(self.rnn_out_dim)
        elif self.pooling != "last":
            raise ValueError(f"Không hỗ trợ pooling={pooling!r}. Chỉ dùng 'last' hoặc 'attention'.")

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.rnn_out_dim, 1)

    def forward_with_attention(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass trả về (logits [B], attention_weights [B, L] hoặc None)."""
        emb = self.embedding(input_ids)  # [B, L, E]

        cpu_lengths = lengths.clamp(min=1).to(dtype=torch.int64, device="cpu")
        packed_emb = pack_padded_sequence(
            emb,
            cpu_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        if self.rnn_type == "lstm":
            packed_out, (h_n, _) = self.rnn(packed_emb)
        else:
            packed_out, h_n = self.rnn(packed_emb)

        attn_weights: torch.Tensor | None = None
        if self.pooling == "attention":
            rnn_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=input_ids.size(1)
            )
            feat, attn_weights = self.attn(rnn_out, lengths)
        else:
            if self.bidirectional:
                feat = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                feat = h_n[-1]

        feat_drop = self.dropout(feat)
        logits = self.fc(feat_drop)  # [B, 1]
        return logits.squeeze(-1), attn_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass trả về logits [B]."""
        logits, _ = self.forward_with_attention(input_ids, lengths)
        return logits

    def predict_proba(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Trả về xác suất [B] sau khi qua Sigmoid."""
        logits = self.forward(input_ids, lengths)
        return torch.sigmoid(logits)

    def count_parameters(self) -> dict[str, Any]:
        """Đếm tổng số tham số, tham số huấn luyện được và trạng thái embedding."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "embedding_trainable": bool(self.embedding.weight.requires_grad),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        config: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Lưu checkpoint bao gồm state_dict và cấu hình mô hình."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_data: dict[str, Any] = {
            "state_dict": self.state_dict(),
            "model_kwargs": {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "rnn_type": self.rnn_type,
                "bidirectional": self.bidirectional,
                "num_layers": self.num_layers,
                "dropout": self.dropout_rate,
                "padding_idx": self.padding_idx,
                "freeze": self.freeze,
                "pooling": self.pooling,
            },
            "config": config or {},
        }
        if extra:
            checkpoint_data.update(extra)
        torch.save(checkpoint_data, p)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
    ) -> tuple[SentimentRNN, dict[str, Any]]:
        """Tải mô hình từ checkpoint (hỗ trợ cả PyTorch >= 2.6)."""
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        model = cls(**ckpt["model_kwargs"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model, ckpt
