"""Test cho Task D2: SentimentRNN (LSTM, GRU, BiLSTM, Attention, Frozen Embeddings, Overfit 1 batch, Save/Load)."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.models.rnn import SentimentRNN


def test_model_shapes():
    vocab_size = 100
    batch_size = 4
    seq_len = 10

    lstm = SentimentRNN(vocab_size=vocab_size, embed_dim=16, hidden_dim=32, rnn_type="lstm", bidirectional=False)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.tensor([10, 8, 5, 2])
    logits = lstm(x, lengths)
    assert logits.shape == torch.Size([batch_size])
    probs = lstm.predict_proba(x, lengths)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    gru = SentimentRNN(vocab_size=vocab_size, embed_dim=16, hidden_dim=32, rnn_type="gru", bidirectional=False)
    assert gru(x, lengths).shape == torch.Size([batch_size])

    bilstm = SentimentRNN(vocab_size=vocab_size, embed_dim=16, hidden_dim=32, rnn_type="lstm", bidirectional=True)
    assert bilstm(x, lengths).shape == torch.Size([batch_size])

    attn_model = SentimentRNN(vocab_size=vocab_size, embed_dim=16, hidden_dim=32, rnn_type="lstm", bidirectional=True, pooling="attention")
    logits_attn, weights = attn_model.forward_with_attention(x, lengths)
    assert logits_attn.shape == torch.Size([batch_size])
    assert weights is not None and weights.shape == torch.Size([batch_size, seq_len])


def test_frozen_embeddings_parameters():
    vocab_size = 100
    embed_dim = 16
    weights = np.random.randn(vocab_size, embed_dim).astype(np.float32)

    m_ft = SentimentRNN(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=32, pretrained_embeddings=weights, freeze=False)
    counts_ft = m_ft.count_parameters()

    m_frozen = SentimentRNN(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=32, pretrained_embeddings=weights, freeze=True)
    counts_frozen = m_frozen.count_parameters()

    emb_params = vocab_size * embed_dim
    assert counts_frozen["total"] == counts_ft["total"]
    assert counts_frozen["trainable"] == counts_ft["trainable"] - emb_params
    assert counts_ft["embedding_trainable"] is True
    assert counts_frozen["embedding_trainable"] is False


def test_save_load_checkpoint():
    vocab_size = 50
    model = SentimentRNN(vocab_size=vocab_size, embed_dim=16, hidden_dim=32, rnn_type="lstm", bidirectional=True)
    x = torch.randint(0, vocab_size, (2, 8))
    lengths = torch.tensor([8, 6])

    model.eval()
    with torch.no_grad():
        out1 = model(x, lengths)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model.pt"
        model.save_checkpoint(ckpt_path, config={"lr": 0.001})

        loaded_model, ckpt = SentimentRNN.load_checkpoint(ckpt_path)
        loaded_model.eval()
        with torch.no_grad():
            out2 = loaded_model(x, lengths)

        assert torch.allclose(out1, out2, atol=1e-5)
        assert ckpt["config"]["lr"] == 0.001


def test_overfit_single_batch():
    """Verification gate: Overfit một batch 32 mẫu tới loss gần 0."""
    torch.manual_seed(42)
    vocab_size = 200
    batch_size = 32
    seq_len = 20

    model = SentimentRNN(
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_dim=64,
        rnn_type="lstm",
        bidirectional=False,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    y = torch.randint(0, 2, (batch_size,), dtype=torch.float32)

    model.train()
    final_loss = 1.0
    for _ in range(80):
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < 0.05, f"Loss sau khi overfit 1 batch là {final_loss:.4f}, kỳ vọng < 0.05"
