"""Test cho Task D1 (& D7 truncation): Tokenizer, Vocab, Dataset và DataLoader."""
import tempfile
from pathlib import Path
import torch
import pytest

from src.data.tokenizer import tokenize
from src.data.vocab import Vocab, PAD_IDX, UNK_IDX
from src.data.dataset import IMDbDataset, collate_imdb_batch, create_dataloader, truncate_tokens


def test_tokenizer_keeps_negation_and_punctuation():
    text = "I do not like it! It wasn't good? Can't watch again."
    tokens = tokenize(text)
    assert "not" in tokens
    assert "n't" in tokens
    assert "!" in tokens
    assert "?" in tokens
    assert all(t == t.lower() for t in tokens)
    assert "i" in tokens and "do" in tokens


def test_truncate_tokens_modes():
    toks = ["t0", "t1", "t2", "t3", "t4", "t5"]
    assert truncate_tokens(toks, 4, "post") == ["t0", "t1", "t2", "t3"]
    assert truncate_tokens(toks, 4, "pre") == ["t2", "t3", "t4", "t5"]
    assert truncate_tokens(toks, 4, "head_tail") == ["t0", "t1", "t4", "t5"]
    with pytest.raises(ValueError):
        truncate_tokens(toks, 4, "invalid_mode")


def test_vocab_build_and_serialization():
    corpus = [
        ["i", "love", "this", "movie", "!"],
        ["i", "hate", "this", "film", "!"],
        ["this", "movie", "was", "great"],
    ]
    vocab = Vocab.build_from_tokens(corpus, min_freq=1)

    assert vocab.pad_idx == PAD_IDX == 0
    assert vocab.unk_idx == UNK_IDX == 1
    assert len(vocab) > 2
    assert "this" in vocab

    tokens = ["this", "movie", "unknownword123", "!"]
    encoded = vocab.encode(tokens)
    assert encoded[0] == vocab["this"]
    assert encoded[2] == UNK_IDX

    decoded = vocab.decode(encoded)
    assert decoded[0] == "this"
    assert decoded[2] == "<unk>"

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "vocab.json"
        vocab.save(save_path)
        loaded = Vocab.load(save_path)
        assert len(loaded) == len(vocab)
        assert loaded.encode(tokens) == encoded


def test_imdb_dataset_and_dataloader():
    vocab = Vocab({"<pad>": 0, "<unk>": 1, "great": 2, "movie": 3, "bad": 4})
    texts = [
        "great movie",
        "bad bad bad",
        "this movie was really great",
        "",
    ]
    labels = [1, 0, 1, 0]

    dataset = IMDbDataset(
        texts=texts,
        labels=labels,
        vocab=vocab,
        max_len=4,
        truncation="head_tail",
    )

    assert len(dataset) == 4
    item0 = dataset[0]
    assert item0["length"] == 2
    assert item0["label"] == 1.0
    assert not item0["was_truncated"]

    item2 = dataset[2]
    assert item2["n_tokens"] == 5
    assert item2["was_truncated"]
    assert item2["length"] == 4

    # Chuỗi rỗng vẫn có length=1 (<unk>) để pack_padded_sequence không lỗi
    item3 = dataset[3]
    assert item3["length"] == 1
    assert item3["input_ids"] == [UNK_IDX]

    loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == torch.Size([2, 3])
    assert batch["lengths"].tolist() == [2, 3]
    assert batch["labels"].shape == torch.Size([2])
