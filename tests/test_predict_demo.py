"""Test cho Task B6 & hàm bàn giao predict(text): làm sạch HTML, occlusion CPU < 1s và Gradio app."""
import tempfile
import time
from pathlib import Path
import pytest

from src.data.vocab import Vocab
from src.demo.app import analyze_review, build_demo, load_demo_examples
from src.models.rnn import SentimentRNN
from src.predict import SentimentPredictor


def test_sentiment_predictor_cleans_html_and_runs_fast_occlusion():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        vocab = Vocab({"<pad>": 0, "<unk>": 1, "not": 2, "bad": 3, "great": 4, "film": 5, "!": 6})
        vocab_file = tmp / "vocab.json"
        vocab.save(vocab_file)

        model = SentimentRNN(vocab_size=len(vocab), embed_dim=16, hidden_dim=16, rnn_type="lstm")
        ckpt_file = tmp / "best.pt"
        model.save_checkpoint(ckpt_file, config={"data": {"max_len": 20, "truncation": "head_tail"}})

        predictor = SentimentPredictor(checkpoint_path=ckpt_file, vocab_path=vocab_file, device="cpu")
        # Đảm bảo <br /> được loại bỏ bởi clean_text trước khi tokenize
        enc = predictor._encode_one("Great film!<br /><br />Not bad!")
        assert "<" not in enc["tokens"] and "br" not in enc["tokens"]

        label, prob = predictor.predict("Great film! Not bad!")
        assert label in (0, 1)
        assert 0.0 <= prob <= 1.0

        t0 = time.perf_counter()
        probs_dict, html_out, speed_str = analyze_review("Not a bad film at all!<br />Great!", predictor=predictor)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"Demo CPU vượt quá 1s ({elapsed:.2f}s)"
        assert len(probs_dict) == 2
        assert "<span" in html_out
        assert "ms" in speed_str


def test_load_demo_examples():
    examples = load_demo_examples()
    assert len(examples) >= 2


def test_build_gradio_blocks(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    gradio = pytest.importorskip("gradio")
    assert gradio is not None
    demo = build_demo()
    assert demo is not None
