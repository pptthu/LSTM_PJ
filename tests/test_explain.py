"""Test T3: occlusion, deletion test, tên riêng — dùng predictor giả để không cần checkpoint."""
import numpy as np
import torch

from src.eval import explain


class FakeVocab:
    unk_idx, pad_idx = 1, 0


class FakeModel:
    """p_pos tăng theo số token 'good' (id 7), giảm theo 'bad' (id 8)."""

    def predict_proba(self, ids, lengths):
        good = (ids == 7).sum(1).float()
        bad = (ids == 8).sum(1).float()
        return torch.sigmoid(good - bad)


class FakePred:
    vocab, model, device = FakeVocab(), FakeModel(), torch.device("cpu")
    words = {"good": 7, "bad": 8}

    def _encode_one(self, text):
        toks = text.lower().replace(".", "").split()
        return {"tokens": toks, "input_ids": [self.words.get(w, 5) for w in toks], "n_tokens": len(toks)}

    def predict_batch(self, texts):
        return explain.probs_for_ids(self, [self._encode_one(t)["input_ids"] for t in texts]).tolist()


def test_occlusion_scores_point_to_sentiment_words():
    o = explain.occlusion(FakePred(), "the film was good good and fine")
    assert o["label"] == 1
    assert o["tokens"][int(np.argmax(o["scores"]))] == "good"
    assert np.allclose([s for w, s in zip(o["tokens"], o["scores"]) if w not in ("good",)], 0)


def test_deletion_top_beats_random():
    texts = ["good good good plain plain plain plain plain plain plain plain plain plain"] * 5
    t = explain.deletion_test(FakePred(), texts, ks=(1, 3))
    top = t[t["how"] == "top_occlusion"].set_index("k")["mean_conf_drop"]
    rnd = t[t["how"] == "random"].set_index("k")["mean_conf_drop"]
    assert (top > rnd).all()


def test_proper_nouns_skip_sentence_start():
    assert explain.proper_nouns("The movie with Tom Hanks was fine. Then it ended, and Meryl sang.") == {"tom", "hanks", "meryl"}
