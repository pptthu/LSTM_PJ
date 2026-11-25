"""Dự đoán cảm xúc cho một hoặc nhiều câu văn bản (Inference API / CLI).

Bàn giao Đại → Bảo → Thư (docs/TASKS.md §4, docs/phases/DEMO_REPORT.md):
- Hàm cấp module `predict(text: str) -> tuple[int, float]` trả về (nhãn, xác suất tích cực).
- Đi qua cùng hàm `clean_text`, `tokenize`, `truncate_tokens` và `vocab` như lúc train.
- Hỗ trợ giải thích nhanh theo batch (occlusion trên CPU < 1 giây/review) cho demo.
- CLI: `python -m src.predict "Not bad at all!"`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any
import torch

from src.data.clean import clean_text
from src.data.dataset import collate_imdb_batch, truncate_tokens
from src.data.tokenizer import tokenize
from src.data.vocab import Vocab
from src.models.rnn import SentimentRNN
from src.utils.paths import data_dir, experiment_dir


def _find_best_checkpoint(ckpt_root: Path | None = None) -> tuple[Path, Path]:
    """Tự động chọn checkpoint `.pt` có `val_acc` cao nhất trong `experiments/checkpoints/`."""
    root = ckpt_root if ckpt_root is not None else experiment_dir("checkpoints")
    pts = sorted(root.glob("*/seed_*/best.pt"))
    if not pts:
        raise FileNotFoundError(f"Chưa tìm thấy checkpoint 'best.pt' nào trong {root}")

    best_pt = pts[0]
    best_acc = -1.0
    res_root = experiment_dir("results")

    for pt in pts:
        exp_name = pt.parents[1].name
        seed_name = pt.parent.name
        m_file = res_root / exp_name / seed_name / "metrics.json"
        acc = -1.0
        if m_file.exists():
            try:
                acc = float(json.loads(m_file.read_text(encoding="utf-8")).get("best_val_acc", -1.0))
            except Exception:
                acc = -1.0
        if acc < 0:
            try:
                _, ckpt = SentimentRNN.load_checkpoint(pt, device="cpu")
                acc = float(ckpt.get("val_acc", -1.0))
            except Exception:
                acc = -1.0
        if acc > best_acc:
            best_acc = acc
            best_pt = pt

    exp_dir = best_pt.parents[1]
    vocab_path = exp_dir / "vocab.json"
    if not vocab_path.exists():
        vocab_path = data_dir("processed") / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Không tìm thấy vocab.json cho checkpoint {best_pt}")
    return best_pt, vocab_path


class SentimentPredictor:
    """Class tải checkpoint SentimentRNN và thực hiện suy luận + occlusion nhanh trên CPU/GPU."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        if checkpoint_path is None or vocab_path is None:
            auto_ckpt, auto_vocab = _find_best_checkpoint()
            checkpoint_path = checkpoint_path or auto_ckpt
            vocab_path = vocab_path or auto_vocab

        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.vocab = Vocab.load(self.vocab_path)
        self.model, self.ckpt = SentimentRNN.load_checkpoint(self.checkpoint_path, device=self.device)
        self.model.eval()

        data_cfg = self.ckpt.get("config", {}).get("data", {})
        self.max_len: int = int(data_cfg.get("max_len", 250))
        self.truncation: str = str(data_cfg.get("truncation", "post"))

    def _encode_one(self, text: str, uid: str = "pred_0") -> dict[str, Any]:
        cleaned = clean_text(text)
        tokens_all = tokenize(cleaned)
        n_tokens = len(tokens_all)
        was_truncated = n_tokens > self.max_len
        tokens = truncate_tokens(tokens_all, self.max_len, self.truncation) if was_truncated else tokens_all
        input_ids = self.vocab.encode(tokens)
        n_unk = sum(1 for tid in input_ids if tid == self.vocab.unk_idx)
        length = max(1, len(input_ids))
        if not input_ids:
            input_ids = [self.vocab.unk_idx]
        return {
            "uid": uid,
            "text": text,
            "tokens": tokens,
            "input_ids": input_ids,
            "length": length,
            "rating": 0,
            "n_tokens": n_tokens,
            "n_unk": n_unk,
            "was_truncated": was_truncated,
        }

    def predict(self, text: str) -> tuple[int, float]:
        """Dự đoán cho một chuỗi văn bản -> (label: 0|1, prob_positive: float)."""
        probs = self.predict_batch([text])
        prob = float(probs[0])
        label = 1 if prob >= 0.5 else 0
        return label, prob

    def predict_batch(self, texts: list[str]) -> list[float]:
        """Dự đoán xác suất lớp Positive cho một danh sách văn bản."""
        if not texts:
            return []
        items = [self._encode_one(t, uid=f"pred_{i}") for i, t in enumerate(texts)]
        batch = collate_imdb_batch(items, pad_idx=self.vocab.pad_idx)
        input_ids = batch["input_ids"].to(self.device)
        lengths = batch["lengths"]

        with torch.no_grad():
            probs = self.model.predict_proba(input_ids, lengths).cpu().tolist()

        if isinstance(probs, float):
            return [probs]
        return [float(p) for p in probs]

    def explain_occlusion(self, text: str, max_tokens: int = 120) -> dict[str, Any]:
        """Tính độ ảnh hưởng của từng token bằng occlusion theo batch (nhanh < 0.2s trên CPU).

        score_i = base_prob - prob_without_token_i:
        - score_i > 0: token đẩy dự đoán về phía Positive (1).
        - score_i < 0: token đẩy dự đoán về phía Negative (0).
        """
        enc = self._encode_one(text)
        tokens: list[str] = enc["tokens"][:max_tokens]
        ids: list[int] = enc["input_ids"][: len(tokens)]
        if not tokens:
            return {"label": 0, "prob": 0.5, "tokens": [], "scores": []}

        variants: list[dict[str, Any]] = [
            {
                "uid": "base",
                "text": text,
                "input_ids": ids,
                "length": len(ids),
                "rating": 0,
                "n_tokens": len(ids),
                "n_unk": 0,
                "was_truncated": False,
            }
        ]
        for i in range(len(ids)):
            masked = ids[:i] + [self.vocab.unk_idx] + ids[i + 1 :]
            variants.append(
                {
                    "uid": f"occ_{i}",
                    "text": text,
                    "input_ids": masked,
                    "length": len(masked),
                    "rating": 0,
                    "n_tokens": len(masked),
                    "n_unk": 0,
                    "was_truncated": False,
                }
            )

        batch = collate_imdb_batch(variants, pad_idx=self.vocab.pad_idx)
        with torch.no_grad():
            probs = self.model.predict_proba(batch["input_ids"].to(self.device), batch["lengths"]).cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]

        base_prob = float(probs[0])
        scores = [round(base_prob - float(p), 6) for p in probs[1:]]
        return {
            "label": 1 if base_prob >= 0.5 else 0,
            "prob": base_prob,
            "tokens": tokens,
            "scores": scores,
        }


_DEFAULT_PREDICTOR: Any = None


def get_default_predictor(device: str | None = "cpu") -> SentimentPredictor:
    """Tải checkpoint tốt nhất có sẵn trong experiments/checkpoints."""
    return SentimentPredictor(device=device)


def predict(text: str) -> tuple[int, float]:
    """Hàm dự đoán dùng chung cho CLI, demo Gradio và `notebooks/presentation.ipynb`.

    Ưu tiên checkpoint SentimentRNN tốt nhất; nếu chưa có `.pt` trên máy thì tự động
    fallback sang `baseline_tfidf_lr.joblib` (nếu có).
    """
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        try:
            _DEFAULT_PREDICTOR = get_default_predictor(device="cpu")
        except FileNotFoundError:
            bl_ckpt = experiment_dir("checkpoints") / "baseline_tfidf_lr.joblib"
            if bl_ckpt.exists():
                import joblib

                bundle = joblib.load(bl_ckpt)
                vec = bundle["vectorizer"]
                clf = bundle.get("model") or bundle.get("classifier")

                def _bl_predict(raw_text: str) -> tuple[int, float]:
                    pr = float(clf.predict_proba(vec.transform([clean_text(raw_text)]))[0, 1])
                    return (1 if pr >= 0.5 else 0), pr

                _DEFAULT_PREDICTOR = _bl_predict
            else:
                raise

    if callable(_DEFAULT_PREDICTOR) and not isinstance(_DEFAULT_PREDICTOR, SentimentPredictor):
        return _DEFAULT_PREDICTOR(text)
    return _DEFAULT_PREDICTOR.predict(text)


def reset_default_predictor_cache() -> None:
    """Xoá cache predictor mặc định (hữu ích trong unit test)."""
    global _DEFAULT_PREDICTOR
    _DEFAULT_PREDICTOR = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Dự đoán cảm xúc cho câu review phim tiếng Anh.")
    parser.add_argument("text", type=str, nargs="?", default="Not a bad film at all! I really enjoyed the ending.", help="Câu review cần dự đoán")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Đường dẫn best.pt tuỳ chọn")
    parser.add_argument("--vocab", type=Path, default=None, help="Đường dẫn vocab.json tuỳ chọn")
    parser.add_argument("--device", type=str, default="cpu", help="Thiết bị ('cpu' hoặc 'cuda')")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.checkpoint and args.vocab:
        predictor = SentimentPredictor(checkpoint_path=args.checkpoint, vocab_path=args.vocab, device=args.device)
        label, prob = predictor.predict(args.text)
    else:
        label, prob = predict(args.text)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    sentiment = "Positive (1)" if label == 1 else "Negative (0)"
    print(f"Text      : {args.text}")
    print(f"Prediction: {sentiment} | P(pos) = {prob:.4f} | Latency = {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
