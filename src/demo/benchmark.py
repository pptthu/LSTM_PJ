"""Đo thời gian demo trên CPU cho mục tiêu M5 (< 1 giây/review, gồm cả tô màu từ).

Chạy từ gốc repo:
    python -m src.demo.benchmark
    python -m src.demo.benchmark --checkpoint <best.pt> --vocab <vocab.json>

Đo đúng hàm `analyze_review` mà giao diện Gradio gọi, trên 100 review mẫu + 30 câu tự viết.
Mô hình nạp một lần trước khi đo (như khi demo đang mở). Kết quả: experiments/results/demo/latency.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.demo.app import analyze_review
from src.predict import SentimentPredictor, get_default_predictor
from src.utils.paths import data_dir, experiment_dir


def run(predictor: SentimentPredictor, out: Path, warmup: int = 3) -> dict:
    texts = (pd.read_csv(data_dir("sample") / "sample_100.csv")["text"].tolist()
             + pd.read_csv(data_dir("sample") / "handwritten_30.csv")["text"].tolist())
    for t in texts[:warmup]:
        analyze_review(t, predictor=predictor)
    ms, n_tok = [], []
    for t in texts:
        t0 = time.perf_counter()
        analyze_review(t, predictor=predictor)
        ms.append((time.perf_counter() - t0) * 1000)
        n_tok.append(predictor._encode_one(t)["n_tokens"])
    ms = np.array(ms)
    res = {
        "checkpoint": predictor.checkpoint_path.relative_to(predictor.checkpoint_path.parents[2]).as_posix(),
        "max_len": predictor.max_len, "n_reviews": len(texts),
        "ms_median": round(float(np.median(ms)), 1), "ms_p95": round(float(np.percentile(ms, 95)), 1),
        "ms_max": round(float(ms.max()), 1), "share_under_1s": round(float((ms < 1000).mean()), 4),
        "tokens_median": int(np.median(n_tok)), "tokens_max": int(max(n_tok)),
        "m5_pass": bool(ms.max() < 1000),
        "machine": {"cpu": platform.processor() or platform.machine(), "os": platform.platform(),
                    "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
        "note": "gồm dự đoán + occlusion (120 token đầu) + dựng HTML; mô hình đã nạp sẵn",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--vocab", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=experiment_dir("results", "demo") / "latency.json")
    a = ap.parse_args()
    pred = (SentimentPredictor(a.checkpoint, a.vocab, device="cpu") if a.checkpoint
            else get_default_predictor(device="cpu"))
    print(json.dumps(run(pred, a.out), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
