"""Giao diện Demo Gradio phân loại cảm xúc review phim IMDb (Task B6).

Đặc tả (docs/phases/DEMO_REPORT.md & src/demo/README.md):
- Nhập review -> trả về nhãn (Positive / Negative), xác suất từng lớp, tô màu từ quan trọng.
- Nạp sẵn các ví dụ đại diện từ `data/sample/sample_100.csv` và `data/sample/handwritten_30.csv`.
- Đo thời gian suy luận + giải thích trên CPU (< 1 giây / review).
- Gọi lại `src/predict.py` và ưu tiên `src/eval/explain.py` (nếu có).

Chạy từ gốc repo:
    python -m src.demo.app
"""
from __future__ import annotations

import html
import re
import time
from typing import Any
import pandas as pd

from src.data.clean import clean_text
from src.data.tokenizer import tokenize
from src.predict import SentimentPredictor, get_default_predictor, predict
from src.utils.paths import data_dir, experiment_dir


def _render_highlighted_tokens(tokens: list[str], scores: list[float]) -> str:
    """Render HTML tô màu các từ đóng góp mạnh vào Positive (xanh) / Negative (đỏ)."""
    if not tokens:
        return "<i>Chưa có văn bản đầu vào.</i>"

    max_abs = max((abs(s) for s in scores), default=0.0)
    spans: list[str] = []
    for tok, sc in zip(tokens, scores):
        safe_tok = html.escape(tok)
        if max_abs <= 1e-6 or abs(sc) < 0.005:
            spans.append(f"<span style='padding:2px 4px;margin:1px;display:inline-block;'>{safe_tok}</span>")
            continue

        alpha = min(0.75, 0.15 + 0.60 * (abs(sc) / max_abs))
        if sc > 0:
            bg = f"rgba(34, 197, 94, {alpha:.2f})"
            tip = f"+{sc:.3f} (hướng Positive)"
        else:
            bg = f"rgba(239, 68, 68, {alpha:.2f})"
            tip = f"{sc:.3f} (hướng Negative)"
        spans.append(
            f"<span title='{tip}' style='background:{bg};border-radius:4px;"
            f"padding:2px 5px;margin:1px;display:inline-block;'>{safe_tok}</span>"
        )

    legend = (
        "<div style='margin-bottom:8px;font-size:0.9em;color:#555;'>"
        "<span style='background:rgba(34,197,94,0.45);padding:2px 6px;border-radius:4px;'>Xanh: đẩy về Positive</span> "
        "<span style='background:rgba(239,68,68,0.45);padding:2px 6px;border-radius:4px;margin-left:8px;'>Đỏ: đẩy về Negative</span>"
        "</div>"
    )
    return legend + "<div style='line-height:1.9;font-size:1.02em;'>" + " ".join(spans) + "</div>"


def load_demo_examples(max_examples: int = 8) -> list[list[str]]:
    """Đọc ví dụ mẫu từ `handwritten_30.csv` và `sample_100.csv`."""
    examples: list[str] = []
    hw_path = data_dir("sample") / "handwritten_30.csv"
    s100_path = data_dir("sample") / "sample_100.csv"

    if hw_path.exists():
        hw = pd.read_csv(hw_path)
        for _, row in hw.groupby("case_type", sort=False).head(1).iterrows():
            examples.append(str(row["text"]))

    if s100_path.exists() and len(examples) < max_examples:
        s100 = pd.read_csv(s100_path)
        for _, row in s100.groupby("case_type", sort=False).head(1).iterrows():
            if len(examples) >= max_examples:
                break
            # review IMDb gốc chứa thẻ <br /> — bỏ khi hiển thị trên nút ví dụ
            examples.append(re.sub(r"(?:\s*<br\s*/?>\s*)+", " ", str(row["text"])).strip())

    if not examples:
        examples = [
            "Not a bad film at all! The ending surprised me and I would happily watch it again.",
            "The cast is wonderful, but the script gives them nothing to do and the last act is a mess.",
        ]
    return [[ex] for ex in examples[:max_examples]]


def analyze_review(
    text: str,
    predictor: SentimentPredictor | None = None,
) -> tuple[dict[str, float], str, str]:
    """Hàm xử lý một review trên CPU: trả về (xác suất 2 lớp, HTML tô màu từ, thông tin thời gian)."""
    if not text or not text.strip():
        return {"Positive (1)": 0.5, "Negative (0)": 0.5}, "<i>Vui lòng nhập nội dung review.</i>", "0.0 ms"

    t0 = time.perf_counter()
    model_source = "SentimentRNN (CPU)"

    # 1. Nếu truyền trực tiếp SentimentPredictor hoặc có checkpoint .pt trên máy
    active_predictor = predictor
    if active_predictor is None:
        try:
            active_predictor = get_default_predictor(device="cpu")
        except FileNotFoundError:
            active_predictor = None

    if active_predictor is not None:
        # Ưu tiên gọi src.eval.explain nếu Thư đã cung cấp (T3), ngược lại dùng occlusion tích hợp
        tokens: list[str]
        scores: list[float]
        try:
            from src.eval.explain import explain_text  # type: ignore

            exp_res = explain_text(text, predictor=active_predictor)
            prob_pos = float(exp_res["prob"])
            tokens = list(exp_res["tokens"])
            scores = list(exp_res["scores"])
        except Exception:
            occ = active_predictor.explain_occlusion(text)
            prob_pos = float(occ["prob"])
            tokens = occ["tokens"]
            scores = occ["scores"]
        model_source = f"SentimentRNN ({active_predictor.checkpoint_path.parent.parent.name}, CPU)"
    else:
        # Fallback sang baseline TF-IDF + LR nếu chưa chép checkpoint .pt về máy
        bl_ckpt = experiment_dir("checkpoints") / "baseline_tfidf_lr.joblib"
        if not bl_ckpt.exists():
            return (
                {"Positive (1)": 0.5, "Negative (0)": 0.5},
                "<b>Chưa tìm thấy checkpoint mô hình trong <code>experiments/checkpoints/</code>.</b>",
                "N/A",
            )
        _, prob_pos = predict(text)
        import joblib

        bundle = joblib.load(bl_ckpt)
        vec = bundle["vectorizer"]
        clf = bundle.get("model") or bundle.get("classifier")
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)[:120]
        vocab_map = vec.vocabulary_
        coefs = clf.coef_[0]
        scores = [float(coefs[vocab_map[t]]) * 0.05 if t in vocab_map else 0.0 for t in tokens]
        model_source = "Baseline TF-IDF + LR (CPU fallback)"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    probs_dict = {
        "Positive (Tích cực — 1)": round(prob_pos, 4),
        "Negative (Tiêu cực — 0)": round(1.0 - prob_pos, 4),
    }
    highlighted_html = _render_highlighted_tokens(tokens, scores)
    speed_badge = f"{elapsed_ms:.1f} ms trên CPU ({'< 1s ĐẠT M5' if elapsed_ms < 1000 else '> 1s'}) · Nguồn: {model_source}"
    return probs_dict, highlighted_html, speed_badge


def build_demo(predictor: SentimentPredictor | None = None):
    """Khởi tạo đối tượng `gr.Blocks` cho giao diện demo."""
    import gradio as gr

    with gr.Blocks(title="IMDb Sentiment Analysis — LSTM Demo") as demo:
        gr.Markdown(
            "# 🎬 Phân loại cảm xúc Review Phim IMDb\n"
            "Nhập đoạn đánh giá phim tiếng Anh để xem dự đoán cảm xúc, xác suất và các từ ảnh hưởng mạnh nhất."
        )
        with gr.Row():
            with gr.Column(scale=5):
                inp = gr.Textbox(
                    label="Nội dung Review (tiếng Anh)",
                    placeholder="Ví dụ: Not a bad film at all! The acting was great...",
                    lines=6,
                )
                btn = gr.Button("Dự đoán cảm xúc", variant="primary")
            with gr.Column(scale=5):
                out_label = gr.Label(num_top_classes=2, label="Dự đoán & Xác suất")
                out_speed = gr.Textbox(label="Thời gian suy luận (CPU)", interactive=False)

        out_html = gr.HTML(label="Giải thích mức độ ảnh hưởng từng từ (Occlusion)")

        gr.Examples(
            examples=load_demo_examples(),
            inputs=[inp],
            label="Ví dụ mẫu (từ handwritten_30.csv & sample_100.csv)",
        )

        def _handler(txt: str):
            return analyze_review(txt, predictor=predictor)

        btn.click(fn=_handler, inputs=[inp], outputs=[out_label, out_html, out_speed])
        inp.submit(fn=_handler, inputs=[inp], outputs=[out_label, out_html, out_speed])

    return demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chạy giao diện Demo Gradio (Task B6).")
    parser.add_argument("--port", type=int, default=7860, help="Cổng chạy server local")
    parser.add_argument("--share", action="store_true", help="Tạo public link Gradio")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
