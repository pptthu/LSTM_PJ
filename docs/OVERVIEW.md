# OVERVIEW.md — Lộ trình & Trạng thái dự án

- **Motivation/Background**: Cần một chỗ duy nhất để cả nhóm biết dự án đang ở phase nào, ai làm gì, checkpoint nào là tốt nhất.
- **Purpose**: Lộ trình sống của dự án. Cập nhật đầu mỗi buổi làm việc.
- **Overview Pipeline**: DATA_PREP → BASELINE → MODEL → TRAINING → EVAL → DEMO_REPORT (+ mở rộng nếu kịp).
- **Detailed Plan**: §1 Roadmap theo phase; §2 Phần mở rộng; §3 Kiến trúc; §4 Checkpoint; §5 Ràng buộc phần cứng.
- **References**: [PURPOSE.md](PURPOSE.md) §10, [phases/](phases/), [progress/](progress/).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T10:13:39+07:00

---

## 1. Roadmap theo phase

Hạn nộp: **thứ 4, 25/11/2026, trước 14:00**. Mốc theo [PURPOSE.md](PURPOSE.md) §10.

| # | Phase | Đặc tả | Tiến độ | Người chính | Hạn | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Chuẩn bị dữ liệu & split | [DATA_PREP](phases/DATA_PREP.md) | [status](progress/DATA_PREP_STATUS.md) | Bảo | T2 23/11 tối | Done (B1–B4) |
| 2 | Baseline Majority + TF-IDF + LR | [BASELINE](phases/BASELINE.md) | [status](progress/BASELINE_STATUS.md) | Bảo | T2 23/11 tối | Done (val; test ở D8) |
| 3 | Định nghĩa mô hình | [MODEL](phases/MODEL.md) | [status](progress/MODEL_STATUS.md) | Đại | T2 23/11 tối | In Progress (D1–D2 xong code+test; D3 chờ `.npy` thật) |
| 4 | Huấn luyện (3 seed) | [TRAINING](phases/TRAINING.md) | [status](progress/TRAINING_STATUS.md) | Đại | T3 24/11, mốc dừng 14:00 | In Progress (D4 xong code+test; chờ chạy Colab) |
| 5 | Đánh giá, phân tích lỗi, giải thích | [EVAL](phases/EVAL.md) | [status](progress/EVAL_STATUS.md) | Đại (test), Thư (phân tích lỗi) | T3 24/11 tối | In Progress (D8 xong code+test; chờ chạy Colab) |
| 6 | Demo, README, báo cáo & slide | [DEMO_REPORT](phases/DEMO_REPORT.md) | [status](progress/DEMO_REPORT_STATUS.md) | Bảo (demo), Thư (báo cáo) | T4 25/11 trước 14:00 | In Progress (B6, T4 xong code khung) |

## 2. Phần mở rộng (chia mức theo chi phí; Thư chốt 23/11: giữ hết, tốn thời gian quá thì để sau)

| ID | Việc | Mức | Thí nghiệm | Người | Khi nào | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| E3 | DistilBERT (mốc tham chiếu) — **không train lại**, trích kết quả Lab 3 (cùng IMDb, cùng split) | A | — | Thư (khi viết báo cáo) | T4 | To Do |
| E2 | `max_len` 200 / 400 / 600 + cách cắt (đầu / cuối / đầu + cuối), mô hình tốt nhất, 1 seed | A | EXP-07 | Đại | T3 chiều | To Do |
| — | Grid nhỏ M1 (hidden 128/256 × dropout 0,3/0,5, 1 seed) — chỉ khi pilot M1 < 85% val | A, có điều kiện | — | Đại | T3 sáng | To Do |
| E1 | Attention pooling trên BiLSTM (đối chiếu với occlusion, không dùng attention một mình để giải thích) | B | EXP-06 | Đại | T3 tối, nếu đạt mốc dừng 14:00 | To Do |
| E4 | Đánh giá chéo Rotten Tomatoes | C — để sau | EXP-09 | — | Ghi vào "Hướng phát triển" | Để sau |

## 2b. Các module (12)

Task chi tiết: [TASKS.md](TASKS.md). ⚠️ = phần nhóm trình bày trước bị thiếu, bắt buộc có section riêng trong notebook.

| # | Module | File | Người | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Data: tải, làm sạch, split | `src/data/download.py`, `clean.py`, `split.py` | Bảo | Đã có (chép Lab 3) |
| 2 | Prepare: loại trùng → `reviews.parquet` | `src/data/prepare.py` | Bảo | To Do |
| 3 | **EDA** ⚠️ | `src/data/eda.py` | Bảo | Hàm đã có; chạy lại sau loại trùng |
| 4 | Tokenizer, vocab, Dataset | `src/data/tokenizer.py`, `vocab.py`, `dataset.py` | Đại | To Do |
| 5 | Model: `SentimentRNN` + GloVe | `src/models/rnn.py`, `glove.py` | Đại | To Do |
| 6 | Baseline: Majority, TF-IDF + LR | `src/models/baseline.py` | Bảo | To Do |
| 7 | Train: vòng train, early stopping (val acc), checkpoint | `src/train.py`, `src/training/` | Đại | To Do |
| 8 | **Monitor training** ⚠️ | `src/training/monitor.py` | Đại | To Do |
| 9 | **Evaluation** ⚠️ | `src/evaluate.py`, `src/eval/metrics.py` | Đại | To Do |
| 10 | **Error analysis** + giải thích ⚠️ | `src/eval/error_analysis.py`, `explain.py` | Thư | To Do |
| 11 | Predict + demo Gradio | `src/predict.py`, `src/demo/app.py` | Bảo | To Do |
| 12 | Notebook trình bày (tiếng Việt) | `notebooks/presentation.ipynb` | Thư | To Do |

## 3. Kiến trúc

```
review text → clean → tokenize → ids [B, max_len] + lengths [B]
→ Embedding [B, L, 100] → pack → LSTM/GRU (uni/bi) → h_last [B, 128 or 256]
→ Dropout → Linear → logit [B, 1]
```

Chi tiết: [MODELS.md](MODELS.md).

## 4. Checkpoint

| Mô hình | Val acc (mean ± std) | Test acc (mean ± std) | Đường dẫn |
| :--- | :--- | :--- | :--- |
| TBD | — | — | `experiments/checkpoints/<exp>/` |

## 5. Ràng buộc phần cứng

- Máy cá nhân có thể không có GPU: chạy thử với tập con nhỏ, train thật trên Colab/Kaggle.
- Demo phải chạy trên CPU, < 1 giây/review.
- Ghi phần cứng, thời gian train và thời gian inference vào từng báo cáo thí nghiệm.
