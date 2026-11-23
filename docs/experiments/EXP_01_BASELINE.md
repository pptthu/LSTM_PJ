# EXP-01 Baseline Majority + TF-IDF + LR

- **Motivation/Background**: Cần mốc sàn (majority) và mốc cổ điển mạnh (TF-IDF + LR) để biết LSTM có thật sự tốt hơn hay không.
- **Purpose**: Ghi cấu hình, kết quả val và cách dựng lại hai baseline M0a/M0b.
- **Overview Pipeline**: Chuẩn bị dữ liệu (B1) → M0a → M0b quét `C` trên val → lưu kết quả.
- **Detailed Plan**: Header; Objective; Setup; Results; Analysis; Reproduce; Links.
- **References**: [../phases/BASELINE.md](../phases/BASELINE.md), [../DATA.md](../DATA.md) §3, `src/models/baseline.py`.
- **Created**: 2026-11-23
- **Last Updated**: 2026-11-23

---

## Header

- **Title:** Baseline Majority + TF-IDF + Logistic Regression
- **Date created:** 2026-11-23
- **Last updated:** 2026-11-23
- **Description:** Mốc sàn (majority) và mốc cổ điển (TF-IDF 1–2 gram + LR), chọn `C` trên val.
- **Status:** Done (val); test để phase EVAL (D8) chạy một lần
- **Experiment ID:** EXP-01

## Objective

Đo hai mốc trên **val**: (1) M0a luôn đoán lớp nhiều nhất phải ≈ 50% accuracy; (2) M0b TF-IDF + LR phải rõ ràng tốt hơn majority và tạo mốc để LSTM vượt. Không tối ưu quá mức: chỉ chọn `C` trên val.

## Single variable changed / held constant

- **Changed:** mô hình (majority → TF-IDF + LR; và `C` trong M0b).
- **Held constant:** dữ liệu `data/processed/reviews.parquet` (train 22.304 / val 2.473), cột `text_clean`, split Lab 3, metric, ngưỡng 0,5.

## Setup

- **Data / split:** B1 (`src/data/prepare.py`), train 22.304 review, val 2.473 review. **Test không đọc** trong thí nghiệm này.
- **M0a:** lớp nhiều nhất trong train (= 1, positive 50,25% train).
- **M0b:** `TfidfVectorizer(ngram_range=(1,2), min_df=2)` **fit trên train**, `LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)`; val chỉ `transform`. Quét `C ∈ {0,01; 0,1; 1; 10}`; chọn theo val accuracy (hoà → macro-F1 → `C` nhỏ hơn). Đã thử thêm C = 20/30/50/100: accuracy **giảm** (0,8969 / 0,8961 / 0,8957 / 0,8953) nên giữ lưới ban đầu.
- **Đặc trưng:** 398.264 (unigram + bigram, `min_df=2`). Số tham số LR = 398.265.

## Results

| Mô hình | Val acc | Macro-F1 | Weighted-F1 | ROC-AUC | Precision neg/pos | Recall neg/pos |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| M0a Majority (`label=1`) | 0,4994 | 0,3331 | 0,3327 | 0,5000 | 0,000 / 0,499 | 0,000 / 1,000 |
| M0b TF-IDF + LR, `C=10` | **0,8985** | **0,8985** | 0,8985 | **0,9623** | 0,912 / 0,886 | 0,883 / 0,914 |

Quét `C` (`experiments/results/baseline/c_grid.csv`):

| C | Val acc | Macro-F1 | ROC-AUC | Train (s) | Inference (ms/2.473 review) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,01 | 0,7820 | 0,7810 | 0,8747 | 0,9 | 1,2 |
| 0,1 | 0,8346 | 0,8344 | 0,9202 | 1,0 | 1,0 |
| 1 | 0,8799 | 0,8798 | 0,9544 | 2,4 | 0,9 |
| **10 (chọn)** | **0,8985** | **0,8985** | **0,9623** | 4,8 | 1,3 |

Confusion matrix M0b trên val (hàng = nhãn thật, cột = dự đoán; `[[TN, FP], [FN, TP]]`):

```
[[1093,  145],
 [ 106, 1129]]
```

Thời gian: vectorize 9,5 s; train 4,8 s; inference 0,0022 ms/review (CPU). Tổng lần chạy ~24 s.

## Analysis / interpretation

- Majority đúng như kỳ vọng ≈ 50% nhưng precision lớp âm = 0 (không bao giờ đoán âm) → macro-F1 chỉ 0,333. Đây là mốc sàn thật sự.
- TF-IDF + LR đạt 0,8985 val accuracy — mốc mạnh cho LSTM vượt; khoảng cách majority → M0b là +39,9 điểm.
- Accuracy tăng đơn điệu tới `C=10` rồi giảm khi `C` lớn hơn → `C=10` là điểm tốt nhất trên val, không phải do chọn cận trên của lưới gốc.
- Hai lớp gần cân bằng: precision/recall âm–dương chênh < 3 điểm; macro-F1 ≈ accuracy nên không có lớp nào bị bỏ rơi.
- Sai số còn lại chủ yếu ở review vừa khen vừa chê / phủ định phức tạp — phân tích lỗi ở T2 (Thư) sẽ đối chiếu với `predictions_val.csv`.
- **Chưa** có số test: baseline chỉ chạy test ở D8, cùng lúc với LSTM, để so McNemar.

## Reproduce

```bash
python -m src.data.prepare      # nếu chưa có data/processed/reviews.parquet
python -m src.models.baseline   # ~25 s trên CPU, ghi experiments/results/baseline/
python -m pytest                # 23 passed trên nhánh feat/baseline
```

Đầu ra (`experiments/results/baseline/`): `metrics.json` (M0a + M0b + lưới C), `config.json` (tham số + phiên bản sklearn), `c_grid.csv`, `predictions_val.csv` (`uid, text, label, rating, prob, pred` — dùng cho McNemar với `predictions_val.csv` của LSTM). Checkpoint `experiments/checkpoints/baseline_tfidf_lr.joblib` (không commit) chứa `{vectorizer, model, config}`; load lại bằng `joblib.load` cho cùng dự đoán (đã kiểm).

## Links

- Code: [`src/models/baseline.py`](../../src/models/baseline.py)
- Kết quả: [`experiments/results/baseline/`](../../experiments/results/baseline/)
- Status: [../progress/BASELINE_STATUS.md](../progress/BASELINE_STATUS.md)
