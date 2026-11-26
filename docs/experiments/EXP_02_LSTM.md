# EXP-02 LSTM (mô hình đề bài)

- **Motivation/Background**: Đề bài yêu cầu Embedding + LSTM + Linear; đây là mô hình RNN đầu tiên, làm mốc cho mọi biến thể sau.
- **Purpose**: Ghi cấu hình, kết quả 3 seed và cách chạy lại M1.
- **Overview Pipeline**: Tokenize → vocab 25.000 từ (chỉ train) → Embedding 100d → LSTM 1 chiều 128 → hidden cuối → Linear → sigmoid.
- **Detailed Plan**: Header; Objective; Single variable; Setup; Results; Analysis; Reproduce; Links.
- **References**: `configs/lstm.yaml`, `src/models/rnn.py`, `src/train.py`, `experiments/results/lstm/`.
- **Created**: 2026-11-26
- **Last Updated**: 2026-11-26

---

## Header

- **Title:** LSTM 1 chiều, embedding học từ đầu
- **Date created:** 2026-11-26
- **Last updated:** 2026-11-26
- **Description:** Mô hình bắt buộc của đề bài, 3 seed, `max_len` = 400.
- **Status:** Done
- **Experiment ID:** EXP-02 (M1)

## Objective

Đo LSTM theo đúng đề bài trên cùng split với baseline. Mục tiêu đăng ký **M1: test accuracy ≥ 85%**.

## Single variable changed / held constant

- **Changed:** mô hình (TF-IDF + LR → Embedding + LSTM + Linear).
- **Held constant:** dữ liệu `data/processed/reviews.parquet` (train 22.304 / val 2.473 / test 25.000), tokenizer, vocab, `max_len` = 400 (chốt ở pilot D5, xem [EXP_07](EXP_07_MAXLEN.md)), seed 42 / 123 / 2025.

## Setup

- **Mô hình:** Embedding 100d (khởi tạo ngẫu nhiên, train được) → LSTM 1 chiều, hidden 128, 1 lớp → hidden state cuối (`pack_padded_sequence`, không đọc padding) → dropout 0,5 → Linear(128 → 1).
- **Dữ liệu:** vocab 25.000 từ, `min_freq` = 2, chỉ xây từ train; cắt đuôi (`post`) ở 400 token.
- **Huấn luyện:** Adam, lr 1e-3, weight decay 1e-5, batch 64, gradient clipping 5,0, tối đa 20 epoch, early stopping theo **val accuracy** với patience 3, giữ checkpoint tốt nhất.
- **Phần cứng:** Kaggle Tesla T4, 26/11/2026, code `e1f38ec`.

## Results

| | Val acc | Test acc | Test macro-F1 | Test ROC-AUC | Tham số | Thời gian train/seed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| M1 LSTM | 86,09 ± 0,33% | **86,48 ± 0,32%** | 0,8648 ± 0,0032 | 0,9321 | 2.617.889 | 162,7 s |
| M0b TF-IDF + LR | 89,85% | 89,94% | 0,8994 | 0,9630 | 398.265 | 5,5 s |

Theo từng seed:

| Seed | Epoch tốt nhất / dừng | Val acc | Train acc tại epoch tốt nhất | Test acc |
| :--- | :--- | :--- | :--- | :--- |
| 42 | 7 / 10 | 0,8601 | 0,9071 | 0,8611 |
| 123 | 11 / 14 | 0,8645 | 0,9605 | 0,8662 |
| 2025 | 11 / 14 | 0,8581 | 0,9582 | 0,8671 |

Nguồn: `experiments/results/lstm/summary.json`, `test_summary.json`, `seed_*/history.csv`. Learning curve: `experiments/plots/training/lstm_curves.png`.

## Analysis / interpretation

- **M1 đạt** (86,48% ≥ 85%). Lần chạy 1 với `max_len` = 250 chỉ được 84,92%, không đạt — `max_len` quyết định việc đạt hay trượt.
- LSTM **kém TF-IDF + LR khoảng 3,5 điểm**. TF-IDF đọc toàn bộ review; LSTM chỉ thấy 400 token và phải tự học nghĩa từ với 22 nghìn review.
- **Overfit rõ:** train acc tại epoch tốt nhất ~94%, cao hơn val ~8 điểm; sau đó val loss tăng (0,44 ± 0,07, độ lệch giữa seed lớn nhất trong 6 mô hình).
- **Hội tụ chậm và không đều:** seed 42 tốt nhất ở epoch 7, hai seed còn lại ở epoch 11. LSTM 1 chiều với hidden cuối phải "nhớ" thông tin qua 400 bước.

## Reproduce

```bash
python -m src.train --config configs/lstm.yaml --seed 42      # lặp lại với 123, 2025
python -m src.evaluate --exp lstm --split val
```

## Links

- So sánh tiếp: [EXP_03 GRU](EXP_03_GRU.md), [EXP_04 BiLSTM](EXP_04_BILSTM.md). Bảng tổng: [README.md](README.md) §3.
