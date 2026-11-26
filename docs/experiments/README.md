# experiments/README.md — Chỉ mục thí nghiệm

- **Motivation/Background**: Mỗi thí nghiệm đổi đúng một biến so với thí nghiệm trước để biết thay đổi nào tạo ra khác biệt.
- **Purpose**: Bảng chỉ mục mọi thí nghiệm và bảng kết quả so sánh với baseline.
- **Overview Pipeline**: EXP_01 baseline → EXP_02 LSTM → EXP_03 GRU → EXP_04 BiLSTM → EXP_05 BiLSTM + GloVe → (mở rộng) EXP_06–09.
- **Detailed Plan**: §1 Quy tắc; §2 Chỉ mục; §3 Bảng kết quả.
- **References**: [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md), [../MODELS.md](../MODELS.md), [../PURPOSE.md](../PURPOSE.md) §6–§7.
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-26T09:00:00+07:00

---

## 1. Quy tắc

- Tên file: `EXP_<NN>_<TEN_NGAN>.md`, copy từ [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md).
- Giữ nguyên split, `max_len`, vocab, optimizer giữa các thí nghiệm, trừ biến đang thử.
- Mỗi cấu hình RNN chạy 3 seed (42, 123, 2025); báo mean ± std.
- Số val dùng để chọn; số test chỉ ghi sau phase EVAL.

## 2. Chỉ mục

| ID | Tiêu đề | Biến thay đổi so với trước | Người | Trạng thái | File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-01 | Baseline Majority + TF-IDF + LR | — | Bảo | Done | `EXP_01_BASELINE.md` |
| EXP-02 | LSTM | mô hình RNN đầu tiên | Đại (code), Thư (chạy) | Done | `EXP_02_LSTM.md` |
| EXP-03 | GRU | LSTM → GRU | Đại (code), Thư (chạy) | Done | `EXP_03_GRU.md` |
| EXP-04 | BiLSTM | 1 chiều → 2 chiều | Đại (code), Thư (chạy) | Done | `EXP_04_BILSTM.md` |
| EXP-05 | BiLSTM + GloVe | embedding ngẫu nhiên → GloVe (đóng băng / fine-tune) | Đại (code), Thư (chạy) | Done | `EXP_05_BILSTM_GLOVE.md` |
| EXP-06 | (Mở rộng) Attention pooling | hidden cuối → attention pooling | Đại (code), Thư (chạy) | Done | `EXP_06_ATTENTION.md` |
| EXP-07 | (Mở rộng) Ablation `max_len` | `max_len` 200 / 400 / 600 + cách cắt | Thư | Done | `EXP_07_MAXLEN.md` |
| EXP-08 | (Mở rộng) DistilBERT | RNN → Transformer fine-tune | — | Trích Lab 3, không train lại | `EXP_08_DISTILBERT.md` |
| EXP-09 | (Mở rộng) Rotten Tomatoes | đánh giá chéo, không train lại | TBD | To Do | `EXP_09_ROTTEN_TOMATOES.md` |

## 3. Bảng kết quả

Số lấy từ `experiments/results/comparison_val.csv` và `comparison_test.csv`. RNN: mean ± std (ddof = 1) của 3 seed (42, 123, 2025). Chạy trên Kaggle Tesla T4 ngày 26/11/2026, code `e1f38ec`, `max_len` = 400 (chốt ở D5). Mô hình chọn theo **val**: GRU (88,27%), test chỉ chạy một lần.

| Mô hình | Val acc | Test acc | Test macro-F1 | Test ROC-AUC | Tham số (train được) | Thời gian train/seed (s) | Inference (ms/review, GPU T4, theo batch) | Nguồn |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| M0a Majority | 49,94% | 50,00% | 0,3333 | 0,5000 | 0 | — | — | [EXP_01](EXP_01_BASELINE.md) |
| M0b TF-IDF + LR (`C=10`) | 89,85% | 89,94% | 0,8994 | 0,9630 | 398.265 | 5,5 | 0,3211 | [EXP_01](EXP_01_BASELINE.md) |
| M1 LSTM | 86,09 ± 0,33% | 86,48 ± 0,32% | 0,8648 ± 0,0032 | 0,9321 | 2.617.889 | 162,7 | 0,31 | `experiments/results/lstm/` |
| M2 GRU | 88,27 ± 0,45% | 88,16 ± 0,18% | 0,8816 ± 0,0018 | 0,9499 | 2.588.449 | 122,8 | 0,315 | `experiments/results/gru/` |
| M3 BiLSTM | 86,29 ± 0,93% | 86,53 ± 0,45% | 0,8652 ± 0,0045 | 0,9369 | 2.735.777 | 194,2 | 0,377 | `experiments/results/bilstm/` |
| M4a BiLSTM + GloVe (frozen) | 87,76 ± 0,31% | 87,87 ± 0,29% | 0,8787 ± 0,0028 | 0,9505 | 2.735.777 (235.777) | 227,4 | 0,379 | `experiments/results/bilstm_glove_frozen/` |
| M4b BiLSTM + GloVe (fine-tune) | 86,70 ± 0,65% | 86,72 ± 0,34% | 0,8670 ± 0,0035 | 0,9441 | 2.735.777 | 109,5 | 0,383 | `experiments/results/bilstm_glove_ft/` |
| E1 BiLSTM + attention pooling | 88,26 ± 0,58% | 88,28 ± 0,16% | 0,8828 ± 0,0016 | 0,9527 | 2.768.802 | 171,2 | 0,411 | `experiments/results/bilstm_attention/` |
| DistilBERT (tham chiếu Lab 3, không train lại) | — | 93,00–93,40% | — | — | 66.955.010 | — | — | Lab 3 |

**McNemar (GRU seed tốt nhất vs M0b, test):** M0b_TFIDF_LR tốt hơn gru có ý nghĩa thống kê (p = 0.000000 < 0.05; đúng riêng 1519 so với 1086 mẫu).

### 3.1 Chốt `max_len` — pilot D5 (LSTM, seed 42, val)

Quy tắc ghi trước khi chạy: val acc cao nhất; chênh < 0.5 điểm thì chọn max_len nhỏ hơn. Kết quả: chốt **400** (600 chỉ hơn 0,28 điểm). Best epoch đều dưới trần 20 epoch.

| `max_len` | Val acc | Best epoch | Thời gian (s) |
| :--- | :--- | :--- | :--- |
| 200 | 0,8302 | 13 | 162,2 |
| 250 | 0,8374 | 10 | 138,7 |
| 400 | 0,8601 | 7 | 128,5 |
| 600 | 0,8629 | 9 | 191,5 |

### 3.2 Mở rộng A — cách cắt và `max_len` trên mô hình tốt nhất (D7, 1 seed, val)

GRU seed 42 với `max_len` = 400, cắt `post` (mô hình chính): val acc 0,8827.

| Biến thể (GRU, seed 42) | `max_len` | Cách cắt | % train bị cắt | Val acc |
| :--- | :--- | :--- | :--- | :--- |
| `maxlen_200_post` | 200 | post | 52,32% | 0,8471 |
| `maxlen_600_post` | 600 | post | 7,88% | 0,8852 |
| `maxlen_400_pre` | 400 | pre | 18,73% | 0,8864 |
| `maxlen_400_head_tail` | 400 | head_tail | 18,73% | 0,8856 |

### 3.3 Lần chạy 1 (không dùng)

Lần chạy đầu (26/11, 04:11) dùng `max_len` = 250 mặc định vì bước D5 khi đó chỉ chạy 1 cấu hình, chưa chốt `max_len` như plan. Kết quả: mô hình tốt nhất BiLSTM + attention test 86,58%. Nhóm chạy lại toàn bộ D5–D8 theo đúng plan; bảng trên là lần chạy 2.
