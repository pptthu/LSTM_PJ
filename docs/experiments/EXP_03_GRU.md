# EXP-03 GRU

- **Motivation/Background**: Đề bài gợi ý thử GRU. GRU có 2 cổng thay vì 3 như LSTM, ít tham số hơn; cần biết có kém đi không.
- **Purpose**: So GRU với LSTM khi chỉ đổi loại ô hồi quy.
- **Overview Pipeline**: Giống [EXP_02](EXP_02_LSTM.md), thay LSTM bằng GRU.
- **Detailed Plan**: Header; Objective; Single variable; Setup; Results; Analysis; Reproduce; Links.
- **References**: `configs/gru.yaml`, `experiments/results/gru/`.
- **Created**: 2026-11-26
- **Last Updated**: 2026-11-26

---

## Header

- **Title:** GRU 1 chiều, embedding học từ đầu
- **Date created:** 2026-11-26
- **Last updated:** 2026-11-26
- **Description:** LSTM → GRU, giữ nguyên mọi thứ khác; 3 seed.
- **Status:** Done — **mô hình được chọn theo val** (mean val acc cao nhất trong 6 mô hình).
- **Experiment ID:** EXP-03 (M2)

## Objective

GRU có tốt bằng hoặc hơn LSTM trên IMDb không, khi ít tham số và nhanh hơn?

## Single variable changed / held constant

- **Changed:** `rnn_type`: `lstm` → `gru`.
- **Held constant:** dữ liệu, vocab, embedding 100d, hidden 128, 1 chiều, hidden cuối, dropout 0,5, `max_len` 400, optimizer, early stopping, 3 seed.

## Setup

Như [EXP_02](EXP_02_LSTM.md) §Setup, chỉ khác ô hồi quy là GRU.

## Results

| | Val acc | Test acc | Test macro-F1 | Test ROC-AUC | Tham số | Thời gian train/seed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| M1 LSTM | 86,09 ± 0,33% | 86,48 ± 0,32% | 0,8648 | 0,9321 | 2.617.889 | 162,7 s |
| **M2 GRU** | **88,27 ± 0,45%** | **88,16 ± 0,18%** | **0,8816** | 0,9499 | 2.588.449 | 122,8 s |
| Chênh GRU − LSTM | +2,18 | +1,68 | +0,0168 | +0,0178 | −29.440 | −25% |

| Seed | Epoch tốt nhất / dừng | Val acc | Train acc tại epoch tốt nhất | Test acc |
| :--- | :--- | :--- | :--- | :--- |
| 42 | 7 / 10 | 0,8827 | 0,9651 | 0,8796 |
| 123 | 7 / 10 | 0,8872 | 0,9681 | 0,8821 |
| 2025 | 6 / 9 | 0,8783 | 0,9532 | 0,8830 |

McNemar (GRU seed 123 vs TF-IDF + LR, test): baseline đúng riêng 1.519 review, GRU đúng riêng 1.086, p < 0,001 → **baseline tốt hơn có ý nghĩa thống kê** (`experiments/results/mcnemar.json`).

## Analysis / interpretation

- **GRU hơn LSTM khoảng 2 điểm** trên cả val và test, dù ít tham số hơn và train nhanh hơn 25%.
- **Ổn định hơn:** 3 seed đều tốt nhất ở epoch 6–7; val loss tại điểm tốt nhất 0,337 ± 0,005 so với 0,44 ± 0,07 của LSTM. Cấu trúc 2 cổng dễ tối ưu hơn trên chuỗi dài 400 token với lượng dữ liệu vừa phải.
- **Đạt M2** (88,16% ≥ 88%, macro-F1 0,8816 ≥ 0,88).
- GRU vẫn overfit (train ~96% so với val ~88%) và vẫn **thua TF-IDF + LR 1,8 điểm**.
- Được chọn vì có mean val acc cao nhất (88,27%), chỉ hơn BiLSTM + attention 0,01 điểm — thực chất hai mô hình ngang nhau ([EXP_06](EXP_06_ATTENTION.md)).

## Reproduce

```bash
python -m src.train --config configs/gru.yaml --seed 42      # lặp lại với 123, 2025
```

## Links

- [EXP_02 LSTM](EXP_02_LSTM.md), [EXP_06 Attention](EXP_06_ATTENTION.md). Phân tích lỗi trên GRU: `experiments/results/error_analysis/`.
