# EXP-07 (Mở rộng) `max_len` và cách cắt

- **Motivation/Background**: Review IMDb dài (median 173 từ, 10% dài hơn 455 từ) và hay đặt kết luận ở cuối. Cắt ngắn có thể làm mất câu quyết định.
- **Purpose**: (1) Chốt `max_len` trước 18 lần chạy chính (pilot D5); (2) đo ảnh hưởng của `max_len` và cách cắt trên mô hình tốt nhất (mở rộng A, D7).
- **Overview Pipeline**: Pilot LSTM 4 mức `max_len` trên val → chốt → 18 lần chạy chính → D7 trên GRU.
- **Detailed Plan**: Header; Objective; Single variable; Setup; Results; Analysis; Reproduce; Links.
- **References**: `experiments/results/pilot_maxlen.csv`, `pilot_decision.json`, `maxlen_comparison.csv`, `src/data/dataset.py` (`truncate_tokens`).
- **Created**: 2026-11-26
- **Last Updated**: 2026-11-26

---

## Header

- **Title:** Ablation `max_len` (200 / 250 / 400 / 600) và cách cắt (`post` / `pre` / `head_tail`)
- **Date created:** 2026-11-26
- **Last updated:** 2026-11-26
- **Description:** Pilot D5 chốt `max_len` = 400; D7 thử cách cắt trên GRU. 1 seed (42), chỉ val.
- **Status:** Done
- **Experiment ID:** EXP-07

## Objective

1. `max_len` bao nhiêu là đủ cho LSTM trên IMDb?
2. Khi phải cắt, giữ phần đầu, phần cuối hay cả hai?

## Single variable changed / held constant

- **Changed:** `max_len`; ở D7 thêm `truncation` (`post` = giữ đầu, cắt đuôi; `pre` = giữ cuối; `head_tail` = giữ một phần đầu + một phần cuối).
- **Held constant:** mọi thứ khác của mô hình nền (LSTM ở D5, GRU ở D7), seed 42, chỉ đánh giá trên val.

## Setup

- **Quy tắc chốt (ghi trước khi chạy):** lấy val acc cao nhất; nếu chênh < 0,5 điểm thì chọn `max_len` nhỏ hơn (chạy nhanh hơn).
- Chỉ 1 seed nên chênh lệch < 0,5 điểm coi như ngang nhau (độ lệch giữa seed của LSTM là ± 0,33 điểm).

## Results

**Pilot D5 — LSTM, seed 42, val:**

| `max_len` | % train bị cắt | Val acc | Epoch tốt nhất | Thời gian (s) |
| :--- | :--- | :--- | :--- | :--- |
| 200 | 52,3% | 0,8302 | 13 | 162,2 |
| 250 | 38,6% | 0,8374 | 10 | 138,7 |
| **400** | 18,7% | **0,8601** | 7 | 128,5 |
| 600 | 7,9% | 0,8629 | 9 | 191,5 |

→ 600 chỉ hơn 400 là 0,28 điểm (< 0,5) nên **chốt 400**. Epoch tốt nhất đều dưới trần 20 epoch, giữ nguyên trần và patience 3.

**Mở rộng A (D7) — GRU, seed 42, val:**

| Biến thể | `max_len` | Cách cắt | Val acc |
| :--- | :--- | :--- | :--- |
| `maxlen_200_post` | 200 | post | 0,8471 |
| GRU chính (seed 42) | 400 | post | 0,8827 |
| `maxlen_400_pre` | 400 | pre | 0,8864 |
| `maxlen_400_head_tail` | 400 | head_tail | 0,8856 |
| `maxlen_600_post` | 600 | post | 0,8852 |

**Lần chạy 1 (`max_len` = 250, không theo plan) so với lần chạy 2 (400), test acc:**

| Mô hình | 250 | 400 | Chênh |
| :--- | :--- | :--- | :--- |
| LSTM | 84,92% | 86,48% | +1,56 |
| GRU | 86,58% | 88,16% | +1,58 |
| BiLSTM | 84,33% | 86,53% | +2,20 |
| BiLSTM + GloVe đóng băng | 86,21% | 87,87% | +1,66 |
| BiLSTM + GloVe fine-tune | 84,78% | 86,72% | +1,94 |
| BiLSTM + attention | 86,58% | 88,28% | +1,70 |

## Analysis / interpretation

- **`max_len` là siêu tham số ảnh hưởng lớn nhất trong đồ án:** từ 250 lên 400, **mọi** mô hình tăng 1,6–2,2 điểm test; LSTM đi từ trượt M1 (< 85%) sang đạt.
- **Qua 400 thì gần như bão hoà:** 600 chỉ hơn 0,3 điểm (cả LSTM và GRU) nhưng chậm hơn ~40–50%.
- **Cách cắt không quan trọng ở 400:** `pre` và `head_tail` hơn `post` 0,3–0,4 điểm với 1 seed — trong mức nhiễu. Ở 400 chỉ 18,7% review bị cắt, nên cách cắt chỉ ảnh hưởng một phần nhỏ dữ liệu. Phân tích lỗi vẫn cho thấy review > 600 token có accuracy thấp nhất (80,8%, so với baseline 89,9%).
- **Bài học quy trình:** lần chạy 1 bỏ qua bước chốt `max_len` của pilot D5 (chỉ chạy 1 cấu hình mặc định 250). Phát hiện qua D7 và chạy lại toàn bộ D5–D8 đúng plan. Lab 3 (DistilBERT, 256 → 512 subword: +2,04 điểm) cho kết luận giống nhau.

## Reproduce

Pipeline Kaggle (stage `d5_pilot`, `d7_maxlen`) dùng `run_training` với config sửa `max_len` / `truncation`:

```bash
# ví dụ một biến thể: sửa data.max_len và data.truncation trong một bản sao của configs/gru.yaml
python -m src.train --config <bản sao>.yaml --seed 42 --exp-name maxlen_400_pre
```

## Links

- [README.md](README.md) §3.1–3.3, [EXP_02 LSTM](EXP_02_LSTM.md), [EXP_03 GRU](EXP_03_GRU.md).
