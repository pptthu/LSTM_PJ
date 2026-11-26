# EXP-04 BiLSTM

- **Motivation/Background**: Đề bài gợi ý thử Bidirectional LSTM. Đọc thêm chiều ngược có thể giúp thấy cả phần cuối review (nơi hay có kết luận).
- **Purpose**: So LSTM 2 chiều với LSTM 1 chiều.
- **Overview Pipeline**: Giống [EXP_02](EXP_02_LSTM.md), LSTM 2 chiều, nối hidden cuối của hai chiều (2 × 128).
- **Detailed Plan**: Header; Objective; Single variable; Setup; Results; Analysis; Reproduce; Links.
- **References**: `configs/bilstm.yaml`, `experiments/results/bilstm/`.
- **Created**: 2026-11-26
- **Last Updated**: 2026-11-26

---

## Header

- **Title:** BiLSTM, embedding học từ đầu
- **Date created:** 2026-11-26
- **Last updated:** 2026-11-26
- **Description:** 1 chiều → 2 chiều; 3 seed.
- **Status:** Done
- **Experiment ID:** EXP-04 (M3)

## Objective

Đọc review theo cả hai chiều có làm tăng accuracy so với LSTM 1 chiều không?

## Single variable changed / held constant

- **Changed:** `bidirectional`: `false` → `true` (Linear nhận 256 thay vì 128).
- **Held constant:** dữ liệu, vocab, embedding 100d, hidden 128 mỗi chiều, hidden cuối, dropout 0,5, `max_len` 400, optimizer, early stopping, 3 seed.

## Setup

Như [EXP_02](EXP_02_LSTM.md) §Setup. Hidden cuối của chiều xuôi (token cuối thật, nhờ `pack_padded_sequence`) và chiều ngược (token đầu) được nối lại.

## Results

| | Val acc | Test acc | Test macro-F1 | Test ROC-AUC | Tham số | Thời gian train/seed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| M1 LSTM | 86,09 ± 0,33% | 86,48 ± 0,32% | 0,8648 | 0,9321 | 2.617.889 | 162,7 s |
| M3 BiLSTM | 86,29 ± **0,93**% | 86,53 ± 0,45% | 0,8652 | 0,9369 | 2.735.777 | 194,2 s |
| Chênh BiLSTM − LSTM | +0,20 | +0,05 | +0,0004 | +0,0048 | +117.888 | +19% |

| Seed | Epoch tốt nhất / dừng | Val acc | Train acc tại epoch tốt nhất | Test acc |
| :--- | :--- | :--- | :--- | :--- |
| 42 | 8 / 11 | 0,8730 | 0,9579 | 0,8704 |
| 123 | 9 / 12 | 0,8609 | 0,9327 | 0,8630 |
| 2025 | 8 / 11 | 0,8548 | 0,9388 | 0,8624 |

## Analysis / interpretation

- **Hai chiều gần như không giúp:** +0,20 điểm val, +0,05 điểm test — nhỏ hơn nhiều so với độ lệch giữa các seed (± 0,93 điểm val, lớn nhất trong 6 mô hình). Không thể nói BiLSTM tốt hơn LSTM.
- Tốn thêm 4,5% tham số và chậm hơn 19%.
- Lý do có thể: cảm xúc review phần lớn nằm ở **từ khoá** (great, worst, 7/10…) chứ không ở ngữ cảnh phải đọc ngược mới hiểu; và chiều ngược chỉ cho ra hidden ở token đầu, không phải tóm tắt tốt của cả review.
- BiLSTM là nền cho [EXP_05 GloVe](EXP_05_BILSTM_GLOVE.md) và [EXP_06 Attention](EXP_06_ATTENTION.md) — hai thay đổi này mới làm tăng điểm.

## Reproduce

```bash
python -m src.train --config configs/bilstm.yaml --seed 42      # lặp lại với 123, 2025
```

## Links

- [EXP_02 LSTM](EXP_02_LSTM.md), [EXP_05 GloVe](EXP_05_BILSTM_GLOVE.md), [EXP_06 Attention](EXP_06_ATTENTION.md).
