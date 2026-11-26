# EXP-06 (Mở rộng) BiLSTM + attention pooling

- **Motivation/Background**: Lấy hidden cuối buộc mô hình dồn cả review dài vào một vector ở cuối chuỗi. Attention pooling cho mô hình tự chọn token quan trọng ở bất kỳ vị trí nào.
- **Purpose**: So attention pooling với hidden cuối trên BiLSTM.
- **Overview Pipeline**: BiLSTM → điểm attention cho từng token (có mask padding) → trung bình có trọng số → Linear.
- **Detailed Plan**: Header; Objective; Single variable; Setup; Results; Analysis; Reproduce; Links.
- **References**: `configs/bilstm_attention.yaml`, `src/models/rnn.py`, `experiments/results/bilstm_attention/`.
- **Created**: 2026-11-26
- **Last Updated**: 2026-11-26

---

## Header

- **Title:** BiLSTM + attention pooling (mở rộng mức B)
- **Date created:** 2026-11-26
- **Last updated:** 2026-11-26
- **Description:** Hidden cuối → attention pooling; 3 seed.
- **Status:** Done
- **Experiment ID:** EXP-06 (E1)

## Objective

Cho mô hình nhìn vào mọi vị trí trong review (thay vì chỉ hidden cuối) có tăng accuracy không?

## Single variable changed / held constant

- **Changed:** `pooling`: `last` → `attention`.
- **Held constant:** BiLSTM 128 × 2, embedding 100d ngẫu nhiên, dropout 0,5, vocab, `max_len` 400, optimizer, early stopping, 3 seed.

## Setup

Như [EXP_04](EXP_04_BILSTM.md). Thêm một lớp tính điểm cho mỗi output BiLSTM, softmax có mask (token padding nhận trọng số 0), tổng có trọng số → Linear. Thêm 33.025 tham số.

## Results

| | Val acc | Test acc | Test macro-F1 | Test ROC-AUC | Tham số | Thời gian train/seed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| M3 BiLSTM (hidden cuối) | 86,29 ± 0,93% | 86,53 ± 0,45% | 0,8652 | 0,9369 | 2.735.777 | 194,2 s |
| **E1 BiLSTM + attention** | 88,26 ± 0,58% | **88,28 ± 0,16%** | **0,8828** | **0,9527** | 2.768.802 | 171,2 s |
| M2 GRU (được chọn) | 88,27 ± 0,45% | 88,16 ± 0,18% | 0,8816 | 0,9499 | 2.588.449 | 122,8 s |

| Seed | Epoch tốt nhất / dừng | Val acc | Train acc tại epoch tốt nhất | Test acc |
| :--- | :--- | :--- | :--- | :--- |
| 42 | 4 / 7 | 0,8876 | 0,9354 | 0,8837 |
| 123 | 4 / 7 | 0,8839 | 0,9366 | 0,8838 |
| 2025 | 3 / 6 | 0,8763 | 0,9118 | 0,8810 |

## Analysis / interpretation

- **Attention là thay đổi có lợi nhất trên BiLSTM:** +1,97 điểm val, +1,75 điểm test so với hidden cuối; ROC-AUC và val loss tại điểm tốt nhất (0,291) tốt nhất trong 6 mô hình.
- **Hội tụ nhanh** (epoch tốt nhất 3–4) vì gradient đi thẳng tới token quan trọng, không phải truyền ngược qua 400 bước.
- **Ngang GRU:** val thấp hơn GRU 0,01 điểm, test cao hơn 0,12 điểm — đều nhỏ hơn độ lệch giữa các seed. Nhóm chọn mô hình **theo val** nên chọn GRU; không đổi lựa chọn sau khi xem test.
- Trên 30 câu tự viết (lần chạy 1, `max_len` 250), attention đoán đúng 3/5 câu "vừa khen vừa chê", GRU lần chạy 2 chỉ 1/5 — gợi ý attention xử lý đổi giọng tốt hơn, nhưng mẫu quá nhỏ để kết luận.

## Reproduce

```bash
python -m src.train --config configs/bilstm_attention.yaml --seed 42      # lặp lại với 123, 2025
```

## Links

- [EXP_03 GRU](EXP_03_GRU.md), [EXP_04 BiLSTM](EXP_04_BILSTM.md).
