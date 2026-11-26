# EXP-05 BiLSTM + GloVe (đóng băng và fine-tune)

- **Motivation/Background**: Đề bài gợi ý thử GloVe. Embedding học từ đầu với 22 nghìn review dễ overfit; vector từ học sẵn trên Wikipedia + Gigaword có thể giúp.
- **Purpose**: So embedding ngẫu nhiên với GloVe 100d, hai cách: đóng băng (M4a) và cho train tiếp (M4b).
- **Overview Pipeline**: Giống [EXP_04](EXP_04_BILSTM.md), embedding khởi tạo từ `glove.6B.100d`.
- **Detailed Plan**: Header; Objective; Single variable; Setup; Results; Analysis; Reproduce; Links.
- **References**: `configs/bilstm_glove_frozen.yaml`, `configs/bilstm_glove_ft.yaml`, `src/models/glove.py`, [../DATA.md](../DATA.md) §6.
- **Created**: 2026-11-26
- **Last Updated**: 2026-11-26

---

## Header

- **Title:** BiLSTM + GloVe 100d (frozen / fine-tune)
- **Date created:** 2026-11-26
- **Last updated:** 2026-11-26
- **Description:** Embedding ngẫu nhiên → GloVe; hai biến thể; 3 seed mỗi biến thể.
- **Status:** Done
- **Experiment ID:** EXP-05 (M4a, M4b)

## Objective

1. GloVe có tốt hơn embedding học từ đầu không?
2. Nên đóng băng GloVe hay cho train tiếp?

## Single variable changed / held constant

- **Changed:** khởi tạo embedding (ngẫu nhiên → GloVe); và `freeze_embeddings` (`true` ở M4a, `false` ở M4b).
- **Held constant:** BiLSTM 128 × 2, hidden cuối, dropout 0,5, vocab, `max_len` 400, optimizer, lr 1e-3, early stopping, 3 seed.

## Setup

- **GloVe:** `glove.6B.100d.txt` (Pennington et al., 2014), lọc theo vocab 25.000 từ → ma trận `data/processed/glove_100d_vocab.npy`. **Tỷ lệ phủ vocab: 98,72%**; từ không có trong GloVe khởi tạo ngẫu nhiên.
- **M4a:** embedding đóng băng → chỉ train 235.777 tham số (8,6% của 2.735.777).
- **M4b:** embedding train tiếp với cùng lr 1e-3.

## Results

| | Val acc | Test acc | Test macro-F1 | Test ROC-AUC | Tham số train được | Thời gian train/seed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| M3 BiLSTM (ngẫu nhiên) | 86,29 ± 0,93% | 86,53 ± 0,45% | 0,8652 | 0,9369 | 2.735.777 | 194,2 s |
| **M4a GloVe đóng băng** | **87,76 ± 0,31%** | **87,87 ± 0,29%** | 0,8787 | 0,9505 | 235.777 | 227,4 s |
| M4b GloVe fine-tune | 86,70 ± 0,65% | 86,72 ± 0,34% | 0,8670 | 0,9441 | 2.735.777 | 109,5 s |

| Mô hình | Seed | Epoch tốt nhất / dừng | Val acc | Train acc tại epoch tốt nhất | Test acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| M4a | 42 | 12 / 15 | 0,8767 | 0,9273 | 0,8794 |
| M4a | 123 | 9 / 12 | 0,8751 | 0,8996 | 0,8756 |
| M4a | 2025 | 14 / 17 | 0,8811 | 0,9367 | 0,8812 |
| M4b | 42 | 4 / 7 | 0,8734 | 0,9419 | 0,8665 |
| M4b | 123 | 3 / 6 | 0,8670 | 0,8908 | 0,8708 |
| M4b | 2025 | 3 / 6 | 0,8605 | 0,8840 | 0,8642 |

## Analysis / interpretation

- **GloVe đóng băng tốt nhất trong nhóm:** +1,47 điểm val, +1,34 điểm test so với BiLSTM ngẫu nhiên; độ lệch giữa seed nhỏ nhất (± 0,31). Chỉ train 8,6% tham số nên **ít overfit hơn** (train acc tại điểm tốt nhất ~92%, thấp hơn các mô hình khác).
- **Fine-tune GloVe làm mất phần lớn lợi ích** (chỉ +0,41 điểm val so với ngẫu nhiên): hội tụ rất nhanh (epoch tốt nhất 3–4) rồi overfit — với lr 1e-3, 2,5 triệu tham số embedding bị kéo lệch khỏi GloVe chỉ sau vài epoch. Thử lr nhỏ hơn cho embedding là hướng mở rộng.
- Đổi lại, đóng băng **hội tụ chậm** (epoch tốt nhất 9–14) nên tổng thời gian train lâu nhất (227 s/seed).
- GloVe vẫn chỉ là **học sẵn ở mức từ**; mô hình vẫn thua TF-IDF + LR 2,1 điểm. Các kết quả > 90% trong tài liệu đều học sẵn cả mạng (language model, Transformer).

## Reproduce

```bash
# cần data/glove/glove.6B.100d.txt (xem docs/DATA.md §6)
python -m src.train --config configs/bilstm_glove_frozen.yaml --seed 42
python -m src.train --config configs/bilstm_glove_ft.yaml --seed 42
```

## Links

- [EXP_04 BiLSTM](EXP_04_BILSTM.md), [EXP_06 Attention](EXP_06_ATTENTION.md).
