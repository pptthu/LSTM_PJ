# IMDb Sentiment Analysis with LSTM

- **Motivation/Background**: Đồ án môn Học sâu (UTH), chủ đề 3 (nhóm LLMs) — phân loại cảm xúc review phim IMDb.
- **Purpose**: Điểm vào chính của repo: mục tiêu, cấu trúc thư mục, cài đặt, lệnh chạy, kết quả chính.
- **Overview Pipeline**: IMDb raw → làm sạch, loại trùng → split train/val theo phim → baseline Majority, TF-IDF + LR → LSTM / GRU / BiLSTM / BiLSTM + GloVe (3 seed) → đánh giá test, phân tích lỗi, giải thích → demo Gradio.
- **Detailed Plan**: §1 Mục tiêu; §2 Cấu trúc thư mục; §3 Cài đặt; §4 Lệnh chạy; §5 Kết quả; §6 Tài liệu.
- **References**: [docs/PURPOSE.md](docs/PURPOSE.md), [docs/README.md](docs/README.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-24T15:05:41+07:00

---

## 1. Mục tiêu

Phân loại review phim IMDb thành `positive`/`negative` bằng Embedding + LSTM + Linear, so sánh với GRU, BiLSTM, GloVe và baseline TF-IDF + Logistic Regression. Đề bài đầy đủ: [docs/PURPOSE.md](docs/PURPOSE.md).

## 2. Cấu trúc thư mục

```
DL-Project/
├── configs/            # YAML mỗi thí nghiệm (lstm.yaml, gru.yaml, ...)
├── data/               # KHÔNG commit, trừ sample/ — xem data/README.md
│   ├── raw/            # aclImdb_v1 giải nén (+ rotten_tomatoes/ nếu làm E4)
│   ├── splits/         # CSV train/val/test cố định
│   ├── glove/          # glove.6B.100d.txt
│   ├── processed/      # ma trận GloVe đã lọc theo vocab (.npy)
│   └── sample/         # sample_100.csv, handwritten_30.csv (được commit)
├── docs/               # Tài liệu: PURPOSE, OVERVIEW, DATA, MODELS, phases, progress, experiments, bugs
├── experiments/        # KHÔNG commit checkpoint — xem experiments/README.md
├── notebooks/          # EDA, kết quả, phân tích lỗi, demo.ipynb
├── src/
│   ├── train.py        # Entry: train một config với một seed
│   ├── evaluate.py     # Entry: đánh giá checkpoint trên val/test
│   ├── predict.py      # Entry + hàm predict(text) -> (label, prob)
│   ├── data/           # Tải, làm sạch, loại trùng, split, vocab, Dataset/DataLoader
│   ├── models/         # Baseline, SentimentRNN (LSTM/GRU/BiLSTM), nạp GloVe
│   ├── training/       # Vòng train, early stopping, lưu checkpoint
│   ├── eval/           # Metric, phân tích lỗi, giải thích mô hình
│   ├── demo/           # App Gradio
│   └── utils/          # Seed, logging, đường dẫn
└── tests/              # Pytest: shape, overfit 1 batch, save/load
```

## 3. Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Trên Kaggle:** không cài `requirements.txt` (bản ghim sẽ nâng thư viện của Kaggle) — xem [docs/phases/TRAINING.md](docs/phases/TRAINING.md) mục "Chạy trên Kaggle".

## 4. Lệnh chạy

Lệnh cụ thể ghi vào đây khi code từng phase xong. Thứ tự: tải dữ liệu → chia split → baseline → train (3 seed) → evaluate → demo. Xem [docs/OVERVIEW.md](docs/OVERVIEW.md).

Đã chạy được (chạy từ gốc repo):

```bash
python -m pytest                     # test offline, không cần dữ liệu thật
python -m src.data.download          # tải aclImdb (84 MB) vào data/raw/, đọc 2 split, lưu bộ đệm parquet
python -m src.data.eda               # EDA train / val / test → experiments/results/eda, experiments/plots/eda
```

Split train/val dùng lại của Lab 3: `data/splits/train_val_split.json` (22.499 / 2.501, chia theo phim) — xem [docs/DATA.md](docs/DATA.md) §3.

Dự kiến:

```bash
python -m src.train --config configs/lstm.yaml --seed 42      # dự kiến
python -m src.evaluate --exp lstm --split val                  # dự kiến
python -m src.predict "not bad at all"                         # dự kiến
python -m src.demo.app                                         # dự kiến
```

## 5. Kết quả

Test IMDb 25.000 review, mean ± std 3 seed, `max_len` = 400 (chốt ở pilot D5). Chọn mô hình theo val; test chạy một lần. Bảng đầy đủ: [docs/experiments/README.md](docs/experiments/README.md).

| Mô hình | Val acc | Test acc | Test macro-F1 |
| :--- | :--- | :--- | :--- |
| M0b TF-IDF + LR (baseline) | 89,85% | **89,94%** | 0,8994 |
| M1 LSTM (đề bài) | 86,09 ± 0,33% | 86,48 ± 0,32% | 0,8648 |
| M2 GRU (**chọn theo val**) | **88,27 ± 0,45%** | 88,16 ± 0,18% | 0,8816 |
| M3 BiLSTM | 86,29 ± 0,93% | 86,53 ± 0,45% | 0,8652 |
| M4a BiLSTM + GloVe (đóng băng) | 87,76 ± 0,31% | 87,87 ± 0,29% | 0,8787 |
| M4b BiLSTM + GloVe (fine-tune) | 86,70 ± 0,65% | 86,72 ± 0,34% | 0,8670 |
| E1 BiLSTM + attention | 88,26 ± 0,58% | 88,28 ± 0,16% | 0,8828 |

Baseline TF-IDF + LR vẫn hơn mô hình RNN tốt nhất, có ý nghĩa thống kê (McNemar, `experiments/results/mcnemar.json`). Phân tích lỗi: `experiments/results/error_analysis/`; giải thích mô hình: `experiments/results/explain/`.

## 6. Tài liệu

Chia task và trạng thái: [docs/TASKS.md](docs/TASKS.md). Mục lục: [docs/README.md](docs/README.md). Prompt để AI tạo/cập nhật từng file `.md`: [docs/MD_PROMPTS.md](docs/MD_PROMPTS.md).
