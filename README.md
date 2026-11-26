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

Chạy từ gốc repo, theo thứ tự. Bước 4–5 nên chạy trên GPU (Kaggle T4: khoảng 2–4 phút mỗi lần train); các bước khác chạy được trên CPU.

```bash
# 0. Kiểm tra code (offline, không cần dữ liệu thật)
python -m pytest

# 1. Dữ liệu: tải aclImdb (84 MB) → làm sạch, loại trùng, áp split Lab 3 → data/processed/reviews.parquet
python -m src.data.download
python -m src.data.prepare            # in số đã loại; ghi experiments/results/data_prep/prepare_stats.json
python -m src.data.eda                # experiments/results/eda/, experiments/plots/eda/

# 2. Baseline M0a / M0b (chọn C trên val)
python -m src.models.baseline
python -m src.models.baseline --eval-test     # chỉ chạy ở bước 6, một lần

# 3. GloVe (chỉ cần cho M4a/M4b): tải glove.6B.zip, giải nén glove.6B.100d.txt vào data/glove/ (xem docs/DATA.md §6)

# 4. Train: 6 config × 3 seed (42, 123, 2025); max_len = 400 đã chốt ở pilot D5
python -m src.train --config configs/gru.yaml --seed 42
#    config: lstm, gru, bilstm, bilstm_glove_frozen, bilstm_glove_ft, bilstm_attention

# 5. Đánh giá trên val, so sánh, chọn mô hình theo mean val acc
python -m src.evaluate --exp gru --split val
python -m src.evaluate --compare --split val

# 6. Test — một lần, sau khi đã chọn mô hình (ghi predictions_test.csv, không commit vì ~33 MB)
python -m src.evaluate --exp gru --split test
python -m src.evaluate --compare --split test
python -m src.evaluate --mcnemar-a experiments/results/gru/predictions_test.csv     --mcnemar-b experiments/results/baseline/predictions_test.csv --name-a gru

# 7. Phân tích lỗi (T2, cần predictions_test.csv của bước 6) và giải thích mô hình (T3)
python -m src.eval.error_analysis
python -m src.eval.explain

# 8. Dự đoán, demo, đo thời gian CPU (M5)
python -m src.predict "The cast is wonderful, but the script is a mess."
python -m src.demo.app                # mở http://127.0.0.1:7860
python -m src.demo.benchmark
```

Split train/val dùng lại của Lab 3: `data/splits/train_val_split.json` (22.499 / 2.501, chia theo phim) — xem [docs/DATA.md](docs/DATA.md) §3. Pipeline Kaggle đã dùng để chạy D5–D8 (cùng các bước trên) và kiểm M6: [scripts/kaggle/](scripts/kaggle/README.md). Checkpoint (`experiments/checkpoints/`) không có trong repo; `predict`, `demo`, `explain` tự chọn mô hình chính tốt nhất theo `experiments/results/` khi checkpoint có trên máy.

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
