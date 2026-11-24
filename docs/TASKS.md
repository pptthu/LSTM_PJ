# TASKS.md — Chia task và trạng thái

- **Motivation/Background**: Hạn nộp thứ 4 25/11/2026 trước 14:00; 3 người làm song song nên cần một chỗ ghi rõ ai làm gì, giao gì cho ai, lúc nào.
- **Purpose**: Danh sách task cụ thể theo từng người, file bàn giao giữa các người, và trạng thái đã làm / chưa làm.
- **Overview Pipeline**: Dữ liệu (Bảo) → mô hình + train (Đại) → đánh giá (Đại) → phân tích lỗi (Thư) → demo (Bảo) → báo cáo (Thư).
- **Detailed Plan**: §1 Đã làm được; §2 Chưa làm được; §3 Task từng người; §4 File bàn giao; §5 Mốc kiểm tra; §6 Luật chung.
- **References**: [OVERVIEW.md](OVERVIEW.md) (12 module), [PURPOSE.md](PURPOSE.md) §10, [phases/](phases/).
- **Created**: 2026-11-23T10:12:39+07:00
- **Last Updated**: 2026-11-24T12:10:00+07:00

---

## 1. Đã làm được (tính đến 23/11 sáng)

| Việc | Bằng chứng |
| :--- | :--- |
| Bản đăng ký 10 mục, bộ docs khung (phases, progress, DATA, MODELS…) | `docs/` (commit `54a51d8`) |
| Chép từ Lab 3: tải + đọc IMDb gốc (rating, mã phim), bộ đệm parquet | `src/data/download.py` |
| Chép từ Lab 3: làm sạch `<br />`, hash tìm review trùng | `src/data/clean.py` |
| Dùng lại split theo phim của Lab 3 (22.499 / 2.501), file đã commit | `data/splits/train_val_split.json`, `src/data/split.py` |
| Chép từ Lab 3: hàm EDA (độ dài, % bị cắt ở 200/400/600, trùng lặp, phủ định, 3 hình) chạy trên train/val/test | `src/data/eda.py` |
| 17 test offline (cây aclImdb giả + kiểm file split) | `python -m pytest` → 17 passed |
| Chạy thử EDA trên dữ liệu thật: 22.499 / 2.501 / 25.000 review; 123 review test trùng train (khớp Lab 3); 0 nhóm trùng khác nhãn | `python -m src.data.eda` (kết quả chạy thử không commit) |
| Chốt quyết định: split Lab 3; quy tắc loại trùng; chọn checkpoint theo val acc; phần mở rộng chia mức A/B/C; hạn 14:00 T4; báo cáo PDF tiếng Việt + tiếng Anh, notebook tiếng Việt | [DATA.md](DATA.md) §3–§4, [phases/TRAINING.md](phases/TRAINING.md), [OVERVIEW.md](OVERVIEW.md) §2 |
| Owner + hạn cho từng phase; phần Monitor training trong TRAINING; bảng so sánh + McNemar trong EVAL | `docs/phases/*.md`, `docs/progress/*.md` |
| B1–B2: loại trùng + test (Bảo) | `src/data/prepare.py`, `tests/test_prepare.py`; train 22.499→22.304, val 2.501→2.473, test 25.000; 0 hash/movie chung giữa split; `python -m pytest` → 32 passed; số trong `experiments/results/data_prep/prepare_stats.json` |
| B1–B2 (bổ sung sau review PR #4) | `test_uids_dup_train.txt` (123 uid test trùng train), `content_hash` = 1029824945594009414 trong `prepare_stats.json`, `check_contract` raise khi `movie_id` chung; `python -m pytest` → 35 passed |
| B3: EDA trên dữ liệu đã loại trùng (Bảo) | `python -m src.data.eda` → 8 CSV + 6 hình trong `experiments/results/eda/`, `experiments/plots/eda/`; DATA.md §5 có số thật (n = 49.777; median 173 từ; phủ định 89,5% review) |
| B4: `sample_100.csv` (Bảo) | `src/data/sample.py` → 100 review test (50/50), có rating 4/7, phủ định, ngắn, dài; tất định seed 42; `tests/test_sample.py` 7 test |
| B5: baseline M0a/M0b (Bảo) | `python -m src.models.baseline` → M0a val acc 0,4994; M0b `C=10` val acc 0,8985, macro-F1 0,8985, AUC 0,9623; `experiments/results/baseline/`; [EXP_01](experiments/EXP_01_BASELINE.md) |

## 2. Chưa làm được

| Việc | Ai | Ghi chú |
| :--- | :--- | :--- |
| Tokenizer, vocab, Dataset; `SentimentRNN`; GloVe | Đại | D1–D3 |
| Vòng train + monitor; pilot; 15 lần chạy chính; mở rộng A/B | Đại | D4–D7 |
| Chạy test một lần; bảng so sánh; McNemar; file dự đoán | Đại | D8 |
| 30 câu tự viết; phân tích lỗi; giải thích mô hình | Thư | T1–T3 |
| Demo Gradio | Bảo | B6 |
| README, notebook, báo cáo tiếng Việt + tiếng Anh, slide | Thư | T4–T5 |
| Thêm Bảo, Đại làm collaborator; điền MSSV trong PURPOSE | Thư | trước khi giao task |

## 3. Task từng người

Trạng thái: ⬜ chưa làm · 🟨 đang làm · ✅ xong (ghi bằng chứng vào `docs/progress/<PHASE>_STATUS.md`).

### Bảo — Dữ liệu, EDA, baseline, demo

| ID | Task | File | Đầu ra | Hạn | Phụ thuộc | TT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| B1 | Đọc 2 split → `apply_split` → làm sạch → loại trùng theo [DATA.md](DATA.md) §4; in số đã loại | `src/data/prepare.py` | `data/processed/reviews.parquet` | T2 23/11 tối | — | ✅ |
| B2 | Test loại trùng: không còn hash chung giữa train/val và test; test đủ 25.000 | `tests/test_prepare.py` | test pass | T2 tối | B1 | ✅ |
| B3 | **EDA** sau khi loại trùng: chạy `src.data.eda`; thêm từ phổ biến từng lớp (cột, không dùng biểu đồ tròn); điền [DATA.md](DATA.md) §3–§5 | `src/data/eda.py` | `experiments/results/eda/`, `experiments/plots/eda/` | T2 tối | B1 | ✅ |
| B4 | Chọn 100 review từ test có ca khó (phủ định, dài, rating 4/7) | `src/data/sample.py` | `data/sample/sample_100.csv` | T2 tối | B1 | ✅ |
| B5 | Baseline Majority + TF-IDF (1–2 gram) + LR, chọn `C` trên val; lưu dự đoán val/test cùng định dạng §4 | `src/models/baseline.py` | `experiments/results/baseline/` | T2 tối (val); test chạy cùng lúc với Đại ở D8 | B1 | ✅ |
| B6 | Demo Gradio: nhập review → nhãn, xác suất, tô màu từ; đo thời gian trên CPU | `src/predict.py`, `src/demo/app.py` | ảnh chụp demo | T4 25/11 10:00 | D7 (checkpoint), T3 (hàm giải thích) | ⬜ |

### Đại — Mô hình, huấn luyện, monitor, evaluation

| ID | Task | File | Đầu ra | Hạn | Phụ thuộc | TT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| D1 | Tokenizer (lowercase, giữ `not`, `n't`, `!`, `?`, không bỏ stopword), vocab chỉ từ train, Dataset/DataLoader trả độ dài thật | `src/data/tokenizer.py`, `vocab.py`, `dataset.py` | test pass | T2 tối | B1 (có thể code trước trên cây aclImdb giả) | ⬜ |
| D2 | `SentimentRNN` (LSTM/GRU, 1/2 chiều, `pack_padded_sequence`, trả logit) | `src/models/rnn.py` | test shape, overfit 1 batch, save/load | T2 tối | — | ⬜ |
| D3 | GloVe 6B 100d lọc theo vocab, ghi tỷ lệ phủ | `src/models/glove.py` | `data/processed/glove_100d_vocab.npy` | T2 tối | D1 | ⬜ |
| D4 | Vòng train + early stopping theo **val acc** + **monitor** (`history.csv`, learning curve) theo [phases/TRAINING.md](phases/TRAINING.md) | `src/train.py`, `src/training/monitor.py`, `configs/*.yaml` | chạy 1 epoch trên 500 mẫu không lỗi | T2 tối | D1, D2 | ⬜ |
| D5 | Pilot M1 trên Kaggle → chốt `max_len`, trần epoch, patience (ghi lý do vào progress) | — | log pilot | T3 24/11 sáng | B1, D4 | ⬜ |
| D6 | M1–M4b × 3 seed (42, 123, 2025); bảng so sánh val; hình learning curve | — | `experiments/results/<exp>/`, `experiments/plots/training/` | T3 **14:00 (mốc dừng)** | D5 | ⬜ |
| D7 | Mở rộng A: `max_len` 200/400/600 + cách cắt (1 seed); B (nếu đạt mốc dừng): attention pooling × 3 seed | — | bảng `max_len` | T3 chiều–tối | D6 | ⬜ |
| D8 | **Evaluation**: chốt mô hình theo val → chạy test **một lần**; metric đủ, confusion matrix, ROC, McNemar vs B1, số tham số, thời gian; xuất `predictions_test.csv` | `src/evaluate.py`, `src/eval/metrics.py` | `experiments/results/comparison_*.csv`, `mcnemar.json`, file dự đoán | T3 tối | D6, B5 | ⬜ |

### Thư — Chốt, phân tích lỗi, báo cáo

| ID | Task | File | Đầu ra | Hạn | Phụ thuộc | TT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| T0 | Duyệt + merge PR; thêm collaborator; điền MSSV | — | — | liên tục | — | ⬜ |
| T1 | Viết 30 câu tự viết **trước khi** chạy mô hình: 5 tích cực, 5 tiêu cực, 5 phủ định, 5 mỉa mai, 5 vừa khen vừa chê, 5 ngoài miền | `data/sample/handwritten_30.csv` | file 30 dòng (`text,label,case_type`) | T2 tối | — | ⬜ |
| T2 | **Error analysis**: accuracy theo 7 nhóm tình huống (kèm số mẫu); ≥ 10 ví dụ sai/nhóm có giải thích; gán tay 50 lỗi (vừa khen vừa chê, mỉa mai); kết quả 30 câu | `src/eval/error_analysis.py` | `experiments/results/error_analysis/` | T3 tối – T4 sáng | D8 | ⬜ |
| T3 | Giải thích mô hình: occlusion (+ deletion test) trên mẫu review; kiểm "đường tắt" | `src/eval/explain.py` | hình heatmap từ | T4 sáng | D8 | ⬜ |
| T4 | Notebook trình bày (tiếng Việt), section: EDA → Baseline → Model → **Monitor training** → **Evaluation** → **Error analysis** → Demo; chỉ đọc `experiments/` | `notebooks/presentation.ipynb` | notebook chạy hết | T4 12:00 | B3, D6, D8, T2 | ⬜ |
| T5 | README, báo cáo PDF tiếng Việt → dịch tiếng Anh, slide; kiểm số khớp giữa các bản | `README.md`, `reports/` | 2 PDF + slide | **T4 trước 14:00** | tất cả | ⬜ |

## 4. File bàn giao

| Từ → đến | File | Cột / nội dung | Hạn |
| :--- | :--- | :--- | :--- |
| Bảo → Đại | `data/processed/reviews.parquet` (không commit; gửi file hoặc Đại tự chạy `python -m src.data.prepare`) | `uid, split, text, text_clean, label, rating, movie_id`; `split` ∈ train/val/test | T2 tối |
| Đại → Thư | `experiments/results/<exp>/predictions_<split>.csv` | `uid, text, label, rating, prob, pred, n_tokens, n_unk, was_truncated` | T3 tối |
| Đại → Bảo | checkpoint tốt nhất + `vocab.json` + hàm `predict(text) -> (label, prob)` | — | T3 tối |
| Bảo → Đại | dự đoán baseline cùng định dạng file dự đoán | để chạy McNemar | T3 tối |

## 5. Mốc kiểm tra

| Khi nào | Phải có | Nếu chưa có |
| :--- | :--- | :--- |
| T2 23/11 22:00 | `reviews.parquet`, EDA, baseline val; `SentimentRNN` + train chạy được trên tập con; 30 câu tự viết | Báo ngay trong nhóm; Thư quyết cắt gì |
| T3 24/11 14:00 | 5 cấu hình × 3 seed xong | Bỏ attention pooling (mức B), chỉ làm `max_len` |
| T3 24/11 22:00 | Test chạy xong, file dự đoán gửi Thư | Ưu tiên M1 + mô hình tốt nhất |
| T4 25/11 12:00 | Notebook + báo cáo tiếng Việt xong | Rút gọn bản tiếng Anh |
| T4 25/11 14:00 | **Nộp** | — |

## 6. Luật chung

- Nhánh riêng cho mỗi việc (`feat/<phase>-<viec>`), PR vào `main`; Bảo và Đại review chéo, Thư merge ([references/GIT_WORKFLOW.md](references/GIT_WORKFLOW.md)). Chạy `python -m pytest` trước khi mở PR.
- Chọn mô hình theo val; **test chỉ chạy một lần** sau khi chốt.
- Mọi số trong báo cáo phải có file trong `experiments/results/`; không sửa số khi không có file đi kèm.
- Không commit dữ liệu thô, GloVe, checkpoint.
- Làm xong task nào: đổi TT ở file này + ghi bằng chứng vào `docs/progress/<PHASE>_STATUS.md`.
