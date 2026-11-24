# TRAINING.md — Huấn luyện

- **Motivation/Background**: Mọi thí nghiệm cần cùng một vòng train để kết quả so sánh được; số phải tái lập được qua 3 seed.
- **Purpose**: Vòng train dùng chung cho mọi mô hình RNN, có early stopping, lưu checkpoint tốt nhất, chạy 3 seed.
- **Overview Pipeline**: Phạm vi → đặc tả → artifact → kiểm chứng.
- **Detailed Plan**: Header; Goals; Input/Output; How to do it; Detailed plan / gotchas; Verification gates; Links.
- **References**: [OVERVIEW.md](../OVERVIEW.md), [PURPOSE.md](../PURPOSE.md) §4 (M6), §7.
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-24T15:05:41+07:00

---

## Header

- **Title:** Huấn luyện
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-24
- **Description:** Vòng train dùng chung cho mọi mô hình RNN, có early stopping, lưu checkpoint tốt nhất, chạy 3 seed.
- **Status:** To Do
- **Owner:** Đại
- **Hạn:** T3 24/11 — mốc dừng 14:00: chưa xong 5 cấu hình × 3 seed thì bỏ attention pooling

## Goals / Purpose

- Xong: `python -m src.train --config configs/<exp>.yaml --seed <s>` chạy được; lưu `best.pt` theo **val acc**; **monitor**: log + learning curve mỗi lần chạy; mỗi cấu hình có 3 seed và file mean ± std.
- Mốc: pilot sáng T3 24/11 — LSTM ≥ 85% trên val (nếu không: grid nhỏ, OVERVIEW §2).
- Không làm: tìm siêu tham số tự động diện rộng.

## Input / Output

- **Input:** `data/processed/reviews.parquet` (Bảo), `configs/<exp>.yaml`.
- **Output:**
  - `experiments/runs/<exp>/seed_<s>/history.csv` (log từng epoch, xem mục Monitor).
  - `experiments/checkpoints/<exp>/vocab.json`, `experiments/checkpoints/<exp>/seed_<s>/best.pt` (không commit).
  - `experiments/results/<exp>/seed_<s>/metrics.json` (val), `experiments/results/<exp>/summary.json` (mean ± std 3 seed).
  - `experiments/plots/training/<exp>_curves.png` (commit).

## How to do it (general plan)

1. Dataset/DataLoader (`src/data/dataset.py`): tokenize, map id, cắt/đệm `max_len`, trả về độ dài thật.
2. `BCEWithLogitsLoss`, Adam (lr 1e-3), gradient clipping 5.0 (`src/training/`).
3. Mỗi epoch: `model.train()` → train; `model.eval()` + `torch.no_grad()` → val.
4. Early stopping và chọn checkpoint theo **val accuracy** (Thư chốt 2026-11-23; căn cứ Lab 3: val loss thấp nhất ở epoch 1 rồi tăng, trong khi val acc còn tăng tới epoch 3–5). Patience và trần epoch chốt sau pilot; lưu checkpoint tốt nhất kèm config.
5. Chạy seed 42, 123, 2025; gộp mean ± std vào `summary.json`.

## Monitor training progress (`src/training/monitor.py`)

Nhóm trình bày trước bị thiếu phần này → bắt buộc có hình trong notebook và báo cáo.

1. **Log mỗi epoch** vào `history.csv`: `epoch, train_loss, train_acc, val_loss, val_acc, val_f1, grad_norm_mean, lr, epoch_seconds`; in cùng dòng ra màn hình (log Kaggle giữ lại được).
2. **Learning curve** cho từng cấu hình: loss và acc của train/val theo epoch, chồng 3 seed; đánh dấu epoch tốt nhất (theo val acc) và epoch dừng sớm.
3. **Chẩn đoán**: khoảng cách train acc − val acc theo epoch (overfit); val loss tăng trong khi val acc còn tăng (lý do chọn theo val acc); grad norm bất thường (nổ gradient).
4. **Bảng tóm tắt** mỗi lần chạy trong `metrics.json`: epoch tốt nhất, epoch dừng, val acc tốt nhất, tổng thời gian train, phần cứng (GPU Kaggle).
5. Notebook có section riêng "Theo dõi quá trình huấn luyện", chỉ đọc `history.csv` và hình, không train lại.

## Chạy trên Kaggle

`requirements.txt` ghim `pandas==3.1.1`, `pyarrow==26.0.1`, `scikit-learn==1.9.1` theo máy đã tạo dữ liệu và baseline (Bảo, Python 3.12). Cài nguyên file lên Kaggle sẽ nâng các gói này và có thể làm hỏng thư viện có sẵn của Kaggle.

1. **Không** chạy `pip install -r requirements.txt` trên Kaggle. Dùng thư viện có sẵn (torch, numpy, pandas, pyarrow, scikit-learn, matplotlib, pyyaml); thiếu gói nào thì `pip install <gói>` riêng gói đó. Gradio chỉ dùng cho demo trên máy, không cần trên Kaggle.
2. Ô đầu notebook in phiên bản và ghi vào log pilot (để báo cáo có số):
   ```python
   import sys, torch, pandas, pyarrow, sklearn, numpy
   print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
         pandas.__version__, pyarrow.__version__, sklearn.__version__, numpy.__version__)
   ```
3. Dữ liệu: đưa `data/processed/reviews.parquet` lên Kaggle dưới dạng **dataset riêng** (không commit vào repo). Sau khi nạp, kiểm `content_hash` khớp `experiments/results/data_prep/prepare_stats.json`:
   ```python
   int(pd.util.hash_pandas_object(df, index=False).sum()) == 1029824945594009414
   ```
4. GloVe: dùng dataset GloVe 6B 100d có sẵn trên Kaggle, không tải 822 MB.
5. Kết quả (`history.csv`, `metrics.json`, hình) tải về máy rồi commit vào `experiments/` theo [TASKS.md](../TASKS.md); checkpoint không commit.

## Detailed plan / gotchas

- Cố định seed (`torch`, `numpy`, `random`) qua `src/utils/seed.py`; split không đổi theo seed.
- Không đánh giá test trong vòng train.
- Ghi phần cứng, thời gian mỗi epoch và tổng thời gian train.
- Nếu epoch tốt nhất = epoch cuối (chưa hội tụ): tăng trần epoch rồi chạy lại, ghi vào progress.

## Verification gates

- [ ] Chạy 1 epoch trên tập con 500 mẫu không lỗi; `history.csv` có đủ cột.
- [ ] Load `best.pt` cho đúng val acc đã log.
- [ ] Mỗi cấu hình có hình learning curve 3 seed.
- [ ] Chạy lại cùng config + seed lệch ≤ 1% (M6).

## Links

- Progress tracking: [../progress/TRAINING_STATUS.md](../progress/TRAINING_STATUS.md)
- Related: [OVERVIEW.md](../OVERVIEW.md)
