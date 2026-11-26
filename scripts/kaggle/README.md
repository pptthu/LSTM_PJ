# scripts/kaggle — script đã dùng để chạy trên Kaggle

Hai kernel Kaggle (GPU Tesla T4) tạo ra toàn bộ kết quả mô hình trong `experiments/results/`. Kaggle giữ output của từng lần chạy ở tab **Output** của kernel (tài khoản `pptt1103`, chế độ private).

| Thư mục | Kernel Kaggle | Làm gì | Kết quả trong repo |
| :--- | :--- | :--- | :--- |
| `d1_d8/` | `pptt1103/lstm-pj-d1-d8`, **Version 2** (26/11/2026, 06:58–08:23) | Dữ liệu (B1) → GloVe (D3) → pilot chốt `max_len` (D5) → 6 mô hình × 3 seed (D6) → mở rộng A (D7) → test 1 lần + McNemar (D8) | `experiments/results/**`, `experiments/plots/{training,eval}/` |
| `m6/` | `pptt1103/lstm-pj-m6-repro` | M6: clone repo sạch, chạy đúng lệnh README §4, train lại GRU seed 42, so với số đã commit | `experiments/results/repro/m6_gru_seed42.json` |

Version 1 của `lstm-pj-d1-d8` (26/11, 03:05–04:11) là **lần chạy 1** dùng `max_len` = 250 mặc định — không dùng, xem `docs/experiments/README.md` §3.3.

## `d1_d8/pipeline.py`

- Clone repo ở commit `e1f38ec` (code train/eval sau PR #18), đặt `CUDA_VISIBLE_DEVICES=0`.
- Mỗi giai đoạn ghi trạng thái vào `stages.json` và chép `experiments/results`, `plots` ra `/kaggle/working` ngay, để lỗi giữa chừng không mất phần đã chạy; chạy lại thì bỏ qua giai đoạn đã `ok`.
- D5 chốt `max_len` theo quy tắc ghi sẵn (val acc cao nhất; chênh < 0,5 điểm thì chọn giá trị nhỏ hơn) rồi **ghi đè `max_len` lúc chạy** cho D6–D8. Sau lần chạy, `configs/*.yaml` trong repo được sửa thành 400 cho khớp (PR #19).
- Cần dataset Kaggle `ktgiahieu/glove6b100d` (GloVe 6B 100d) và bật Internet (tải aclImdb, clone repo).
- `LSTMPJ_SMOKE=1 LSTMPJ_REPO=<repo> LSTMPJ_OUT=<thư mục>`: chạy thử 1 epoch trên máy CPU với dữ liệu nhỏ.

## Chạy lại

```bash
pip install kaggle            # cần token Kaggle của bạn ở ~/.kaggle/ (không commit)
# sửa "id" trong kernel-metadata.json thành <tài-khoản-của-bạn>/<tên-kernel>
kaggle kernels push -p scripts/kaggle/d1_d8 --accelerator NvidiaTeslaT4
kaggle kernels status <tài-khoản>/<tên-kernel>
kaggle kernels output <tài-khoản>/<tên-kernel> -p <thư mục>
```

Thời gian lần chạy 2: khoảng 1 giờ 25 phút GPU. Output khoảng 1,2 GB (checkpoint 6 mô hình, `predictions_*.csv`); repo chỉ commit các file nhỏ (số, bảng, hình), không commit checkpoint và `predictions_*.csv`.
