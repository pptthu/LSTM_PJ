# DATA_PREP.md — Chuẩn bị dữ liệu & split

- **Motivation/Background**: Mọi phase sau dùng chung một split cố định; split sai làm sai mọi số đo.
- **Purpose**: Tải IMDb, làm sạch, loại trùng, chia train/val theo phim, EDA, tạo dữ liệu mẫu.
- **Overview Pipeline**: Phạm vi → đặc tả → artifact → kiểm chứng.
- **Detailed Plan**: Header; Goals; Input/Output; How to do it; Detailed plan / gotchas; Verification gates; Links.
- **References**: [OVERVIEW.md](../OVERVIEW.md), [PURPOSE.md](../PURPOSE.md) §5, §9, [DATA.md](../DATA.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T21:10:00+07:00

---

## Header

- **Title:** Chuẩn bị dữ liệu & split
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-23
- **Description:** Tải IMDb, làm sạch, loại trùng, chia train/val theo phim, EDA, tạo dữ liệu mẫu.
- **Status:** Done — B1–B4 hoàn tất (PR #4, #9, #10)
- **Owner:** Bảo
- **Hạn:** T2 23/11 tối — giao `reviews.parquet` cho Đại

## Goals / Purpose

- Xong: `data/processed/reviews.parquet` (đã loại trùng, đúng split Lab 3) giao cho Đại; bảng trùng lặp & rò rỉ trong [DATA.md](../DATA.md) §4 đã điền số thật; EDA trên train/val/test xong; `data/sample/sample_100.csv` có đủ dòng.
- Không làm: tokenize/vocab cho mô hình (thuộc MODEL — Đại). 30 câu tự viết (`data/sample/handwritten_30.csv`) do Thư viết.

## Input / Output

- **Input:** `aclImdb_v1.tar.gz` từ Stanford; `data/splits/train_val_split.json` (đã commit, từ Lab 3).
- **Output:**
  - `data/processed/reviews.parquet` (không commit), cột `uid, split, text, text_clean, label, rating, movie_id`; `split` ∈ train/val/test.
  - `experiments/results/eda/*.csv`, `experiments/plots/eda/*.png` (commit).
  - `data/sample/sample_100.csv` (cột `text,label,case_type,rating,uid`).

## Đã có sẵn khi dựng repo (2026-11-23, chép từ Lab 3)

| Việc | Code | Kiểm |
| :--- | :--- | :--- |
| Tải + giải nén, kiểm 4 × 12.500 file; bộ đệm `data/raw/imdb_<split>.parquet` | `src/data/download.py` (`python -m src.data.download`) | `tests/test_data.py` |
| Đọc `rating` từ tên file, mã phim từ `urls_*.txt` (dòng i = review id i) | `src/data/download.py` | `tests/test_data.py` |
| Làm sạch `<br />`, hash nội dung | `src/data/clean.py` | `tests/test_data.py` |
| Split train/val theo phim (file Lab 3) + `apply_split` | `src/data/split.py`, `data/splits/train_val_split.json` | `tests/test_split_file.py` |
| Hàm EDA: độ dài, % bị cắt ở 200/400/600, review trùng, phủ định, 3 hình | `src/data/eda.py` (`python -m src.data.eda`) | `tests/test_data.py` |

Máy Thư đã có sẵn `data/raw/imdb_{train,test}.parquet` (chép từ Lab 3) nên không phải tải lại; máy khác chạy `python -m src.data.download` (tải 84 MB, đọc 50k file lần đầu khá lâu trên Windows).

## Việc của Bảo

1. Viết `src/data/prepare.py`: đọc 2 split → `apply_split` → làm sạch → **loại trùng** theo [DATA.md](../DATA.md) §4 → ghi `data/processed/reviews.parquet`. In và ghi số đã loại.
2. Test cho bước loại trùng (dùng cây aclImdb giả trong `tests/conftest.py`): không còn hash chung giữa train/val và test; test vẫn đủ số dòng.
3. Chạy EDA trên dữ liệu đã loại trùng; thêm "từ phổ biến của từng lớp" (cột, không dùng biểu đồ tròn); điền [DATA.md](../DATA.md) §3–§5.
4. Chọn `sample_100.csv` từ test (có ca khó: phủ định, dài, rating 4/7).
5. Baseline: xem [BASELINE.md](BASELINE.md).

## Detailed plan / gotchas

- Dòng thứ i của `urls_pos.txt` ứng với review có `id = i` trong `pos/`.
- Làm sạch `<br />` trước khi hash để bắt được bản trùng khác thẻ HTML.
- Không xoá review trong test; trùng train/val–test thì bỏ ở train/val.
- Không tạo split mới: luôn dùng `data/splits/train_val_split.json`.
- Không đọc test khi chọn siêu tham số; EDA trên test chỉ để mô tả.

## Verification gates

- [x] train + val = 25.000 − số trùng đã loại; test = 25.000 — 22.304 + 2.473 = 24.777; test 25.000.
- [x] 0 mã phim chung giữa train và val, và giữa train và test — sau khi bỏ 2 review train của `tt0406816` (dữ liệu thật có 1 phim lọt cả train và test).
- [x] 0 hash chung giữa train/val và test — `check_contract` raise nếu khác 0.
- [x] Chạy lại `prepare.py` cho file giống hệt — sha256 `e6245753…afb7cd` cả 2 lần.
- [x] `python -m pytest` pass — 32 passed (15 test mới ở `tests/test_prepare.py`).

## Links

- Progress tracking: [../progress/DATA_PREP_STATUS.md](../progress/DATA_PREP_STATUS.md)
- Related: [OVERVIEW.md](../OVERVIEW.md), [DATA.md](../DATA.md)
