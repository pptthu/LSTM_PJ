# DATA_PREP_STATUS.md — Tiến độ: Chuẩn bị dữ liệu & split

- **Motivation/Background**: Theo dõi nhanh phase Chuẩn bị dữ liệu & split mà không cần đọc lại toàn bộ đặc tả.
- **Purpose**: Ghi việc đã xong kèm bằng chứng, blocker, quyết định và bước tiếp theo.
- **Overview Pipeline**: Log → blocker → quyết định → bước tiếp theo.
- **Detailed Plan**: Header; Checklist; Log; Blockers; Decisions; Next step; Links.
- **References**: [../phases/DATA_PREP.md](../phases/DATA_PREP.md), [../OVERVIEW.md](../OVERVIEW.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T19:40:00+07:00

---

## Header

- **Title:** Chuẩn bị dữ liệu & split
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-23
- **Status:** In Progress — B1–B4 xong; B5 làm sau
- **Owner:** Bảo
- **Hạn:** T2 23/11 tối — giao `reviews.parquet` cho Đại
- **Phase doc:** [../phases/DATA_PREP.md](../phases/DATA_PREP.md)

## Checklist

Chỉ đánh dấu `[x]` khi có bằng chứng (lệnh, đường dẫn output, commit).

- [x] Đọc xong đặc tả phase
- [x] Code chạy được trên tập con (`python -m src.data.prepare` trên 50k review thật)
- [x] Verification gates trong phase doc đều pass (xem Log 2026-11-23 B1)
- [x] Cập nhật tài liệu liên quan (DATA.md §3–§4)
- [x] B3: EDA sau loại trùng + DATA.md §5 (nhánh `feat/data-eda`)

## Log

- 2026-11-05: Tạo file.
- 2026-11-23: Dựng repo (nhánh `chore/repo-setup`): chép từ Lab 3 phần tải/đọc IMDb, làm sạch, hash, file split, hàm EDA + test (`python -m pytest`: 17 passed). Chạy thử `python -m src.data.eda` trên dữ liệu thật: 22.499 / 2.501 / 25.000 review, 123 review test trùng train. Kết quả chạy thử không commit — Bảo chạy lại sau khi loại trùng.
- 2026-11-23 (B1, nhánh `feat/data-prepare`): `src/data/prepare.py` xong. `python -m src.data.download` (tải 80 MB, đọc 50k file, lưu `data/raw/imdb_{train,test}.parquet`) rồi `python -m src.data.prepare` →
  train 22.499 → **22.304**, val 2.501 → **2.473**, test **25.000** (không đổi). Bỏ 87+11 review trùng trong train/val, 106+17 review trùng với test, 2 review của phim `tt0406816` (phim này có 23 review test). Lệnh: `python -m src.data.prepare`; số: `experiments/results/data_prep/prepare_stats.json`.
  Kiểm: 0 hash chung train/val–test, 0 `movie_id` chung giữa 3 split, 0 nhãn sai, `label` khớp `rating` 100%. Chạy lại 2 lần → sha256 `e6245753…afb7cd` giống hệt.
- 2026-11-23 (B2, cùng nhánh): `tests/test_prepare.py` 15 test — loại trùng trong/giữa split, test không bị sửa, rò rỉ theo phim, nhãn, schema, tất định, đầu vào rỗng/thiếu cột, chạy end-to-end trên cây aclImdb giả. `python -m pytest` → **32 passed**.
- 2026-11-23 (B3, nhánh `feat/data-eda`): thêm vào `src/data/eda.py` phần từ phổ biến từng lớp (có/không stopword), phủ định (tỷ lệ theo split × nhãn, dấu hiệu, cụm 2 từ), ca biên (≤10 từ, ≥300 từ), kiểm chất lượng. `main` đọc `data/processed/reviews.parquet` thay vì dữ liệu gốc. `python -m src.data.eda` (~60 s) ghi 8 CSV + 6 hình vào `experiments/results/eda/`, `experiments/plots/eda/`; số điền vào DATA.md §5. `python -m pytest` → 24 passed trên nhánh này (chưa có test của B1–B2; sau khi merge PR #4 là 39).
- 2026-11-23 (B3, kiểm): 0 review trùng trong train/val, test còn 201 bản thừa (giữ theo quy tắc), 0 hash/movie chung giữa split, 0 nhãn sai; 10 review còn `<i>`; 58,7% review gốc có `<br />`.
- 2026-11-23 (B4, nhánh `feat/data-sample`): `src/data/sample.py` chọn 100 review **chỉ từ test**, tất định (`Random(42)`), 50 âm / 50 dương; nhóm: rating 4 (12), rating 7 (12), phủ định (8/lớp), dài ≥500 từ (10/lớp), ngắn ≤20 từ (5/lớp), ngẫu nhiên phần còn lại. `python -m src.data.sample` ghi `data/sample/sample_100.csv` (cột `uid, text, label, case_type, rating, source_file`); `tests/test_sample.py` 7 test; `python -m pytest` → 24 passed trên nhánh này.

- 2026-11-24 (fix theo review PR #4, nhánh `fix/data-test-dup-list`): `prepare.py` ghi `experiments/results/data_prep/test_uids_dup_train.txt` (**123 uid** test trùng train, sắp theo uid; số dòng trong `prepare_stats.json`); thêm `content_hash` = **1029824945594009414** (so được giữa các máy, thay cho sha256 file); `check_contract` **raise** khi `movie_id` chung giữa các split; bản commit của `prepare_stats.json` bỏ `generated_at` (chạy lại không làm file đổi). DATA.md §3–§4: ghi rõ content_hash vs sha256, quy tắc 4 "Thư chốt 2026-11-23", train 22.304 khác train Lab 3 22.499 ("cùng split, train đã loại trùng"). `python -m pytest` → **35 passed**.

- 2026-11-24 (B3, cập nhật theo review PR #9): `quality_issues` thêm kiểm ô NaN ở cột bắt buộc và ký tự `U+FFFD` (đều 0 trên dữ liệu thật); test truy cập bảng bằng MultiIndex 3 khoá (`scope, split, check`) — sửa lỗi `ValueError: truth value of a Series is ambiguous` trên pandas 3.0.3 (tái hiện và kiểm lại: 8/8 test EDA pass trên pandas 3.0.3, 43 pass trên pandas 3.1.1). Số EDA đã đối chiếu độc lập (median/P90/P95/% cắt, phủ định, top từ, trùng lặp) và chạy 2 lần cho CSV giống hệt.
- 2026-11-24 (nhánh `fix/data-reproducibility`): ghim `pandas==3.1.1`, `pyarrow==26.0.1`, `scikit-learn==1.9.1` trong `requirements.txt` (cài lại trong venv sạch OK, numpy 2.5.5); `_display_path` đổi sang `.as_posix()` (Linux/Windows cùng chuỗi). Chạy `prepare` 2 lần: `prepare_stats.json` giống hệt từng byte, parquet cùng sha256, `content_hash` = 1029824945594009414. `python -m pytest` → 35 passed; venv sạch: 20 passed (`test_prepare` + `test_split_file`).

## Blockers

- Chưa có.

## Decisions

- Chưa có.

## Next step

- B5: baseline (xem [../phases/BASELINE.md](../phases/BASELINE.md)).


## Links

- Phase doc: [../phases/DATA_PREP.md](../phases/DATA_PREP.md)
- Experiments: [../experiments/README.md](../experiments/README.md)
