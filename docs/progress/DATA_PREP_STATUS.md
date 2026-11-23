# DATA_PREP_STATUS.md — Tiến độ: Chuẩn bị dữ liệu & split

- **Motivation/Background**: Theo dõi nhanh phase Chuẩn bị dữ liệu & split mà không cần đọc lại toàn bộ đặc tả.
- **Purpose**: Ghi việc đã xong kèm bằng chứng, blocker, quyết định và bước tiếp theo.
- **Overview Pipeline**: Log → blocker → quyết định → bước tiếp theo.
- **Detailed Plan**: Header; Checklist; Log; Blockers; Decisions; Next step; Links.
- **References**: [../phases/DATA_PREP.md](../phases/DATA_PREP.md), [../OVERVIEW.md](../OVERVIEW.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T19:05:00+07:00

---

## Header

- **Title:** Chuẩn bị dữ liệu & split
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-23
- **Status:** In Progress — B1–B3 xong; B4–B5 làm sau
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

## Blockers

- Chưa có.

## Decisions

- Chưa có.

## Next step

- B4: `data/sample/sample_100.csv`; B5: baseline (xem [../phases/BASELINE.md](../phases/BASELINE.md)).


## Links

- Phase doc: [../phases/DATA_PREP.md](../phases/DATA_PREP.md)
- Experiments: [../experiments/README.md](../experiments/README.md)
