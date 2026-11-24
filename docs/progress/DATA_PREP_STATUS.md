# DATA_PREP_STATUS.md — Tiến độ: Chuẩn bị dữ liệu & split

- **Motivation/Background**: Theo dõi nhanh phase Chuẩn bị dữ liệu & split mà không cần đọc lại toàn bộ đặc tả.
- **Purpose**: Ghi việc đã xong kèm bằng chứng, blocker, quyết định và bước tiếp theo.
- **Overview Pipeline**: Log → blocker → quyết định → bước tiếp theo.
- **Detailed Plan**: Header; Checklist; Log; Blockers; Decisions; Next step; Links.
- **References**: [../phases/DATA_PREP.md](../phases/DATA_PREP.md), [../OVERVIEW.md](../OVERVIEW.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T18:20:00+07:00

---

## Header

- **Title:** Chuẩn bị dữ liệu & split
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-23
- **Status:** In Progress
- **Owner:** Bảo
- **Hạn:** T2 23/11 tối — giao `reviews.parquet` cho Đại
- **Phase doc:** [../phases/DATA_PREP.md](../phases/DATA_PREP.md)

## Checklist

Chỉ đánh dấu `[x]` khi có bằng chứng (lệnh, đường dẫn output, commit).

- [x] Đọc xong đặc tả phase
- [x] Code chạy được trên tập con (`python -m src.data.prepare` trên 50k review thật)
- [x] Verification gates trong phase doc đều pass (xem Log 2026-11-23 B1)
- [x] Cập nhật tài liệu liên quan (DATA.md §3–§4)
- [ ] B3: EDA sau loại trùng + DATA.md §5 (nhánh `feat/data-eda`)

## Log

- 2026-11-05: Tạo file.
- 2026-11-23: Dựng repo (nhánh `chore/repo-setup`): chép từ Lab 3 phần tải/đọc IMDb, làm sạch, hash, file split, hàm EDA + test (`python -m pytest`: 17 passed). Chạy thử `python -m src.data.eda` trên dữ liệu thật: 22.499 / 2.501 / 25.000 review, 123 review test trùng train. Kết quả chạy thử không commit — Bảo chạy lại sau khi loại trùng.
- 2026-11-23 (B1, nhánh `feat/data-prepare`): `src/data/prepare.py` xong. `python -m src.data.download` (tải 80 MB, đọc 50k file, lưu `data/raw/imdb_{train,test}.parquet`) rồi `python -m src.data.prepare` →
  train 22.499 → **22.304**, val 2.501 → **2.473**, test **25.000** (không đổi). Bỏ 87+11 review trùng trong train/val, 106+17 review trùng với test, 2 review của phim `tt0406816` (phim này có 23 review test). Lệnh: `python -m src.data.prepare`; số: `experiments/results/data_prep/prepare_stats.json`.
  Kiểm: 0 hash chung train/val–test, 0 `movie_id` chung giữa 3 split, 0 nhãn sai, `label` khớp `rating` 100%. Chạy lại 2 lần → sha256 `e6245753…afb7cd` giống hệt.
- 2026-11-23 (B2, cùng nhánh): `tests/test_prepare.py` 15 test — loại trùng trong/giữa split, test không bị sửa, rò rỉ theo phim, nhãn, schema, tất định, đầu vào rỗng/thiếu cột, chạy end-to-end trên cây aclImdb giả. `python -m pytest` → **32 passed**.

- 2026-11-24 (fix theo review PR #4, nhánh `fix/data-test-dup-list`): `prepare.py` ghi `experiments/results/data_prep/test_uids_dup_train.txt` (**123 uid** test trùng train, sắp theo uid; số dòng trong `prepare_stats.json`); thêm `content_hash` = **1029824945594009414** (so được giữa các máy, thay cho sha256 file); `check_contract` **raise** khi `movie_id` chung giữa các split; bản commit của `prepare_stats.json` bỏ `generated_at` (chạy lại không làm file đổi). DATA.md §3–§4: ghi rõ content_hash vs sha256, quy tắc 4 "Thư chốt 2026-11-23", train 22.304 khác train Lab 3 22.499 ("cùng split, train đã loại trùng"). `python -m pytest` → **35 passed**.

## Blockers

- Chưa có.

## Decisions

- Chưa có.

## Next step

- B3: chạy EDA trên `data/processed/reviews.parquet` (nhánh `feat/data-eda`), điền DATA.md §5; rồi B4 `sample_100.csv`.

## Links

- Phase doc: [../phases/DATA_PREP.md](../phases/DATA_PREP.md)
- Experiments: [../experiments/README.md](../experiments/README.md)
