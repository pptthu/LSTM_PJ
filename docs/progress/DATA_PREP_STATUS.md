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
- **Status:** To Do
- **Owner:** Bảo
- **Hạn:** T2 23/11 tối — giao `reviews.parquet` cho Đại
- **Phase doc:** [../phases/DATA_PREP.md](../phases/DATA_PREP.md)

## Checklist

Chỉ đánh dấu `[x]` khi có bằng chứng (lệnh, đường dẫn output, commit).

- [ ] Đọc xong đặc tả phase
- [ ] Code chạy được trên tập con
- [ ] Verification gates trong phase doc đều pass
- [ ] Cập nhật tài liệu liên quan (DATA.md §3–§4)

## Log

- 2026-11-05: Tạo file.
- 2026-11-23: Dựng repo (nhánh `chore/repo-setup`): chép từ Lab 3 phần tải/đọc IMDb, làm sạch, hash, file split, hàm EDA + test (`python -m pytest`: 17 passed). Chạy thử `python -m src.data.eda` trên dữ liệu thật: 22.499 / 2.501 / 25.000 review, 123 review test trùng train. Kết quả chạy thử không commit — Bảo chạy lại sau khi loại trùng.

## Blockers

- Chưa có.

## Decisions

- Chưa có.

## Next step

- B4: `python -m src.data.sample` → `data/sample/sample_100.csv`; kiểm bằng `tests/test_sample.py`.
- B5: baseline (xem [../phases/BASELINE.md](../phases/BASELINE.md)).

- 2026-11-23 (B4, nhánh `feat/data-sample`): `src/data/sample.py` chọn 100 review **chỉ từ test**, tất định (`Random(42)`), 50 âm / 50 dương; nhóm: rating 4 (12), rating 7 (12), phủ định (8/lớp), dài ≥500 từ (10/lớp), ngắn ≤20 từ (5/lớp), ngẫu nhiên phần còn lại. `python -m src.data.sample` ghi `data/sample/sample_100.csv` (cột `uid, text, label, case_type, rating, source_file`); `tests/test_sample.py` 7 test; `python -m pytest` → 24 passed trên nhánh này.

## Links

- Phase doc: [../phases/DATA_PREP.md](../phases/DATA_PREP.md)
- Experiments: [../experiments/README.md](../experiments/README.md)
