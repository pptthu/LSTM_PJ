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
- **Status:** Done
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

- B4: `data/sample/sample_100.csv`; B5: baseline (xem [../phases/BASELINE.md](../phases/BASELINE.md)).

- 2026-11-23 (B3, nhánh `feat/data-eda`): thêm vào `src/data/eda.py` phần từ phổ biến từng lớp (có/không stopword), phủ định (tỷ lệ theo split × nhãn, dấu hiệu, cụm 2 từ), ca biên (≤10 từ, ≥300 từ), kiểm chất lượng. `main` đọc `data/processed/reviews.parquet` thay vì dữ liệu gốc. `python -m src.data.eda` (~60 s) ghi 8 CSV + 6 hình vào `experiments/results/eda/`, `experiments/plots/eda/`; số điền vào DATA.md §5. `python -m pytest` → 24 passed trên nhánh này (chưa có test của B1–B2; sau khi merge PR #4 là 39).
- 2026-11-23 (B3, kiểm): 0 review trùng trong train/val, test còn 201 bản thừa (giữ theo quy tắc), 0 hash/movie chung giữa split, 0 nhãn sai; 10 review còn `<i>`; 58,7% review gốc có `<br />`.

## Links

- Phase doc: [../phases/DATA_PREP.md](../phases/DATA_PREP.md)
- Experiments: [../experiments/README.md](../experiments/README.md)
