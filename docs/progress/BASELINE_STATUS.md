# BASELINE_STATUS.md — Tiến độ: Baseline Majority + TF-IDF + LR

- **Motivation/Background**: Theo dõi nhanh phase Baseline Majority + TF-IDF + LR mà không cần đọc lại toàn bộ đặc tả.
- **Purpose**: Ghi việc đã xong kèm bằng chứng, blocker, quyết định và bước tiếp theo.
- **Overview Pipeline**: Log → blocker → quyết định → bước tiếp theo.
- **Detailed Plan**: Header; Checklist; Log; Blockers; Decisions; Next step; Links.
- **References**: [../phases/BASELINE.md](../phases/BASELINE.md), [../OVERVIEW.md](../OVERVIEW.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-24T11:30:00+07:00

---

## Header

- **Title:** Baseline Majority + TF-IDF + LR
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-23
- **Status:** Done
- **Owner:** Bảo
- **Hạn:** T2 23/11 tối
- **Phase doc:** [../phases/BASELINE.md](../phases/BASELINE.md)

## Checklist

Chỉ đánh dấu `[x]` khi có bằng chứng (lệnh, đường dẫn output, commit).

- [x] Đọc xong đặc tả phase
- [x] Code chạy được trên tập con (`python -m src.models.baseline` trên 22.304 train / 2.473 val)
- [x] Verification gates trong phase doc đều pass (xem Log)
- [x] Cập nhật tài liệu liên quan (experiments/README.md, EXP_01_BASELINE.md, MODELS.md)

## Log

- 2026-11-05: Tạo file.
- 2026-11-23 (B5, nhánh `feat/baseline`): `src/models/baseline.py` xong. `python -m src.models.baseline` (~24 s): M0a majority = 1 → val acc **0,4994**, macro-F1 0,3331, AUC 0,5. M0b TF-IDF 1–2 gram (`min_df=2`, 398.264 đặc trưng) + LR quét `C` 0,01/0,1/1/10 → chọn **C=10**: val acc **0,8985**, macro-F1 0,8985, ROC-AUC **0,9623**; confusion matrix `[[1093,145],[106,1129]]`; vectorize 9,5 s, train 4,8 s, inference 0,0022 ms/review. Test **không đọc** (chạy ở D8). Kết quả: `experiments/results/baseline/{metrics.json,config.json,c_grid.csv,predictions_val.csv}`; checkpoint `experiments/checkpoints/baseline_tfidf_lr.joblib` (không commit). `tests/test_baseline.py` 6 test; `python -m pytest` → 23 passed trên nhánh này.
- 2026-11-23 (B5, kiểm thêm): thử `C` = 20/30/50/100 → val acc giảm (0,8969/0,8961/0,8957/0,8953), giữ `C=10`.

- 2026-11-24 (mở rộng lưới `C`, theo review PR #11): lưới cũ `{0,01; 0,1; 1; 10}` dừng ở cận trên → chạy lại với `{0,01; 0,1; 1; 10; 30; 100}`. Val acc: 0,7820 / 0,8346 / 0,8799 / **0,8985** / 0,8961 / 0,8953 → cực đại thật tại `C=10`. Train acc lên 0,9993 (`C=10`) và 1,0000 (`C=30/100`), khoảng cách train–val nới 3,6 → 10,5 điểm: dấu hiệu overfitting khi `C` lớn. Giữ `C=10`. Kết quả + cột train accuracy trong `experiments/results/baseline/c_grid.csv`, `metrics.json`; EXP_01 cập nhật bảng và nhận xét.

## Blockers

- Chưa có.

## Decisions

- Chọn `C` theo val accuracy (hoà → macro-F1 → `C` nhỏ hơn) — tất định; `C=10` là điểm tốt nhất trên val, không phải cận lưới tuỳ tiện vì đã kiểm 20–100.
- Dự đoán val ghi `uid, text, label, rating, prob, pred` để Thư/Đại chạy McNemar với LSTM (D8).
- Không đánh giá test ở bước này; test chạy một lần ở D8.

## Next step

- D8 (Đại): chạy test một lần cho LSTM + baseline, so McNemar trên `predictions_*.csv`.
- B6 (Bảo): demo Gradio sau khi có checkpoint D7.

## Links

- Phase doc: [../phases/BASELINE.md](../phases/BASELINE.md)
- Experiments: [../experiments/README.md](../experiments/README.md)
