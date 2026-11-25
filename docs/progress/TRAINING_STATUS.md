# TRAINING_STATUS.md — Tiến độ: Huấn luyện

- **Motivation/Background**: Theo dõi nhanh phase Huấn luyện mà không cần đọc lại toàn bộ đặc tả.
- **Purpose**: Ghi việc đã xong kèm bằng chứng, blocker, quyết định và bước tiếp theo.
- **Overview Pipeline**: Log → blocker → quyết định → bước tiếp theo.
- **Detailed Plan**: Header; Checklist; Log; Blockers; Decisions; Next step; Links.
- **References**: [../phases/TRAINING.md](../phases/TRAINING.md), [../OVERVIEW.md](../OVERVIEW.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T10:00:45+07:00

---

## Header

- **Title:** Huấn luyện
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-23
- **Status:** In Progress (D4 code + test xong; chờ chạy D5–D7 trên Colab)
- **Owner:** Đại
- **Hạn:** T3 24/11 — mốc dừng 14:00: chưa xong 5 cấu hình × 3 seed thì bỏ attention pooling
- **Phase doc:** [../phases/TRAINING.md](../phases/TRAINING.md)

## Checklist

Chỉ đánh dấu `[x]` khi có bằng chứng (lệnh, đường dẫn output, commit).

- [x] Đọc xong đặc tả phase
- [x] Code chạy được trên tập con (`pytest tests/test_train_monitor.py`)
- [ ] Verification gates trên dữ liệu thật (Colab) đều pass
- [ ] Đủ 3 seed (42, 123, 2025) cho mỗi cấu hình, có `summary.json`
- [ ] Cập nhật tài liệu liên quan (experiments/README.md)

## Log

- 2026-11-05: Tạo file.
- 2026-11-25: Hoàn thiện `src/train.py`, `src/training/monitor.py`, `src/utils/seed.py`, 6 file `configs/*.yaml` và `notebooks/colab_d1_d8.ipynb`. Ghi `history.csv` đồng thời vào `runs/` và `results/`, lấy mẫu cân bằng lớp khi chạy thử `sample_size`, lọc tham số `requires_grad` cho M4a, tính `macro_f1`. Chờ chạy thực nghiệm dữ liệu thật trên Colab.

## Blockers

- Chưa có.

## Decisions

- Chưa có.

## Next step

- Bắt đầu bước 1 trong [../phases/TRAINING.md](../phases/TRAINING.md).

## Links

- Phase doc: [../phases/TRAINING.md](../phases/TRAINING.md)
- Experiments: [../experiments/README.md](../experiments/README.md)
