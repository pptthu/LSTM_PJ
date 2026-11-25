# EVAL_STATUS.md — Tiến độ: Đánh giá, phân tích lỗi & giải thích mô hình

- **Motivation/Background**: Theo dõi nhanh phase Đánh giá, phân tích lỗi & giải thích mô hình mà không cần đọc lại toàn bộ đặc tả.
- **Purpose**: Ghi việc đã xong kèm bằng chứng, blocker, quyết định và bước tiếp theo.
- **Overview Pipeline**: Log → blocker → quyết định → bước tiếp theo.
- **Detailed Plan**: Header; Checklist; Log; Blockers; Decisions; Next step; Links.
- **References**: [../phases/EVAL.md](../phases/EVAL.md), [../OVERVIEW.md](../OVERVIEW.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T10:00:45+07:00

---

## Header

- **Title:** Đánh giá, phân tích lỗi & giải thích mô hình
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-23
- **Status:** In Progress (D8 code + test xong; chờ kết quả chạy Colab)
- **Owner:** Đại (chạy test một lần, metric, McNemar, xuất file dự đoán); Thư (phân tích lỗi, giải thích mô hình, giới hạn)
- **Hạn:** T3 24/11 tối
- **Phase doc:** [../phases/EVAL.md](../phases/EVAL.md)

## Checklist

Chỉ đánh dấu `[x]` khi có bằng chứng (lệnh, đường dẫn output, commit).

- [x] Đọc xong đặc tả phase
- [x] Code chạy được trên tập con (`pytest tests/test_eval.py`)
- [ ] Verification gates trên dữ liệu thật (Colab) đều pass
- [ ] Cập nhật tài liệu liên quan (experiments/README.md, REPORT_CHECKLIST.md)

## Log

- 2026-11-05: Tạo file.
- 2026-11-25: Hoàn thiện `src/eval/metrics.py` (kèm McNemar chính xác nhị thức), `src/evaluate.py` (tương thích cấu trúc baseline của B5, báo cáo thêm kết quả bỏ 123 review test trùng train, CLI đầy đủ) và nhánh `--eval-test` trong `src/models/baseline.py`.

## Blockers

- Chưa có.

## Decisions

- Chưa có.

## Next step

- Bắt đầu bước 1 trong [../phases/EVAL.md](../phases/EVAL.md).

## Links

- Phase doc: [../phases/EVAL.md](../phases/EVAL.md)
- Experiments: [../experiments/README.md](../experiments/README.md)
