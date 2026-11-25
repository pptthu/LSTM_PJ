# DEMO_REPORT_STATUS.md — Tiến độ: Demo, README, báo cáo & slide

- **Motivation/Background**: Theo dõi nhanh phase Demo, README, báo cáo & slide mà không cần đọc lại toàn bộ đặc tả.
- **Purpose**: Ghi việc đã xong kèm bằng chứng, blocker, quyết định và bước tiếp theo.
- **Overview Pipeline**: Log → blocker → quyết định → bước tiếp theo.
- **Detailed Plan**: Header; Checklist; Log; Blockers; Decisions; Next step; Links.
- **References**: [../phases/DEMO_REPORT.md](../phases/DEMO_REPORT.md), [../OVERVIEW.md](../OVERVIEW.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T10:00:45+07:00

---

## Header

- **Title:** Demo, README, báo cáo & slide
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-23
- **Status:** In Progress (B6 code + test xong; chờ checkpoint thật từ Colab)
- **Owner:** Bảo (demo Gradio); Thư (README, notebook, báo cáo PDF tiếng Việt + tiếng Anh, slide)
- **Hạn:** T4 25/11 — nộp trước 14:00
- **Phase doc:** [../phases/DEMO_REPORT.md](../phases/DEMO_REPORT.md)

## Checklist

Chỉ đánh dấu `[x]` khi có bằng chứng (lệnh, đường dẫn output, commit).

- [x] Đọc xong đặc tả phase
- [x] Code chạy được trên tập con (`pytest tests/test_predict_demo.py`)
- [ ] Verification gates trên dữ liệu thật (sau khi có checkpoint Colab) đều pass
- [ ] Cập nhật tài liệu liên quan (README.md, REPORT_CHECKLIST.md)

## Log

- 2026-11-05: Tạo file.
- 2026-11-25: Hoàn thiện `src/predict.py` (làm sạch `<br />`, hàm `predict(text)`, occlusion theo batch trên CPU < 1s) và `src/demo/app.py` (giao diện Gradio nạp ví dụ từ `handwritten_30.csv` & `sample_100.csv`).

## Blockers

- Chưa có.

## Decisions

- Chưa có.

## Next step

- Bắt đầu bước 1 trong [../phases/DEMO_REPORT.md](../phases/DEMO_REPORT.md).

## Links

- Phase doc: [../phases/DEMO_REPORT.md](../phases/DEMO_REPORT.md)
- Experiments: [../experiments/README.md](../experiments/README.md)
