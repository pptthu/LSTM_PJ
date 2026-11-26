# DEMO_REPORT.md — Demo, README, báo cáo & slide

- **Motivation/Background**: Môn học yêu cầu demo, hướng dẫn chạy và báo cáo đủ 7 nội dung; đăng ký cam kết demo CPU < 1 giây/review (M5).
- **Purpose**: Demo Gradio + notebook demo, README, báo cáo cuối kỳ và slide.
- **Overview Pipeline**: Phạm vi → đặc tả → artifact → kiểm chứng.
- **Detailed Plan**: Header; Goals; Input/Output; How to do it; Detailed plan / gotchas; Verification gates; Links.
- **References**: [OVERVIEW.md](../OVERVIEW.md), [PURPOSE.md](../PURPOSE.md) §4 (M5), §8.
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-26T10:00:00+07:00

---

## Header

- **Title:** Demo, README, báo cáo & slide
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-26
- **Description:** Demo Gradio + notebook demo, README, báo cáo cuối kỳ và slide.
- **Status:** In Progress (demo xong; README, báo cáo chưa)
- **Owner:** Bảo (demo Gradio); Thư (README, notebook, báo cáo PDF tiếng Việt + tiếng Anh, slide)
- **Hạn:** T4 25/11 — nộp trước 14:00

## Goals / Purpose

- Xong: `python -m src.demo.app` mở giao diện nhập review → nhãn + xác suất + tô màu từ quan trọng; `notebooks/demo.ipynb` chạy được; README đủ lệnh; báo cáo đủ mục ở [REPORT_CHECKLIST.md](../REPORT_CHECKLIST.md).
- Không làm: triển khai lên server công khai.

## Input / Output

- **Input:** checkpoint tốt nhất + `vocab.json`, `data/sample/sample_100.csv`.
- **Output:** `src/predict.py`, `src/demo/app.py`, `notebooks/demo.ipynb`, ảnh chụp/video demo, báo cáo, slide.

## How to do it (general plan)

1. `src/predict.py`: hàm `predict(text) -> (label, prob)` dùng chung cho CLI, demo và notebook.
2. Tô màu từ: gọi `src/eval/explain.py` (occlusion hoặc gradient × embedding).
3. App Gradio có sẵn ví dụ từ `sample_100.csv`.
4. Đo thời gian inference trên CPU.
5. Ghép báo cáo theo REPORT_CHECKLIST; kiểm tra số liệu trước khi nộp; tập dượt phản biện.

## Detailed plan / gotchas

- Demo phải chạy được trên CPU, < 1 giây/review (gồm cả bước tô màu).
- Văn bản demo đi qua cùng hàm làm sạch và vocab như lúc train.

## Verification gates

- [ ] Clone repo mới, làm theo README, demo chạy được.
- [x] Thời gian 1 review trên CPU < 1 giây (ghi máy đo): GRU seed 123, 130 review, trung vị 60 ms, chậm nhất 199 ms, CPU Intel 12 luồng, Windows 11 (`experiments/results/demo/latency.json`, `python -m src.demo.benchmark`).
- [ ] REPORT_CHECKLIST: 7/7 Đạt.

## Kết quả demo (26/11)

- Mô hình: GRU seed 123 (chọn theo val), `max_len` = 400. `src/predict.py` tự chọn mô hình chính tốt nhất theo mean val acc 3 seed, bỏ qua pilot/ablation.
- Ảnh: `experiments/plots/demo/demo_screenshot.png` (câu khen → Positive 96%), `demo_screenshot_2.png` (câu vừa khen vừa chê, nhãn Negative → mô hình đoán **sai** Positive 97%; "exhausting" bị tô xanh, "but" gần như không ảnh hưởng — ví dụ cho giới hạn "đổi giọng ở cuối").
- Còn: các nút ví dụ lấy từ `sample_100.csv` hiện thẻ `<br />` thô (chỉ phần hiển thị, mô hình vẫn đọc văn bản đã làm sạch).

## Links

- Progress tracking: [../progress/DEMO_REPORT_STATUS.md](../progress/DEMO_REPORT_STATUS.md)
- Related: [OVERVIEW.md](../OVERVIEW.md)
