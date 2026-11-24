# notebooks/

- **`presentation.ipynb` — notebook nộp kèm báo cáo (Thư, T4)**: EDA → Baseline → Mô hình → Theo dõi huấn luyện → Đánh giá → Phân tích lỗi → Giải thích → Demo → Giới hạn. Mở từ thư mục `notebooks/`, Run All; section nào chưa có file sẽ hiện ⏳ và tên người đang làm.
- `01_eda.ipynb` — EDA: phân bố nhãn, độ dài review, từ phổ biến (Bảo).
- `02_results.ipynb` — Đọc `experiments/results/`, vẽ bảng và biểu đồ cho báo cáo (Đại).
- `03_error_analysis.ipynb` — Xem các review sai theo 7 nhóm tình huống, kết quả giải thích mô hình (Thư).
- `demo.ipynb` — Gọi `src/predict.py` trên `data/sample/`, hiện nhãn, xác suất, từ quan trọng (Bảo).

Notebook chỉ đọc kết quả và vẽ; không train trong notebook.
