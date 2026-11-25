# MODEL_STATUS.md — Tiến độ: Định nghĩa mô hình

- **Motivation/Background**: Theo dõi nhanh phase Định nghĩa mô hình mà không cần đọc lại toàn bộ đặc tả.
- **Purpose**: Ghi việc đã xong kèm bằng chứng, blocker, quyết định và bước tiếp theo.
- **Overview Pipeline**: Log → blocker → quyết định → bước tiếp theo.
- **Detailed Plan**: Header; Checklist; Log; Blockers; Decisions; Next step; Links.
- **References**: [../phases/MODEL.md](../phases/MODEL.md), [../OVERVIEW.md](../OVERVIEW.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T10:00:45+07:00

---

## Header

- **Title:** Định nghĩa mô hình
- **Date created:** 2026-11-05
- **Last updated:** 2026-11-23
- **Status:** In Progress (D1–D2 Done; D3 chờ tạo `.npy` trên dữ liệu thật)
- **Owner:** Đại
- **Hạn:** T2 23/11 tối (code + test); pilot sáng T3 24/11
- **Phase doc:** [../phases/MODEL.md](../phases/MODEL.md)

## Checklist

Chỉ đánh dấu `[x]` khi có bằng chứng (lệnh, đường dẫn output, commit).

- [x] Đọc xong đặc tả phase
- [x] Code chạy được trên tập con (`pytest tests/test_d1_data.py tests/test_models.py tests/test_glove.py`)
- [x] Verification gates trong phase doc đều pass (shape, overfit 1 batch loss < 0.05, save/load, M4a frozen params)
- [ ] Cập nhật tài liệu liên quan (MODELS.md, DATA.md §6 — điền sau khi trích `.npy` trên dữ liệu thật)

## Log

- 2026-11-05: Tạo file.
- 2026-11-25: Hoàn thiện `src/data/{tokenizer,vocab,dataset}.py` (hỗ trợ `post`/`pre`/`head_tail`), `src/models/{rnn,glove}.py`, và bộ test `tests/{test_d1_data,test_models,test_glove}.py`. Tất cả verification gates của D1–D2 pass; D3 sẵn sàng trích `glove_100d_vocab.npy` trên Colab.

## Blockers

- Chưa có.

## Decisions

- Chưa có.

## Next step

- Bắt đầu bước 1 trong [../phases/MODEL.md](../phases/MODEL.md).

## Links

- Phase doc: [../phases/MODEL.md](../phases/MODEL.md)
- Experiments: [../experiments/README.md](../experiments/README.md)
