# DATA.md — Đặc tả dữ liệu IMDb

- **Motivation/Background**: Số đo chỉ đáng tin khi split cố định và không rò rỉ. IMDb có nhiều review cùng một phim nên chia ngẫu nhiên dễ rò rỉ.
- **Purpose**: Đặc tả chính thức về nguồn dữ liệu, cấu trúc file, giao thức chia train/val/test, kiểm tra trùng lặp và rò rỉ, GloVe, dữ liệu mẫu.
- **Overview Pipeline**: Tải `aclImdb_v1.tar.gz` → đọc `train/`, `test/` → gắn `rating` (tên file) và `movie_url` (`urls_*.txt`) → làm sạch → hash, loại trùng → chia val từ train theo phim → ghi CSV vào `data/splits/`.
- **Detailed Plan**: §1 Nguồn; §2 Cấu trúc file; §3 Giao thức split; §4 Trùng lặp & rò rỉ; §5 EDA; §6 Vocab & GloVe; §7 Dữ liệu mẫu; §8 Dữ liệu ngoài miền; §9 Hạn chế.
- **References**: [PURPOSE.md](PURPOSE.md) §5, §9; [phases/DATA_PREP.md](phases/DATA_PREP.md).
- **Created**: 2026-11-05T11:21:04+07:00
- **Last Updated**: 2026-11-23T10:01:39+07:00

---

## 1. Nguồn

- **Large Movie Review Dataset v1.0** — Maas, A. L. et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL 2011.
- Link: <https://ai.stanford.edu/~amaas/data/sentiment/> (`aclImdb_v1.tar.gz`, ~80 MB).
- Phương án thay thế: `stanfordnlp/imdb` (Hugging Face Datasets). Bản này không có `movie_url`; nếu phải dùng, ghi rõ trong [progress/DATA_PREP_STATUS.md](progress/DATA_PREP_STATUS.md) là không chia được theo phim.
- Không phát tán lại toàn bộ dữ liệu; repo chỉ commit `data/sample/` (§7).

## 2. Cấu trúc file

```
aclImdb/
├── train/{pos,neg}/<id>_<rating>.txt   # 12.500 + 12.500
├── train/unsup/                        # 50.000 không nhãn — không dùng
├── train/urls_{pos,neg}.txt            # dòng i = URL phim của review id i
├── test/{pos,neg}/<id>_<rating>.txt    # 12.500 + 12.500
├── test/urls_{pos,neg}.txt
└── imdb.vocab
```

- Rating 1–4 → `label = 0` (negative); rating 7–10 → `label = 1` (positive). Rating 5–6 không có trong tập có nhãn.
- `rating` chỉ dùng để phân tích lỗi (nhóm 5 trong [PURPOSE.md](PURPOSE.md) §9), không đưa vào mô hình.
- Theo tác giả: tối đa 30 review mỗi phim; phim ở train và test không trùng nhau.

## 3. Giao thức split

**Dùng lại split của Lab 3** (Thư chốt 2026-11-23): `data/splits/train_val_split.json`, chép từ Lab-DL `labs/lab3/data/splits/train_val_split.json` (nhánh `feat/lab3-redo`). File chỉ chứa danh sách `uid`, được commit.

| Split | Nguồn | Kích thước (trước khi loại trùng) | Ghi chú |
| :--- | :--- | :--- | :--- |
| train | `train/`, các uid trong `train_uids` | 22.499 | Dùng để train và fit vocab/TF-IDF |
| val | `train/`, các uid trong `val_uids` | 2.501 (pos 50,02%) | Chọn epoch, siêu tham số, checkpoint |
| test | toàn bộ `test/` | 25.000 | Giữ nguyên; chỉ chạy một lần cho số cuối |

- Cách Lab 3 tạo file: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`, lấy fold 0 làm val. Nhóm = mã phim, **gộp thêm review trùng nội dung** (union-find) để cả cặp trùng nằm cùng một phía. Kết quả: `groups_in_both = 0`. Code: `src/data/split.py`.
- Lý do dùng lại: train/val giống hệt Lab 3 nên số DistilBERT của Lab 3 so trực tiếp được với LSTM; split đã kiểm; không tốn thời gian làm lại.
- Split cố định cho mọi thí nghiệm; 3 seed huấn luyện (42, 123, 2025) **không** đổi split.
- Loại review trùng (§4) làm **sau** khi áp split → số train/val cuối nhỏ hơn bảng trên một chút; ghi số cuối vào bảng §4.
- Vocab và TF-IDF chỉ fit trên train, không fit trên val/test.
- Bàn giao cho mô hình: `data/processed/reviews.parquet` (không commit), cột `uid, split, text, text_clean, label, rating, movie_id` (`split` ∈ train/val/test).

**Số cuối cùng** (`python -m src.data.prepare`, 2026-11-23):

| Split | Review | pos | neg | % pos | Phim | Dùng cho |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| train | 22.304 | 11.208 | 11.096 | 50,25% | 3.081 | Train + fit vocab/TF-IDF |
| val | 2.473 | 1.235 | 1.238 | 49,94% | 343 | Chọn epoch, siêu tham số, `C` baseline |
| test | 25.000 | 12.500 | 12.500 | 50,00% | 3.581 | Chỉ chạy một lần ở phase EVAL |
| **tổng** | **49.777** | | | | | |

- File: `data/processed/reviews.parquet` (81,2 MB, không commit).
- **So sánh nội dung giữa các máy bằng `content_hash`**, không dùng sha256 file: `content_hash = int(pd.util.hash_pandas_object(df, index=False).sum())` = **`1029824945594009414`** (giống nhau trên máy Bảo và máy Thư). Giá trị ghi trong `experiments/results/data_prep/prepare_stats.json`.
- sha256 file parquet (`e6245753…` trên máy Bảo, `becd19a8…` trên máy Thư) **chỉ trùng khi cùng phiên bản pandas/pyarrow** vì parquet ghi metadata thư viện — khác nhau không có nghĩa là file sai. Chạy lại `prepare` trên cùng máy vẫn cho cùng sha256.
- Danh sách uid test trùng train: `experiments/results/data_prep/test_uids_dup_train.txt` (**123 dòng**, sắp theo uid, commit) — phase EVAL dùng để báo thêm "test acc khi bỏ 123 review trùng train" (§4).
- Thống kê từng bước (đầu vào, số bỏ theo từng quy tắc, kiểm rò rỉ): `data/processed/prepare_stats.json` và bản commit `experiments/results/data_prep/prepare_stats.json`.

- Môi trường đã kiểm (2026-11-24, Python 3.12): `pandas==3.1.1`, `pyarrow==26.0.1`, `scikit-learn==1.9.1` — ghim trong `requirements.txt`. Đường dẫn trong `prepare_stats.json` ghi dạng POSIX (`/`) để giống nhau giữa Linux và Windows.

### Hợp đồng dữ liệu (bàn giao cho Đại)

`data/processed/reviews.parquet`, 49.777 dòng, đúng 7 cột theo thứ tự:

| Cột | Kiểu | Nội dung |
| :--- | :--- | :--- |
| `uid` | string | `{split}_{pos|neg}_{id}`, duy nhất toàn file, ổn định qua mọi lần chạy |
| `split` | string | `train` / `val` / `test` — lọc ra từng tập, không cần file phụ |
| `text` | string | văn bản gốc (còn `<br />`), dùng khi cần hiển thị nguyên văn |
| `text_clean` | string | đã bỏ `<br />`, gộp khoảng trắng; giữ hoa/thường — **đầu vào cho TF-IDF và tokenizer** |
| `label` | int64 | 0 = negative (rating ≤ 4), 1 = positive (rating ≥ 7) |
| `rating` | int64 | 1–4 hoặc 7–10 (không có 5–6) |
| `movie_id` | string | mã phim `tt...` — dùng để kiểm rò rỉ, không đưa vào mô hình |

```python
import pandas as pd
df = pd.read_parquet("data/processed/reviews.parquet")
train = df[df["split"] == "train"].reset_index(drop=True)
val   = df[df["split"] == "val"].reset_index(drop=True)
test  = df[df["split"] == "test"].reset_index(drop=True)   # chỉ dùng ở EVAL
```

Bất biến đã kiểm bằng code (`src.data.prepare.check_contract`, **raise nếu vi phạm**): 0 hash chung giữa train/val và test; 0 `movie_id` chung giữa 3 split; `label` khớp `rating` 100%; không NaN ở `text`/`text_clean`. Test giữ đúng 25.000 review, không sửa một dòng nào.

## 4. Trùng lặp & rò rỉ

Hash = SHA-1 của văn bản đã làm sạch (bỏ `<br />`, gộp khoảng trắng; **giữ hoa/thường**), hàm `src.data.clean.text_hash` — giống Lab 3 để số đếm khớp Lab 3.

**Quy tắc loại bỏ** (Thư chốt 2026-11-23, theo đề xuất của Đại):
- Trùng trong train/val: giữ 1 bản.
- Trùng giữa train/val và test: bỏ bản ở train/val, để test giữ nguyên 25.000.
- Trùng trong test: không xoá (test giữ nguyên); ghi số lượng vào bảng.
- Báo thêm số test khi bỏ các review test trùng với train (so sánh được với Lab 3).
- Review của phim có mặt trong test: bỏ khỏi train/val, giữ test (**Thư chốt 2026-11-23**; dữ liệu thật có 1 phim như vậy — `tt0406816`).

**Train cuối (22.304) không còn giống train Lab 3 (22.499)**: đã loại 98 bản trùng + 123 review trùng với test + 2 review của phim lọt. Khi so với DistilBERT của Lab 3, ghi là **"cùng split (cùng val), train đã loại trùng"** — không so số tuyệt đối của train giữa hai bên.

Số đo thật (2026-11-23, `python -m src.data.prepare`; nguồn: `experiments/results/data_prep/prepare_stats.json`):

| Kiểm tra | Kết quả |
| :--- | :--- |
| Review trùng trong train chính thức (đã loại) | **98** — 87 ở phần train + 11 ở phần val |
| Review trùng giữa train và test (đã loại khỏi train/val) | **123** — 106 ở train + 17 ở val (khớp Lab 3: 123 review test trùng train) |
| Review trùng trong test (chỉ ghi nhận, giữ nguyên) | **201** bản thừa, **394** review nằm trong nhóm trùng |
| `movie_id` chung giữa train và val | **0** |
| `movie_id` chung giữa train và test | **1 phim** (`tt0406816`, "The Guardian"): 2 review train, 23 review test — đã bỏ 2 review train, giữ test |
| `movie_id` chung giữa val và test | **0** |
| Hash chung giữa train/val và test sau khi xử lý | **0** |
| uid test trùng train (đã lưu) | **123**, trong `experiments/results/data_prep/test_uids_dup_train.txt` |
| Nhóm trùng khác nhãn (cùng nội dung, khác nhãn) | **0** ở cả 3 split |
| Tỷ lệ nhãn mỗi split | train 50,25% pos · val 49,94% · test 50,00% |
| `label` không khớp `rating`, `text` rỗng | **0** |

Ghi chú về phim `tt0406816`: aclImdb tách train/test theo phim, nhưng phim này lọt cả hai phía (2 review train id 12498–12499, 23 review test) — ngoại lệ của dữ liệu, không phải lỗi split. Vì split nhóm theo phim và mô hình không được thấy phim của test, `prepare` bỏ 2 review train này (quy tắc 4 trong `src/data/prepare.py`); test vẫn đủ 25.000.

Ghi lệnh sinh từng số vào [progress/DATA_PREP_STATUS.md](progress/DATA_PREP_STATUS.md).

## 5. EDA

Chạy `python -m src.data.eda` (hàm lấy từ Lab 3): ghi `experiments/results/eda/*.csv` và 3 hình vào `experiments/plots/eda/` cho **cả train, val, test** (hướng dẫn của cô: thống kê lại trên từng tập đã chia; dùng cột, histogram, không dùng biểu đồ tròn). Mặc định đếm theo **từ**; khi có tokenizer thì truyền `tokenize=` để đếm token. Nội dung:
- Phân bố nhãn, phân bố `rating`.
- Độ dài review (số từ): trung bình, trung vị, phân vị 90/95 — dùng để chọn `max_len` mặc định.
- Từ phổ biến của từng lớp.

## 6. Vocab & GloVe

- **Vocab:** xây từ train; `<pad>=0`, `<unk>=1`. Lưu `vocab.json` cạnh checkpoint (xem [MODELS.md](MODELS.md) §3). Ngưỡng `min_freq` / `max_size`: TBD, ghi vào `configs/`.
- **GloVe:** `glove.6B.zip` (Wikipedia 2014 + Gigaword 5, 400k từ) — <https://nlp.stanford.edu/projects/glove/>, giấy phép ODC-PDDL. Chỉ dùng `glove.6B.100d.txt`, để ở `data/glove/`.
- Lọc GloVe theo vocab, lưu ma trận `data/processed/glove_100d_vocab.npy` (không commit). Từ không có trong GloVe khởi tạo ngẫu nhiên.
- Tỷ lệ phủ vocab: TBD (đo bằng code).

## 7. Dữ liệu mẫu

Được commit, trong `data/sample/`:

| File | Nội dung | Dùng cho |
| :--- | :--- | :--- |
| `sample_100.csv` | 100 review lấy từ test, có cả ca khó (phủ định, mỉa mai, dài, khen chê lẫn lộn, điểm sát ngưỡng) | Demo, minh họa trong báo cáo |
| `handwritten_30.csv` | ~30 câu nhóm tự viết, gồm câu ngắn và câu ngoài miền phim | Nhóm tình huống 7, thử độ bền demo |

Cột: `text`, `label`, `case_type` (+ `rating`, `source_file` với `sample_100.csv`).

## 8. Dữ liệu ngoài miền (mở rộng)

Rotten Tomatoes (Hugging Face `rotten_tomatoes`), chỉ đánh giá chéo, không train lại. Lưu ở `data/raw/rotten_tomatoes/` (không commit).

## 9. Hạn chế

- Chỉ có review cực tính (≤4 hoặc ≥7 sao): mô hình chưa gặp review trung tính.
- Văn bản có thẻ HTML (`<br />`), cần làm sạch.
- Review dài bị cắt ở `max_len`, có thể mất câu kết luận.
- `sample_100.csv` lấy từ test: chỉ để demo, không dùng chọn mô hình.
