# PROMPT_HANDWRITTEN_30.md — Prompt soạn nháp 30 câu tự viết

- **Motivation/Background**: Bộ 30 câu tự viết (nhóm tình huống 7) do AI soạn nháp; cần lưu prompt để minh bạch việc dùng AI và để chạy lại được.
- **Purpose**: Lưu nguyên văn prompt đã dùng và các chỉnh sửa Thư làm sau đó.
- **Overview Pipeline**: Prompt → AI soạn 30 câu → kiểm định dạng, nhãn, trùng IMDb (`src/data/check_handwritten.py`) → Thư duyệt nhãn, thay câu → `data/sample/handwritten_30.csv`.
- **Detailed Plan**: §1 Prompt; §2 Chỉnh sửa sau khi kiểm.
- **References**: [../DATA.md](../DATA.md) §7.0, `experiments/results/data_prep/handwritten_30_check.json`.
- **Created**: 2026-11-24T15:37:32+07:00
- **Last Updated**: 2026-11-24T15:37:32+07:00

---

## 1. Prompt (nguyên văn)

```
Bạn là người viết dữ liệu kiểm thử cho một mô hình phân loại cảm xúc review phim tiếng Anh (nhị phân: positive = 1, negative = 0). Mô hình được huấn luyện trên bộ IMDb Large Movie Review Dataset (Maas et al., 2011).

NHIỆM VỤ
Viết đúng 30 câu review MỚI, chia 6 nhóm, mỗi nhóm 5 câu:

1. positive — khen rõ ràng. label = 1
2. negative — chê rõ ràng. label = 0
3. negation — có phủ định làm đổi nghĩa ("not bad", "didn't enjoy", "never boring", "no reason to watch"). 3 câu label 1, 2 câu label 0
4. sarcasm — mỉa mai: bề mặt dùng từ khen nhưng ý thật là chê (hoặc ngược lại). Ít nhất 4 câu label 0
5. mixed — vừa khen vừa chê, kết luận cuối cùng quyết định nhãn ("great cast, but..." / "slow start, yet..."). 3 câu label 0, 2 câu label 1
6. out_of_domain — KHÔNG phải review phim (nhà hàng, điện thoại, khách sạn, giao hàng, sách). 3 câu label 1, 2 câu label 0

RÀNG BUỘC BẮT BUỘC
- Tự sáng tác hoàn toàn. KHÔNG chép, KHÔNG diễn đạt lại câu nào từ IMDb, Rotten Tomatoes, SST/SST-2, Amazon/Yelp reviews hay bất kỳ bộ dữ liệu công khai nào bạn có thể đã thấy khi huấn luyện.
- KHÔNG dùng tên phim, diễn viên, đạo diễn, nhân vật có thật (tránh câu bị nhớ thuộc từ dữ liệu). Nói chung chung: "the film", "the lead actress", "the director".
- Tránh các cụm sáo rỗng rất phổ biến trên IMDb như "worst movie ever", "waste of time", "10/10", "must see", "Oscar-worthy".
- Tiếng Anh tự nhiên như người dùng thật viết, 1–3 câu mỗi review, 8–40 từ.
- Không dùng điểm số (x/10, sao) và không dùng emoji, để mô hình phải dựa vào ngôn ngữ.
- Mỗi câu khác nhau về cấu trúc, không lặp khuôn.
- Nhãn phải là cảm xúc thật mà một người đọc bình thường hiểu được, không mơ hồ. Nếu một câu có thể hiểu cả hai cách thì viết lại.

ĐẦU RA
Chỉ trả về một khối CSV, không giải thích thêm, đúng 3 cột theo thứ tự, có dòng tiêu đề, text đặt trong dấu nháy kép:
text,label,case_type
case_type nhận đúng một trong: positive, negative, negation, sarcasm, mixed, out_of_domain

TỰ KIỂM TRA TRƯỚC KHI TRẢ LỜI
- Đủ 30 dòng dữ liệu, mỗi nhóm đúng 5 dòng
- Tổng nhãn: 15 dòng label 1, 15 dòng label 0
- Không có tên riêng có thật, không có điểm số, không có emoji
- Không câu nào là bản chép hoặc gần giống một review bạn nhớ từ dữ liệu công khai
```

## 2. Chỉnh sửa sau khi kiểm (Thư duyệt, 2026-11-24)

Bản AI trả về lệch nhãn 13/17 (cả 5 câu `sarcasm` đều nhãn 0, `mixed` 2/3). Thay 2 câu để thành 15/15:

| Nhóm | Câu bị thay | Câu mới | Nhãn |
| :--- | :--- | :--- | :--- |
| sarcasm | "The dialogue is so sophisticated that every conversation sounds like two people competing to produce the most ridiculous sentence." (0) | "Terrible news for my sleep schedule: I stayed up past 3 a.m. because I simply could not stop watching." | 1 |
| mixed | "Strong performances and thoughtful themes initially caught my attention, yet the slow pacing eventually made the film difficult to enjoy." (0) | "The middle section drags and a few jokes fall flat, but the last twenty minutes are so moving that I left the theater smiling." | 1 |
