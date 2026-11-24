"""Chọn 100 review đại diện từ tập test (B4) → `data/sample/sample_100.csv`.

Mục đích: mẫu nhỏ, cố định, có cả ca khó để demo và minh hoạ trong báo cáo. Mẫu **chỉ** lấy từ
test, không dùng để train hay chọn siêu tham số (không gây rò rỉ vì test không tham gia chọn mô hình).

Cách chọn (tất định, `seed=42`, mỗi review chỉ vào một nhóm):

| Nhóm | Điều kiện | Mỗi lớp |
| :--- | :--- | ---: |
| `rating_4_negative` / `rating_7_positive` | đúng ngưỡng (4 = âm sát biên, 7 = dương sát biên) | 12 |
| `negation` | "not/n't" + từ cảm xúc trong 3 từ ("not good", "didn't like") | 8 |
| `long_review` | ≥ 500 từ (≈ P90 của test) | 10 |
| `short_review` | ≤ 20 từ | 5 |
| `random` | phần còn lại, lấy ngẫu nhiên có seed | 15 |
| **tổng mỗi lớp** | | **50** |

Thứ tự nhóm trên là thứ tự ưu tiên: review đã được chọn cho nhóm trước không lặp lại ở nhóm sau.
Nếu một nhóm không đủ số, phần thiếu bù bằng `random` để luôn đúng 100 dòng, 50 âm / 50 dương.

Chạy từ gốc repo:
    python -m src.data.sample        # cần data/processed/reviews.parquet (B1)
"""
from __future__ import annotations

import random
import re

import pandas as pd

from src.data.clean import text_hash
from src.data.eda import NEGATION_RE
from src.utils.paths import data_dir

SAMPLE_SIZE = 100
SEED = 42
SHORT_MAX_WORDS = 20
LONG_MIN_WORDS = 500
SOURCE_FILE = "test"
COLUMNS = ["uid", "text", "label", "case_type", "rating", "source_file"]

# "not" / "n't" + (tối đa 2 từ) + một từ cảm xúc — ca phủ định khó, có tín hiệu đảo nghĩa rõ.
NEGATED_SENTIMENT_RE = re.compile(
    r"(?:not|n't)\s+(?:\w+\s+){0,2}(?:good|bad|great|like|love|hate|boring|funny|interesting|worst|best)\b",
    re.IGNORECASE,
)

#: Hạn mức từng nhóm cho MỖI lớp (âm/dương); tổng = SAMPLE_SIZE / 2.
QUOTAS = {
    "rating_4_negative": 12,
    "rating_7_positive": 12,
    "negation": 8,
    "long_review": 10,
    "short_review": 5,
}


def _lengths(df: pd.DataFrame) -> pd.Series:
    return df["text_clean"].str.split().str.len() if "text_clean" in df.columns else df["text"].str.split().str.len()


def select_sample(
    test: pd.DataFrame,
    n: int = SAMPLE_SIZE,
    seed: int = SEED,
    short_max: int = SHORT_MAX_WORDS,
    long_min: int = LONG_MIN_WORDS,
) -> tuple[pd.DataFrame, dict]:
    """Chọn đúng `n` review từ `test` (không sửa `test`), cân bằng nhãn, có ca khó.

    Trả về (bảng mẫu, dict thống kê). Tất định với cùng `test` + `seed`.
    """
    required = {"uid", "text", "label", "rating", "text_clean"}
    missing = required - set(test.columns)
    if missing:
        raise ValueError(f"test: missing columns {sorted(missing)}")
    if n <= 0 or n % 2:
        raise ValueError(f"n must be a positive even number, got {n}")
    if not set(test["label"]) <= {0, 1}:
        raise ValueError("label must be 0/1")
    if test["uid"].duplicated().any():
        raise ValueError("uid must be unique in test")

    per_label = n // 2
    df = test.copy()
    lengths = _lengths(df)
    hash_of = dict(zip(df["uid"], df["text"].map(text_hash)))
    rng = random.Random(seed)
    used: set[str] = set()
    used_hashes: set[str] = set()
    picked: list[pd.DataFrame] = []

    def take(mask: pd.Series, quota: int, case: str) -> None:
        # bỏ review đã chọn và review trùng nội dung với review đã chọn (test còn bản trùng)
        pool = [u for u in sorted(df.loc[mask & ~df["uid"].isin(used), "uid"]) if hash_of[u] not in used_hashes]
        chosen = rng.sample(pool, min(quota, len(pool)))
        used.update(chosen)
        used_hashes.update(hash_of[u] for u in chosen)
        if chosen:
            picked.append(df[df["uid"].isin(chosen)].assign(case_type=case))

    negated = df["text_clean"].str.contains(NEGATED_SENTIMENT_RE, regex=True)
    fallback_negation = df["text_clean"].str.contains(NEGATION_RE, regex=True)
    is_neg, is_pos = df["label"] == 0, df["label"] == 1

    for label, mask_label, rating, border_case in (
        (0, is_neg, 4, "rating_4_negative"),
        (1, is_pos, 7, "rating_7_positive"),
    ):
        take(mask_label & (df["rating"] == rating), QUOTAS[border_case], border_case)
        # ca phủ định: ưu tiên "not + từ cảm xúc"; nếu thiếu thì lấy mọi review có dấu hiệu phủ định
        quota = QUOTAS["negation"]
        before = len(used)
        take(mask_label & negated, quota, "negation")
        taken = len(used) - before
        if taken < quota:
            take(mask_label & fallback_negation, quota - taken, "negation")
        take(mask_label & (lengths >= long_min), QUOTAS["long_review"], "long_review")
        take(mask_label & (lengths <= short_max), QUOTAS["short_review"], "short_review")

    # Bù cho đủ 50 mỗi lớp bằng review ngẫu nhiên còn lại; nếu test quá ít văn bản khác nhau
    # thì cho phép trùng nội dung (vẫn khác uid) để mẫu đúng số dòng.
    for label, mask_label in ((0, is_neg), (1, is_pos)):
        have = int((pd.concat(picked)["label"] == label).sum()) if picked else 0
        need = per_label - have
        take(mask_label, need, "random")
        got = int((pd.concat(picked)["label"] == label).sum()) if picked else 0
        if got < per_label:
            pool = sorted(df.loc[mask_label & ~df["uid"].isin(used), "uid"])
            chosen = rng.sample(pool, min(per_label - got, len(pool)))
            used.update(chosen)
            if chosen:
                picked.append(df[df["uid"].isin(chosen)].assign(case_type="random"))

    out = pd.concat(picked, ignore_index=True)
    out["source_file"] = SOURCE_FILE
    out = out[COLUMNS].sort_values(["case_type", "uid"], kind="stable").reset_index(drop=True)
    if len(out) != n:
        raise ValueError(f"selected {len(out)} rows, expected {n}")
    if out["text"].map(text_hash).duplicated().any():
        raise ValueError("sample contains duplicate review texts")
    stats = {
        "n": len(out),
        "seed": seed,
        "short_max_words": short_max,
        "long_min_words": long_min,
        "by_label": out["label"].value_counts().sort_index().to_dict(),
        "by_case": out["case_type"].value_counts().to_dict(),
        "rating": out["rating"].value_counts().sort_index().to_dict(),
    }
    return out, stats


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parquet = data_dir("processed") / "reviews.parquet"
    if not parquet.exists():
        raise SystemExit(f"Chưa có {parquet} — chạy `python -m src.data.prepare` trước (B1).")
    df = pd.read_parquet(parquet)
    test = df[df["split"] == "test"].reset_index(drop=True)
    out, stats = select_sample(test)

    path = data_dir("sample") / "sample_100.csv"
    out.to_csv(path, index=False)
    print(f"Đã ghi {len(out)} review vào {path}")
    print(f"nhãn: {stats['by_label']}")
    print(f"nhóm: {stats['by_case']}")
    print(f"rating: {stats['rating']}")


if __name__ == "__main__":
    main()
