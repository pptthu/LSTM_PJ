"""Kiểm bộ 30 câu tự viết (`data/sample/handwritten_30.csv`): định dạng, cân bằng nhãn, trùng với IMDb.

Trùng với IMDb đo bằng cụm N từ liên tiếp (mặc định 6) trên chữ thường, so với cả 50.000 review
train + test gốc. Trùng cụm thông dụng ("not a bad film at all") là bình thường; cần chú ý khi
một câu trùng gần nguyên văn (nhiều cụm liên tiếp).

Chạy từ gốc repo:
    python -m src.data.check_handwritten    # ghi experiments/results/data_prep/handwritten_30_check.json
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src.data.download import load_split
from src.utils.paths import data_dir, experiment_dir

CASE_TYPES = ("positive", "negative", "negation", "sarcasm", "mixed", "out_of_domain")
N_GRAM = 6
_WORD_RE = re.compile(r"[a-z']+")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower().replace("<br />", " "))


def ngrams(text: str, n: int = N_GRAM) -> set[str]:
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def check_format(df: pd.DataFrame) -> dict:
    """Đúng cột, nhãn 0/1, case_type hợp lệ; trả về số câu theo nhóm × nhãn."""
    if list(df.columns) != ["text", "label", "case_type"]:
        raise ValueError(f"columns must be text,label,case_type, got {list(df.columns)}")
    if not set(df["label"]) <= {0, 1}:
        raise ValueError("label must be 0/1")
    bad = set(df["case_type"]) - set(CASE_TYPES)
    if bad:
        raise ValueError(f"unknown case_type {sorted(bad)}")
    scored = df[df["case_type"] != "out_of_domain"]
    return {
        "n_rows": len(df),
        "by_case_label": {f"{c}/{l}": int(n) for (c, l), n in df.groupby(["case_type", "label"]).size().items()},
        "label_totals": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "scored_label_totals": {str(k): int(v) for k, v in scored["label"].value_counts().sort_index().items()},
        "words_min": int(df["text"].map(lambda t: len(t.split())).min()),
        "words_max": int(df["text"].map(lambda t: len(t.split())).max()),
    }


def imdb_overlap(df: pd.DataFrame, corpus: list[str], n: int = N_GRAM) -> list[dict]:
    """Với mỗi câu: các cụm n từ cũng xuất hiện trong corpus, và có trùng nguyên văn hay không."""
    wanted: dict[str, set[int]] = {}
    for i, t in enumerate(df["text"]):
        for g in ngrams(t, n):
            wanted.setdefault(g, set()).add(i)
    found: dict[int, set[str]] = {}
    exact = set(" ".join(words(t)) for t in df["text"])
    exact_hits = set()
    for doc in corpus:
        w = words(doc)
        joined = " ".join(w)
        for e in exact:
            if e in joined:
                exact_hits.add(e)
        for j in range(len(w) - n + 1):
            g = " ".join(w[j:j + n])
            for i in wanted.get(g, ()):
                found.setdefault(i, set()).add(g)
    return [{
        "row": i,
        "case_type": df["case_type"].iloc[i],
        "shared_ngrams": sorted(found.get(i, ())),
        "exact_in_imdb": " ".join(words(df["text"].iloc[i])) in exact_hits,
    } for i in range(len(df)) if i in found or " ".join(words(df["text"].iloc[i])) in exact_hits]


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = pd.read_csv(data_dir("sample") / "handwritten_30.csv")
    report = check_format(df)
    corpus = pd.concat([load_split("train"), load_split("test")])["text"].tolist()
    overlap = imdb_overlap(df, corpus)
    report.update({
        "ngram": N_GRAM,
        "imdb_reviews_compared": len(corpus),
        "rows_with_shared_ngram": len(overlap),
        "rows_exact_in_imdb": sum(o["exact_in_imdb"] for o in overlap),
        "overlap": overlap,
    })
    out = experiment_dir("results", "data_prep") / "handwritten_30_check.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "overlap"}, ensure_ascii=False, indent=1))
    print(f"→ {out.as_posix()}")


if __name__ == "__main__":
    main()
