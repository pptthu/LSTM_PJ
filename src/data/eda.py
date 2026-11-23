"""Hàm EDA dùng lại từ Lab 3: nhãn, rating, độ dài, review trùng, phủ định + 3 hình (cột, histogram, box).

Lấy từ Lab-DL `labs/lab3/eda/eda.py` (nhánh feat/lab3-redo). Đổi so với Lab 3:
- Không gắn với tokenizer DistilBERT: `tokenize` là hàm str -> list[str] truyền vào. Mặc định đếm TỪ
  (tách theo khoảng trắng trên văn bản đã làm sạch); khi có tokenizer của dự án thì truyền vào để đếm token.
- Nhận nhiều split có tên (train / val / test) để làm lại thống kê trên từng tập đã chia (hướng dẫn của cô,
  M/deepfake_coursework_submission/docs/DESCRIPTION_NOTES.md).
- Vạch cắt mặc định 200 / 400 / 600 (thí nghiệm max_len, plan §4.2); nhãn hình tiếng Anh (plan §10).
Theo hướng dẫn của cô: dùng cột, histogram; không dùng biểu đồ tròn.

Chạy từ gốc repo:
    python -m src.data.eda           # train / val (split Lab 3) / test, đếm theo từ
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.clean import clean_text, text_hash  # noqa: E402

LENGTH_LIMITS = (200, 400, 600)
# Dấu hiệu phủ định: "not", "no" chiếm > 55% dấu hiệu phủ định trong SFU Review corpus
# (Konstantinova et al. 2012); thêm các từ phủ định thường gặp khác (giữ như Lab 3).
NEGATION_RE = re.compile(r"\b(not|no|never|nothing|none|nor|neither|nobody|without)\b|n't", re.IGNORECASE)

Tokenize = Callable[[str], list]


def word_split(text: str) -> list[str]:
    """Tokenize mặc định: tách theo khoảng trắng (đếm số từ)."""
    return text.split()


# ---------------------------------------------------------------- số liệu
def token_lengths(texts, tokenize: Tokenize = word_split) -> np.ndarray:
    """Số đơn vị (từ hoặc token) của từng văn bản, không cắt."""
    return np.asarray([len(tokenize(t)) for t in texts], dtype=np.int64)


def length_summary(lengths, limits=LENGTH_LIMITS) -> dict:
    """mean, median, P90, P95, max và % văn bản dài hơn từng mức (= sẽ bị cắt ở max_len đó)."""
    lengths = np.asarray(lengths)
    out = {
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
    }
    for limit in limits:
        out[f"pct_over_{limit}"] = float((lengths > limit).mean() * 100)
    return out


def duplicate_counts(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Review trùng nội dung trong train, trong test và giữa hai tập.

    "extra" = số bản thừa (nhóm 3 bản giống nhau → 2 bản thừa).
    """
    h_train = train["text"].map(text_hash)
    h_test = test["text"].map(text_hash)
    return {
        "train_extra_copies": int(h_train.duplicated().sum()),
        "train_reviews_in_dup_groups": int(h_train.duplicated(keep=False).sum()),
        "test_extra_copies": int(h_test.duplicated().sum()),
        "test_reviews_in_dup_groups": int(h_test.duplicated(keep=False).sum()),
        "test_reviews_also_in_train": int(h_test.isin(set(h_train)).sum()),
        "train_dup_groups_label_conflict": int(
            train.assign(h=h_train).groupby("h")["label"].nunique().gt(1).sum()
        ),
    }


def negation_share(texts) -> float:
    """% văn bản có ít nhất một từ phủ định."""
    texts = list(texts)
    return float(np.mean([bool(NEGATION_RE.search(t)) for t in texts]) * 100)


def split_summary(name: str, df: pd.DataFrame, tokenize: Tokenize = word_split,
                  limits=LENGTH_LIMITS) -> tuple[dict, np.ndarray]:
    """Số liệu của một split + độ dài từng review (đã làm sạch)."""
    raw = df["text"].tolist()
    clean = [clean_text(t) for t in raw]
    len_raw = token_lengths(raw, tokenize)
    len_clean = token_lengths(clean, tokenize)
    row = {
        "split": name,
        "n_reviews": len(df),
        "n_pos": int((df["label"] == 1).sum()),
        "n_neg": int((df["label"] == 0).sum()),
        "n_movies": int(df["movie_id"].nunique()),
        "max_reviews_per_movie": int(df["movie_id"].value_counts().max()),
        "pct_with_br": float(np.mean(["<br" in t.lower() for t in raw]) * 100),
        "br_units_mean": float((len_raw - len_clean).mean()),
        "pct_with_negation": negation_share(clean),
    }
    row.update({f"len_{k}": v for k, v in length_summary(len_clean, limits).items()})
    return row, len_clean


# ---------------------------------------------------------------- hình (nhãn tiếng Anh)
def plot_label_distribution(summary: pd.DataFrame, path: Path) -> None:
    """Cột: số review neg/pos mỗi split."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["n_neg"], 0.4, label="negative (0)")
    ax.bar(x + 0.2, summary["n_pos"], 0.4, label="positive (1)")
    ax.set_xticks(x, summary["split"])
    ax.set_ylabel("reviews")
    ax.set_title("Label distribution")
    ax.set_ylim(0, max(summary["n_neg"].max(), summary["n_pos"].max()) * 1.25)
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_length_histogram(lengths: pd.DataFrame, path: Path, unit: str = "words",
                          limits=LENGTH_LIMITS) -> None:
    """Histogram độ dài; vạch dọc = các giá trị max_len đang xét, kèm % review dài hơn."""
    fig, ax = plt.subplots(figsize=(7, 4))
    upper = int(np.percentile(lengths["n_units"], 99.5))
    bins = np.linspace(0, upper, 80)
    for split, g in lengths.groupby("split", sort=False):
        ax.hist(g["n_units"].clip(upper=upper), bins=bins, alpha=0.5, label=split)
    top = ax.get_ylim()[1]
    for i, limit in enumerate(limits):
        pct = (lengths["n_units"] > limit).mean() * 100
        ax.axvline(limit, color="k", ls="--", lw=1)
        # so le độ cao để chữ của các vạch gần nhau không chồng lên nhau
        ax.text(limit + 8, top * (0.97 - 0.16 * i), f">{limit}: {pct:.1f}%", fontsize=8, va="top",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1})
    ax.set_xlabel(f"{unit} per review (cleaned; display clipped at P99.5)")
    ax.set_ylabel("reviews")
    ax.set_title(f"Review length ({unit})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_length_by_label(lengths: pd.DataFrame, path: Path, unit: str = "words",
                         limits=LENGTH_LIMITS) -> None:
    """Box plot độ dài theo nhãn (không vẽ điểm ngoại lai)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    data = [lengths.loc[lengths["label"] == lab, "n_units"] for lab in (0, 1)]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks([1, 2], ["negative (0)", "positive (1)"])
    for limit in limits:
        ax.axhline(limit, color="k", ls="--", lw=0.8)
    ax.set_ylabel(unit)
    ax.set_title("Length by label (no outliers)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- B3: từ phổ biến, phủ định, ca biên
# Danh sách stopword tiếng Anh thường gặp (giữ trong file để kết quả không phụ thuộc phiên bản thư viện).
# "not"/"no"/"never" nằm trong danh sách nhưng phần phủ định phân tích riêng bên dưới.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren't as at be because been before being
below between both but by can cannot could did do does doing don't down during each few for from further
had has have having he her here hers herself him himself his how i if in into is isn't it its itself just
me more most my myself no nor not now of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there these they this those
through to too under until up very was wasn't we were weren't what when where which while who whom why
will with won't would would've you your yours yourself yourselves
it's that's i'm i've i'd i'll you're you've you'd you'll he's he'd he'll she's she'd she'll
we're we've we'd we'll they're they've they'd they'll isn't aren't wasn't weren't can't cannot
couldn't shouldn't wouldn't don't doesn't didn't haven't hasn't hadn't let's there's here's what's
who's really even much many get got also still though however upon
""".split())

_EDGE_PUNCT = "\"'`.,!?;:()[]{}<>*_-–—"


def _words(text: str) -> list[str]:
    """Từ (chữ thường) của văn bản đã làm sạch; bỏ dấu câu ở hai đầu."""
    out = []
    for w in clean_text(text).lower().split():
        w = w.strip(_EDGE_PUNCT)
        if w:
            out.append(w)
    return out


def word_counts(texts, drop_stopwords: bool = False) -> Counter:
    """Đếm từ trên văn bản đã làm sạch, chữ thường (từ 2 ký tự trở lên)."""
    counts: Counter = Counter()
    for text in texts:
        for w in _words(text):
            if len(w) < 2:
                continue
            if drop_stopwords and w in STOPWORDS:
                continue
            counts[w] += 1
    return counts


def top_words_table(parts: dict[str, pd.DataFrame], n: int = 25, drop_stopwords: bool = False) -> pd.DataFrame:
    """Từ phổ biến của từng lớp, từng split: cột split, label, rank, word, count."""
    rows = []
    for name, df in parts.items():
        for label, sub in df.groupby("label"):
            counts = word_counts(sub["text"].tolist(), drop_stopwords)
            for rank, (word, count) in enumerate(counts.most_common(n), start=1):
                rows.append({"split": name, "label": int(label), "rank": rank, "word": word, "count": int(count)})
    return pd.DataFrame(rows, columns=["split", "label", "rank", "word", "count"])


def negation_summary(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """% review có ít nhất một dấu hiệu phủ định, theo split × nhãn."""
    rows = []
    for name, df in parts.items():
        for label, sub in df.groupby("label"):
            texts = [clean_text(t) for t in sub["text"]]
            n_cue = int(sum(bool(NEGATION_RE.search(t)) for t in texts))
            rows.append({
                "split": name, "label": int(label), "n_reviews": len(texts),
                "n_with_negation": n_cue,
                "pct_with_negation": round(100 * n_cue / max(len(texts), 1), 3),
            })
    return pd.DataFrame(rows)


def negation_cue_table(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Số lần từng dấu hiệu phủ định xuất hiện ở mỗi lớp (đếm mọi lần, trên cả 3 split)."""
    counts = {0: {}, 1: {}}
    for df in parts.values():
        for label, sub in df.groupby("label"):
            for text in sub["text"]:
                for m in NEGATION_RE.finditer(clean_text(text)):
                    cue = m.group(0).lower()
                    counts[int(label)][cue] = counts[int(label)].get(cue, 0) + 1
    rows = []
    for cue in sorted(set(counts[0]) | set(counts[1])):
        rows.append({
            "cue": cue, "count_negative": counts[0].get(cue, 0),
            "count_positive": counts[1].get(cue, 0),
            "total": counts[0].get(cue, 0) + counts[1].get(cue, 0),
        })
    return pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)


def negation_bigram_table(parts: dict[str, pd.DataFrame], n: int = 20) -> pd.DataFrame:
    """Cụm 2 từ bắt đầu bằng dấu hiệu phủ định ("not good", "n't work"…) — mẫu phủ định thật sự."""
    counts = {0: Counter(), 1: Counter()}
    for df in parts.values():
        for label, sub in df.groupby("label"):
            for text in sub["text"]:
                words = _words(text)
                for cur, nxt in zip(words, words[1:]):
                    if cur.endswith("n't") or cur in {"not", "no", "never", "nothing", "none", "nor", "neither", "nobody", "without"}:
                        counts[int(label)][f"{cur} {nxt}"] += 1
    rows = []
    for label in (0, 1):
        for phrase, count in counts[label].most_common(n):
            rows.append({"label": label, "phrase": phrase, "count": int(count)})
    return pd.DataFrame(rows, columns=["label", "phrase", "count"])


def length_extremes(parts: dict[str, pd.DataFrame], short_max: int = 10, long_min: int = 300) -> pd.DataFrame:
    """Số review rất ngắn (≤ short_max từ) và rất dài (≥ long_min từ) mỗi split."""
    rows = []
    for name, df in parts.items():
        lens = token_lengths([clean_text(t) for t in df["text"]])
        rows.append({
            "split": name, "n_reviews": len(lens),
            "n_short": int((lens <= short_max).sum()),
            "pct_short": round(float((lens <= short_max).mean() * 100), 3),
            "n_long": int((lens >= long_min).sum()),
            "pct_long": round(float((lens >= long_min).mean() * 100), 3),
            "min_words": int(lens.min()), "max_words": int(lens.max()),
        })
    return pd.DataFrame(rows)


_HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")
_LETTER_RE = re.compile(r"[A-Za-z]")


def quality_issues(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Bảng kiểm chất lượng dữ liệu: mỗi dòng = một kiểm tra × split (hoặc toàn cục)."""
    rows = []

    def add(scope: str, split: str, check: str, count: int, detail: str = "") -> None:
        if count:
            rows.append({"scope": scope, "split": split, "check": check, "count": int(count), "detail": detail})

    for name, df in parts.items():
        clean = df["text_clean"] if "text_clean" in df.columns else df["text"].map(clean_text)
        lens = token_lengths(clean.tolist())
        hashes = df["text"].map(text_hash)
        add("split", name, "review trùng nội dung trong split (bản thừa)", int(hashes.duplicated().sum()),
            "giữ 1 bản; test cố ý không xoá")
        add("split", name, "label không khớp rating", int(
            (((df["rating"] <= 4) & (df["label"] == 1)) | ((df["rating"] >= 7) & (df["label"] == 0))).sum()))
        add("split", name, "text_clean rỗng", int((clean.str.len() == 0).sum()))
        add("split", name, "text_clean dưới 3 từ", int((lens < 3).sum()))
        add("split", name, "còn thẻ HTML khác <br /> trong text_clean",
            int(clean.str.contains(_HTML_TAG_RE, regex=True).sum()), "ví dụ <i>, </i> — clean_text chỉ bỏ <br />")
        add("split", name, "không có chữ cái nào", int((~clean.str.contains(_LETTER_RE, regex=True)).sum()))
        add("split", name, "rating ngoài {1–4, 7–10}", int((~df["rating"].isin([1, 2, 3, 4, 7, 8, 9, 10])).sum()))

    holdout = pd.concat([parts[s] for s in ("train", "val") if s in parts])
    if "test" in parts:
        add("global", "train/val–test", "hash chung giữa train/val và test",
            len(set(holdout["text"].map(text_hash)) & set(parts["test"]["text"].map(text_hash))), "rò rỉ nội dung")
    if "train" in parts and "val" in parts:
        add("global", "train–val", "movie_id chung giữa train và val",
            len(set(parts["train"]["movie_id"]) & set(parts["val"]["movie_id"])))
    if "test" in parts:
        add("global", "train/val–test", "movie_id chung giữa train/val và test",
            len(set(holdout["movie_id"]) & set(parts["test"]["movie_id"])), "rò rỉ ở mức phim")
        add("global", "test", "review trùng nội dung trong test (bản thừa, giữ nguyên)",
            int(parts["test"]["text"].map(text_hash).duplicated().sum()))
    return pd.DataFrame(rows, columns=["scope", "split", "check", "count", "detail"])


def plot_top_words(parts: dict[str, pd.DataFrame], path: Path, n: int = 20) -> None:
    """Hai cột ngang: từ phổ biến của lớp negative và positive (đã bỏ stopword)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, label, title in zip(axes, (0, 1), ("Negative (0)", "Positive (1)")):
        texts = pd.concat([df.loc[df["label"] == label, "text"] for df in parts.values()])
        counts = word_counts(texts.tolist(), drop_stopwords=True).most_common(n)[::-1]
        words = [w for w, _ in counts]
        values = [c for _, c in counts]
        ax.barh(words, values, color="#c44e52" if label == 0 else "#4c72b0")
        ax.set_xlabel("count (cleaned text, stopwords removed)")
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Most common words by sentiment (train + val + test)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_negation(neg: pd.DataFrame, path: Path) -> None:
    """Cột: % review có dấu hiệu phủ định theo split × nhãn."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    splits = list(dict.fromkeys(neg["split"]))
    x = np.arange(len(splits))
    for offset, label, color in ((-0.2, 0, "#c44e52"), (0.2, 1, "#4c72b0")):
        vals = []
        for s in splits:
            sub = neg[(neg["split"] == s) & (neg["label"] == label)]["pct_with_negation"]
            vals.append(float(sub.iloc[0]) if len(sub) else 0.0)
        ax.bar(x + offset, vals, 0.4, label=f"label {label}", color=color)
    ax.set_xticks(x, splits)
    ax.set_ylabel("% reviews with a negation cue")
    ax.set_title("Negation cues by split and label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_length_extremes(ext: pd.DataFrame, path: Path) -> None:
    """Cột: % review rất ngắn (≤10 từ) và rất dài (≥300 từ) mỗi split."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, col, title in ((axes[0], "pct_short", "Very short reviews (≤10 words)"),
                           (axes[1], "pct_long", "Very long reviews (≥300 words)")):
        ax.bar(ext["split"], ext[col], color="#55a868")
        ax.set_ylabel("% of split")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_extended_eda(parts: dict[str, pd.DataFrame], results_dir: Path, plots_dir: Path,
                     n_words: int = 25) -> dict:
    """B3: từ phổ biến, phủ định, ca biên, kiểm chất lượng — ghi CSV + hình; trả về bảng."""
    results_dir, plots_dir = Path(results_dir), Path(plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    top_all = top_words_table(parts, n_words, drop_stopwords=False)
    top_content = top_words_table(parts, n_words, drop_stopwords=True)
    neg = negation_summary(parts)
    cues = negation_cue_table(parts)
    bigrams = negation_bigram_table(parts)
    extremes = length_extremes(parts)
    issues = quality_issues(parts)

    paths = {
        "top_words": results_dir / "eda_top_words.csv",
        "top_words_content": results_dir / "eda_top_words_content.csv",
        "negation": results_dir / "eda_negation.csv",
        "negation_cues": results_dir / "eda_negation_cues.csv",
        "negation_bigrams": results_dir / "eda_negation_bigrams.csv",
        "extremes": results_dir / "eda_length_extremes.csv",
        "issues": results_dir / "eda_quality_issues.csv",
        "fig_top_words": plots_dir / "eda_top_words.png",
        "fig_negation": plots_dir / "eda_negation_share.png",
        "fig_extremes": plots_dir / "eda_length_extremes.png",
    }
    top_all.to_csv(paths["top_words"], index=False)
    top_content.to_csv(paths["top_words_content"], index=False)
    neg.to_csv(paths["negation"], index=False)
    cues.to_csv(paths["negation_cues"], index=False)
    bigrams.to_csv(paths["negation_bigrams"], index=False)
    extremes.to_csv(paths["extremes"], index=False)
    issues.to_csv(paths["issues"], index=False)
    plot_top_words(parts, paths["fig_top_words"])
    plot_negation(neg, paths["fig_negation"])
    plot_length_extremes(extremes, paths["fig_extremes"])
    return {
        "paths": paths, "top_words": top_all, "top_words_content": top_content,
        "negation": neg, "negation_cues": cues, "negation_bigrams": bigrams,
        "extremes": extremes, "issues": issues,
    }


# ---------------------------------------------------------------- chạy
def run_eda(parts: dict[str, pd.DataFrame], results_dir: Path, plots_dir: Path,
            tokenize: Tokenize = word_split, unit: str = "words", limits=LENGTH_LIMITS) -> dict:
    """Ghi eda_summary.csv, eda_ratings.csv, eda_lengths.csv và 3 hình; trả về dict đường dẫn + bảng.

    `parts`: {tên split: bảng}, ví dụ {"train": ..., "val": ..., "test": ...}.
    Bảng trùng lặp không nằm ở đây vì cần đúng cặp train chính thức / test: gọi `duplicate_counts`.
    """
    results_dir, plots_dir = Path(results_dir), Path(plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows, length_parts = [], []
    for name, df in parts.items():
        row, lens = split_summary(name, df, tokenize, limits)
        rows.append(row)
        length_parts.append(pd.DataFrame({
            "uid": df["uid"].values, "split": name,
            "label": df["label"].values, "rating": df["rating"].values, "n_units": lens,
        }))
    summary = pd.DataFrame(rows)
    lengths = pd.concat(length_parts, ignore_index=True)
    ratings = (pd.concat([df.assign(split=name) for name, df in parts.items()])
               .groupby(["split", "rating"]).size().unstack("split", fill_value=0).reset_index())

    paths = {
        "summary": results_dir / "eda_summary.csv",
        "ratings": results_dir / "eda_ratings.csv",
        "lengths": results_dir / "eda_lengths.csv",
        "fig_labels": plots_dir / "eda_label_distribution.png",
        "fig_lengths": plots_dir / "eda_length_hist.png",
        "fig_by_label": plots_dir / "eda_length_by_label.png",
    }
    summary.round(3).to_csv(paths["summary"], index=False)
    ratings.to_csv(paths["ratings"], index=False)
    lengths.to_csv(paths["lengths"], index=False)
    plot_label_distribution(summary, paths["fig_labels"])
    plot_length_histogram(lengths, paths["fig_lengths"], unit, limits)
    plot_length_by_label(lengths, paths["fig_by_label"], unit, limits)
    return {"paths": paths, "summary": summary, "ratings": ratings, "lengths": lengths}


def load_processed_parts(path: Path | None = None) -> dict[str, pd.DataFrame]:
    """Đọc `data/processed/reviews.parquet` (B1) thành {"train", "val", "test"}."""
    from src.utils.paths import data_dir

    path = Path(path) if path is not None else data_dir("processed") / "reviews.parquet"
    if not path.exists():
        raise SystemExit(f"Chưa có {path} — chạy `python -m src.data.prepare` trước (B1).")
    df = pd.read_parquet(path)
    return {name: df[df["split"] == name].reset_index(drop=True) for name in ("train", "val", "test")}


def main() -> None:
    import sys

    from src.utils.paths import experiment_dir

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parts = load_processed_parts()
    results, plots = experiment_dir("results", "eda"), experiment_dir("plots", "eda")
    out = run_eda(parts, results, plots)
    ext = run_extended_eda(parts, results, plots)
    dups = pd.DataFrame([duplicate_counts(parts["train"], parts["test"])])
    dups.to_csv(results / "eda_duplicates.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print("== Tổng quan split (dữ liệu đã loại trùng) ==")
        print(out["summary"].round(2).T)
        print("\n== Review trùng còn lại (test giữ nguyên) ==")
        print(dups.T)
        print("\n== % review có phủ định ==")
        print(ext["negation"].to_string(index=False))
        print("\n== Ca biên ==")
        print(ext["extremes"].to_string(index=False))
        print("\n== Vấn đề chất lượng ==")
        print(ext["issues"].to_string(index=False) if len(ext["issues"]) else "không có")
        print("\n== Top 10 từ của mỗi lớp (đã bỏ stopword, cả 3 split) ==")
        print(ext["top_words_content"][ext["top_words_content"]["rank"] <= 10]
              .pivot_table(index="rank", columns=["split", "label"], values="word", aggfunc="first").to_string())
    print("\nĐã ghi:")
    for name, path in {**out["paths"], **ext["paths"]}.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
