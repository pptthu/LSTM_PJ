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
NEGATION_RE = re.compile(r"\b(?:not|no|never|nothing|none|nor|neither|nobody|without)\b|n't", re.IGNORECASE)

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


def main() -> None:
    import sys

    from src.data.download import download_imdb, load_split
    from src.data.split import apply_split
    from src.utils.paths import experiment_dir

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    download_imdb()
    train_all, test = load_split("train"), load_split("test")
    train, val = apply_split(train_all)
    out = run_eda({"train": train, "val": val, "test": test},
                  experiment_dir("results", "eda"), experiment_dir("plots", "eda"))
    dups = pd.DataFrame([duplicate_counts(train_all, test)])
    dups.to_csv(experiment_dir("results", "eda") / "eda_duplicates.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(out["summary"].round(2).T)
        print(dups.T)


if __name__ == "__main__":
    main()
