"""Test phần EDA mở rộng (B3): từ phổ biến, phủ định, ca biên, kiểm chất lượng."""
import pandas as pd

from src.data.eda import (
    length_extremes,
    negation_bigram_table,
    negation_cue_table,
    negation_summary,
    quality_issues,
    run_extended_eda,
    top_words_table,
    word_counts,
)


def frame(rows):
    return pd.DataFrame(rows, columns=["uid", "split", "text", "text_clean", "label", "rating", "movie_id"])


PARTS = {
    "train": frame([
        ("t1", "train", "The movie was not good.<br />Bad acting.", "The movie was not good. Bad acting.", 0, 2, "tt1"),
        ("t2", "train", "A great film, really great.", "A great film, really great.", 1, 9, "tt2"),
    ]),
    "val": frame([
        ("v1", "train", "I don't like it.", "I don't like it.", 0, 3, "tt3"),
    ]),
    "test": frame([
        ("s1", "test", "Good story.", "Good story.", 1, 8, "tt4"),
    ]),
}


def test_word_counts_lowercases_and_drops_stopwords():
    counts = word_counts(["The Movie was GOOD", "good movie."])
    assert counts["movie"] == 2 and counts["good"] == 2
    assert word_counts(["The Movie was GOOD"], drop_stopwords=True) == {"movie": 1, "good": 1}


def test_top_words_table_has_ranks_per_split_and_label():
    top = top_words_table(PARTS, n=5, drop_stopwords=True)
    assert list(top.columns) == ["split", "label", "rank", "word", "count"]
    assert set(top["split"]) == {"train", "val", "test"}
    assert top[top["split"] == "train"].groupby("label")["rank"].max().max() <= 5


def test_negation_summary_and_cues():
    neg = negation_summary(PARTS)
    train_neg = neg[(neg["split"] == "train") & (neg["label"] == 0)].iloc[0]
    train_pos = neg[(neg["split"] == "train") & (neg["label"] == 1)].iloc[0]
    assert train_neg["pct_with_negation"] == 100.0
    assert train_pos["pct_with_negation"] == 0.0
    cues = negation_cue_table(PARTS).set_index("cue")
    assert cues.loc["not", "count_negative"] == 1
    assert cues.loc["n't", "count_negative"] == 1


def test_negation_bigrams_keep_options_after_cue():
    bigrams = negation_bigram_table(PARTS)
    neg = set(bigrams[bigrams["label"] == 0]["phrase"])
    assert "not good" in neg


def test_length_extremes_counts_short_and_long():
    parts = {"train": frame([
        ("a", "train", "one two", "one two", 1, 9, "tt1"),
        ("b", "train", " ".join(["word"] * 300), " ".join(["word"] * 300), 0, 2, "tt2"),
    ])}
    ext = length_extremes(parts, short_max=10, long_min=300).iloc[0]
    assert ext["n_short"] == 1 and ext["n_long"] == 1
    assert ext["min_words"] == 2 and ext["max_words"] == 300


def test_quality_issues_flags_leak_and_label_conflict():
    parts = {
        "train": frame([
            ("t1", "train", "Same.", "Same.", 1, 9, "tt1"),
            ("t2", "train", "Other.", "Other.", 1, 3, "tt1"),  # label 1 nhưng rating 3
        ]),
        "val": frame([]),
        "test": frame([("s1", "test", "Same.", "Same.", 1, 9, "tt9")]),
    }
    issues = quality_issues(parts).set_index(["scope", "split", "check"])["count"].sort_index()
    assert issues[("global", "train/val–test", "hash chung giữa train/val và test")] == 1
    assert issues[("split", "train", "label không khớp rating")] == 1


def test_quality_issues_flags_missing_values():
    train = frame([("t1", "train", "ok text", "ok text", 1, 9, "tt1")])
    train.loc[0, "text_clean"] = None
    issues = quality_issues({"train": train}).set_index(["scope", "split", "check"])["count"].sort_index()
    assert issues[("split", "train", "ô bị thiếu (NaN) ở cột bắt buộc")] == 1


def test_run_extended_eda_writes_all_files(tmp_path):
    out = run_extended_eda(PARTS, tmp_path / "results", tmp_path / "plots")
    for path in out["paths"].values():
        assert path.exists() and path.stat().st_size > 0
    assert (tmp_path / "results" / "eda_top_words_content.csv").exists()
    assert (tmp_path / "plots" / "eda_top_words.png").exists()
