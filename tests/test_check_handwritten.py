"""Test kiểm bộ 30 câu tự viết: định dạng và đo trùng cụm từ với IMDb."""
import pandas as pd
import pytest

from src.data.check_handwritten import check_format, imdb_overlap
from src.utils.paths import data_dir


def test_committed_file_is_balanced_and_valid():
    df = pd.read_csv(data_dir("sample") / "handwritten_30.csv")
    r = check_format(df)
    assert r["n_rows"] == 30
    assert r["label_totals"] == {"0": 15, "1": 15}
    assert all(n == 5 for n in df["case_type"].value_counts())


def test_check_format_rejects_unknown_case_type():
    df = pd.DataFrame({"text": ["a"], "label": [1], "case_type": ["irony"]})
    with pytest.raises(ValueError, match="unknown case_type"):
        check_format(df)


def test_imdb_overlap_finds_shared_ngram_and_exact_copy():
    df = pd.DataFrame({
        "text": ["It is not a bad film at all, I liked it.", "Completely new words here for testing only."],
        "label": [1, 1], "case_type": ["negation", "positive"],
    })
    corpus = ["Honestly it is not a bad film at all.<br />Recommended."]
    out = imdb_overlap(df, corpus, n=6)
    assert [o["row"] for o in out] == [0]
    assert "not a bad film at all" in out[0]["shared_ngrams"]
    assert out[0]["exact_in_imdb"] is False
