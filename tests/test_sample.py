"""Test chọn mẫu 100 review test (B4): đúng 100 dòng, cân bằng nhãn, tất định, không lấy ngoài test."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.sample import COLUMNS, select_sample

SHORT = "Short movie, not good."
LONG = " ".join(["word"] * 550) + " but not bad"
MID = "An ordinary film with some good scenes."


def make_test(n_per_label: int = 100) -> pd.DataFrame:
    rows = []
    for label, ratings in ((0, (1, 2, 3, 4)), (1, (7, 8, 9, 10))):
        prefix = "neg" if label == 0 else "pos"
        for i in range(n_per_label):
            rating = ratings[i % len(ratings)]
            if i % 10 == 0:
                text = f"{SHORT} #{i}"
            elif i % 10 == 1:
                text = f"{LONG} #{i}"
            elif i % 7 == 0:
                text = f"Review {i}: the plot is not good at all."
            else:
                text = f"{MID} #{i}"
            rows.append({
                "uid": f"test_{prefix}_{i}", "split": "test", "text": text, "text_clean": text,
                "label": label, "rating": rating, "movie_id": f"tt{i:07d}",
            })
    return pd.DataFrame(rows)


def test_select_sample_size_balance_and_columns():
    out, stats = select_sample(make_test())
    assert len(out) == 100
    assert list(out.columns) == COLUMNS
    assert out["label"].value_counts().to_dict() == {0: 50, 1: 50}
    assert stats["by_label"] == {0: 50, 1: 50}
    assert (out["source_file"] == "test").all()


def test_select_sample_is_deterministic():
    test = make_test()
    first, _ = select_sample(test, seed=7)
    second, _ = select_sample(test, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_select_sample_only_uses_test_rows_and_preserves_labels():
    test = make_test()
    out, _ = select_sample(test)
    source = test.set_index("uid")
    assert set(out["uid"]) <= set(test["uid"])
    expected = source["label"].reindex(out["uid"]).to_numpy()
    assert (out["label"].to_numpy() == expected).all()
    assert not out["text"].duplicated().any()


def test_select_sample_contains_hard_cases():
    out, _ = select_sample(make_test())
    by_case = out["case_type"].value_counts().to_dict()
    assert by_case["rating_4_negative"] == 12
    assert by_case["rating_7_positive"] == 12
    assert by_case["negation"] >= 1
    assert by_case["long_review"] >= 1
    assert by_case["short_review"] >= 1
    short_lens = out.loc[out["case_type"] == "short_review", "text"].str.split().str.len()
    assert short_lens.between(1, 20).all()


def test_select_sample_does_not_modify_input():
    test = make_test()
    before = test.copy()
    select_sample(test)
    pd.testing.assert_frame_equal(test, before)


def test_select_sample_rejects_odd_size_and_missing_columns():
    with pytest.raises(ValueError, match="even"):
        select_sample(make_test(), n=99)
    with pytest.raises(ValueError, match="missing columns"):
        select_sample(make_test().drop(columns=["rating"]))


def test_committed_sample_file_is_valid():
    """File đã commit phải đúng 100 dòng, 50/50, chỉ lấy từ test."""
    path = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_100.csv"
    if not path.exists():  # bản checkout chưa có file (nhánh khác) — bỏ qua
        pytest.skip("data/sample/sample_100.csv not present")
    sample = pd.read_csv(path)
    assert len(sample) == 100
    assert sample["label"].value_counts().to_dict() == {0: 50, 1: 50}
    assert sample["uid"].str.startswith("test_").all()
    assert sample["uid"].is_unique
    assert sample["case_type"].notna().all()
    assert {4, 7} <= set(sample["rating"])
