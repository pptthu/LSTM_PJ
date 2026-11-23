"""Test bước chuẩn bị dữ liệu (B1): loại trùng, giữ test nguyên, đúng schema, tất định.

Dùng bảng tổng hợp nhỏ (không cần tải 84 MB) + cây aclImdb giả trong `tests/conftest.py`.
"""
import pandas as pd
import pytest

from src.data.prepare import (
    COLUMNS,
    build_processed,
    check_contract,
    committed_stats,
    prepare_dataset,
    remove_duplicates,
)
from src.data.split import make_train_val_split


def frame(rows):
    """rows: (uid, split, text, label, rating, movie_id) → DataFrame đúng cột đầu vào."""
    return pd.DataFrame(rows, columns=["uid", "split", "text", "label", "rating", "movie_id"])


EMPTY = frame([])


# ---------------------------------------------------------------- loại trùng
def test_duplicate_within_train_keeps_smallest_uid():
    train = frame([
        ("train_pos_9", "train", "Same review.", 1, 9, "tt1"),
        ("train_pos_2", "train", "Same review.", 1, 8, "tt2"),
        ("train_pos_3", "train", "Different.", 1, 10, "tt2"),
    ])
    parts, stats = remove_duplicates(train, EMPTY, EMPTY)
    assert parts["train"]["uid"].tolist() == ["train_pos_2", "train_pos_3"]
    assert stats["removed_within_split"]["train"] == 1
    assert stats["outputs"]["train"] == 2


def test_cross_split_duplicates_are_dropped_from_train_and_val_only():
    train = frame([("train_pos_0", "train", "Great film.", 1, 9, "tt1")])
    val = frame([("train_pos_1", "train", "Great film.", 1, 9, "tt2")])
    test = frame([("test_pos_0", "test", "Great film.<br />", 1, 9, "tt9")])  # <br /> không đổi hash
    parts, stats = remove_duplicates(train, val, test)
    assert len(parts["train"]) == 0 and len(parts["val"]) == 0
    assert parts["test"]["uid"].tolist() == ["test_pos_0"]
    assert stats["removed_cross_split"] == {"train": 1, "val": 1}


def test_test_split_is_never_modified_even_with_internal_duplicates():
    test = frame([
        ("test_neg_0", "test", "Bad film.", 0, 1, "tt1"),
        ("test_neg_1", "test", "Bad film.", 0, 1, "tt1"),
        ("test_pos_0", "test", "Good film.", 1, 9, "tt2"),
    ])
    parts, stats = remove_duplicates(EMPTY, EMPTY, test)
    pd.testing.assert_frame_equal(parts["test"], test)
    assert stats["duplicates_in_test_kept"] == 1
    assert stats["test_unchanged"] is True


def test_movie_present_in_test_is_removed_from_train_and_val():
    train = frame([("train_neg_0", "train", "Bad one.", 0, 2, "tt0406816")])
    val = frame([("train_pos_0", "train", "Good one.", 1, 9, "tt0406816")])
    test = frame([("test_neg_0", "test", "Other text.", 0, 3, "tt0406816")])
    parts, stats = remove_duplicates(train, val, test)
    assert len(parts["train"]) == 0 and len(parts["val"]) == 0
    assert stats["removed_movie_leak"]["train"] == 1
    assert stats["removed_movie_leak"]["val"] == 1
    assert stats["removed_movie_leak"]["movie_ids"] == ["tt0406816"]


def test_duplicate_group_with_conflicting_labels_is_counted_not_silently_merged():
    train = frame([
        ("train_pos_1", "train", "Very good film.", 1, 9, "tt1"),
        ("train_neg_1", "train", "Very good film.", 0, 3, "tt2"),
    ])
    _, stats = remove_duplicates(train, EMPTY, EMPTY)
    assert stats["label_conflicts_in_duplicate_groups"]["train"] == 1
    assert stats["removed_within_split"]["train"] == 1


def test_counts_add_up():
    train = frame([
        ("a", "train", "x1", 1, 9, "tt1"),
        ("b", "train", "x1", 1, 9, "tt2"),   # trùng trong train
        ("c", "train", "x2", 0, 2, "tt3"),   # trùng với test
    ])
    val = frame([("d", "train", "x3", 1, 8, "tt4")])
    test = frame([("e", "test", "x2", 0, 2, "tt9")])
    _, stats = remove_duplicates(train, val, test)
    assert stats["outputs"]["train"] == stats["inputs"]["train"] - 2
    assert stats["outputs"]["val"] == stats["inputs"]["val"]
    assert stats["outputs"]["test"] == stats["inputs"]["test"]


# ---------------------------------------------------------------- schema / hợp đồng
def test_build_processed_returns_contract_columns_in_order():
    parts = {
        "train": frame([("train_pos_1", "train", "Good.<br />", 1, 9, "tt1")]),
        "val": frame([("train_neg_1", "train", "Bad.", 0, 2, "tt2")]),
        "test": frame([("test_pos_1", "test", "Nice.", 1, 8, "tt3")]),
    }
    out = build_processed(parts)
    assert list(out.columns) == COLUMNS
    assert out["split"].tolist() == ["train", "val", "test"]  # thứ tự split cố định
    assert out["text_clean"].tolist() == ["Good.", "Bad.", "Nice."]
    assert out[["label", "rating"]].dtypes.tolist() == ["int64", "int64"]
    assert out["movie_id"].iloc[0] == "tt1"


def test_build_processed_is_deterministic_and_preserves_labels():
    parts = {
        "train": frame([
            ("train_pos_2", "train", "two", 1, 9, "tt1"),
            ("train_neg_1", "train", "one", 0, 3, "tt2"),
        ]),
        "val": EMPTY,
        "test": frame([("test_neg_1", "test", "three", 0, 1, "tt3")]),
    }
    first, second = build_processed(parts), build_processed(parts)
    pd.testing.assert_frame_equal(first, second)
    by_uid = first.set_index("uid")["label"]
    assert by_uid["train_pos_2"] == 1 and by_uid["train_neg_1"] == 0 and by_uid["test_neg_1"] == 0


def test_build_processed_rejects_missing_columns():
    bad = pd.DataFrame({"uid": ["a"], "text": ["hello"]})
    with pytest.raises(ValueError, match="missing columns"):
        build_processed({"train": bad})


def test_build_processed_rejects_unknown_split():
    with pytest.raises(ValueError, match="unknown split"):
        build_processed({"holdout": EMPTY})


def test_check_contract_detects_leakage():
    parts = {
        "train": frame([("train_pos_1", "train", "Same.", 1, 9, "tt1")]),
        "val": EMPTY,
        "test": frame([("test_pos_1", "test", "Same.", 1, 9, "tt2")]),
    }
    with pytest.raises(ValueError, match="leakage"):
        build_processed(parts)


def test_check_contract_rejects_movie_leak_between_splits():
    parts = {
        "train": frame([("train_pos_1", "train", "a", 1, 9, "tt0406816")]),
        "val": EMPTY,
        "test": frame([("test_pos_1", "test", "b", 1, 9, "tt0406816")]),
    }
    with pytest.raises(ValueError, match="movie_id shared between splits"):
        build_processed(parts)


def test_committed_stats_drops_generated_at():
    stats = {"n": 1, "generated_at": "2026-11-23T00:00:00+00:00"}
    assert committed_stats(stats) == {"n": 1}


def test_content_hash_depends_on_content_not_on_file_bytes():
    out = build_processed({
        "train": frame([("t1", "train", "a", 1, 9, "tt1")]),
        "val": EMPTY,
        "test": frame([("s1", "test", "b", 1, 8, "tt3")]),
    })
    expected = int(pd.util.hash_pandas_object(out, index=False).sum())
    assert expected == int(pd.util.hash_pandas_object(out.copy(), index=False).sum())
    changed = out.copy()
    changed.loc[0, "text"] = "different"
    assert int(pd.util.hash_pandas_object(changed, index=False).sum()) != expected


def test_check_contract_reports_clean_data():
    out = build_processed({
        "train": frame([("t1", "train", "a", 1, 9, "tt1")]),
        "val": frame([("v1", "train", "b", 0, 2, "tt2")]),
        "test": frame([("s1", "test", "c", 1, 8, "tt3")]),
    })
    contract = check_contract(out)
    assert contract["hash_overlap_trainval_test"] == 0
    assert contract["movie_overlap_train_val"] == 0
    assert contract["label_rating_conflicts"] == 0


# ---------------------------------------------------------------- đầu vào rỗng
def test_empty_inputs_are_handled():
    parts, stats = remove_duplicates(EMPTY, EMPTY, EMPTY)
    out = build_processed(parts)
    assert out.empty and list(out.columns) == COLUMNS
    assert stats["outputs"] == {"train": 0, "val": 0, "test": 0}
    assert stats["test_unchanged"] is True


def test_missing_columns_raise():
    bad = pd.DataFrame({"uid": ["x"], "text": ["t"]})
    with pytest.raises(ValueError, match="missing columns"):
        remove_duplicates(bad, EMPTY, EMPTY)


# ---------------------------------------------------------------- chạy trên cây giả
def test_prepare_dataset_end_to_end_on_fake_imdb(fake_imdb_root, tmp_path):
    from src.data.download import load_split
    from src.data.split import apply_split

    train_all = load_split("train", fake_imdb_root)
    test = load_split("test", fake_imdb_root)
    split = make_train_val_split(train_all, seed=0, n_folds=2)
    out_path = tmp_path / "reviews.parquet"

    results_dir = tmp_path / "results"
    processed, stats = prepare_dataset(
        root=fake_imdb_root, split=split, out_path=out_path, results_dir=results_dir
    )

    loaded = pd.read_parquet(out_path)
    pd.testing.assert_frame_equal(loaded, processed)
    assert list(loaded.columns) == COLUMNS
    assert stats["contract"]["hash_overlap_trainval_test"] == 0
    assert stats["outputs"]["test"] == len(test)  # test giữ nguyên số dòng
    train, val = apply_split(train_all, split)
    assert stats["inputs"] == {"train": len(train), "val": len(val), "test": len(test)}
    # "Very good film." xuất hiện 2 lần ở train gốc (2 phim) nhưng không có trong test → còn 1 bản
    kept = loaded[loaded["text"] == "Very good film."]
    assert len(kept) == 1 and kept.iloc[0]["split"] in {"train", "val"}
    # "Good acting and good plot." có trong test → bỏ khỏi train
    assert not (
        (loaded["split"] == "train") & (loaded["text_clean"] == "Good acting and good plot.")
    ).any()
    # danh sách uid test trùng train: đúng review "Good acting and good plot." phía test
    uids_file = results_dir / "test_uids_dup_train.txt"
    lines = uids_file.read_text(encoding="utf-8").splitlines()
    assert lines == ["test_pos_1"]
    assert stats["test_uids_dup_train"]["n"] == 1
    assert stats["test_uids_dup_train"]["file"].endswith("test_uids_dup_train.txt")
    assert "content_hash" in stats
