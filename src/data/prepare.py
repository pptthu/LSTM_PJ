"""Gộp train/val/test đã làm sạch, loại review trùng, ghi `data/processed/reviews.parquet` (B1).

Quy tắc loại trùng (docs/DATA.md §4, Thư chốt 2026-11-23):

1. Trùng trong train/val: giữ 1 bản (bản có `uid` nhỏ nhất, để chạy lại cho kết quả y hệt).
2. Trùng giữa train/val và test: bỏ bản ở train/val, giữ nguyên review trong test.
3. Trùng trong test: không xoá, chỉ đếm (test giữ đủ 25.000 review).
4. Review của một phim có mặt trong test: bỏ khỏi train/val (split theo phim, chặn rò rỉ ở mức phim).
   Dữ liệu thật có đúng 1 phim như vậy (`tt0406816`, 23 review test, 2 review train) — DATA.md §4.

Split train/val lấy nguyên từ `data/splits/train_val_split.json` (split Lab 3), không chia lại.
Hash là SHA-1 của văn bản đã làm sạch (`src.data.clean.text_hash`) — cùng cách với Lab 3.

Chạy từ gốc repo:
    python -m src.data.prepare        # reviews.parquet + prepare_stats.json + test_uids_dup_train.txt

Bàn giao cho mô hình: `uid, split, text, text_clean, label, rating, movie_id`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.clean import clean_text, text_hash
from src.data.download import download_imdb, load_split
from src.data.split import apply_split
from src.utils.paths import data_dir, experiment_dir

#: Cột của file bàn giao, đúng thứ tự trong docs/TASKS.md §4.
COLUMNS = ["uid", "split", "text", "text_clean", "label", "rating", "movie_id"]
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
#: Cột tối thiểu phải có ở đầu vào của `remove_duplicates` / `build_processed`.
REQUIRED_COLUMNS = {"uid", "split", "text", "label", "rating", "movie_id"}

HASH_COLUMN = "_hash"


# ---------------------------------------------------------------- kiểm tra đầu vào
def _check_required(df: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def _with_hash(df: pd.DataFrame) -> pd.DataFrame:
    """Bản sao có cột `_hash` = SHA-1 của `text` đã làm sạch (dùng nội bộ)."""
    out = df.copy()
    out[HASH_COLUMN] = out["text"].map(text_hash)
    return out


def _dedupe_within(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Bỏ bản trùng nội dung trong một split, giữ bản `uid` nhỏ nhất.

    Trả về (bảng đã bỏ trùng, số bản bị bỏ, số nhóm trùng có nhãn khác nhau).
    """
    hashed = _with_hash(df).sort_values("uid", kind="stable")
    label_conflicts = int(hashed.groupby(HASH_COLUMN)["label"].nunique().gt(1).sum())
    kept = hashed.drop_duplicates(HASH_COLUMN, keep="first")
    return kept.drop(columns=HASH_COLUMN), len(hashed) - len(kept), label_conflicts


def drop_movie_leak(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Bỏ khỏi train/val review của phim nào có review trong test (split theo phim — DATA.md §3).

    aclImdb tách train/test theo phim, nhưng dữ liệu thật vẫn có 1 phim lọt cả hai phía
    (`tt0406816`). Giữ nguyên test, bỏ bản train/val để không rò rỉ ở mức phim.
    """
    test_movies = set(test["movie_id"])
    train_out = train[~train["movie_id"].isin(test_movies)].copy()
    val_out = val[~val["movie_id"].isin(test_movies)].copy()
    involved = sorted(set(train["movie_id"]) & test_movies)
    return train_out, val_out, {
        "train": len(train) - len(train_out),
        "val": len(val) - len(val_out),
        "movie_ids": involved,
    }


# ---------------------------------------------------------------- loại trùng
def remove_duplicates(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Loại review trùng theo docs/DATA.md §4; test giữ nguyên (không xoá dòng nào).

    Thứ tự cố định để chạy lại cho kết quả y hệt:
    1. bỏ trùng trong train, rồi trong val (giữ `uid` nhỏ nhất);
    2. bỏ khỏi train/val mọi review có hash xuất hiện trong test.
    `test` được trả về nguyên vẹn (chỉ copy, không đổi thứ tự dòng).
    """
    for name, df in (("train", train), ("val", val), ("test", test)):
        _check_required(df, name)

    test_out = test.copy()
    test_hashes = set(test_out["text"].map(text_hash))

    train_dedup, train_within, train_label_conflicts = _dedupe_within(train)
    val_dedup, val_within, val_label_conflicts = _dedupe_within(val)
    test_within = int(test_out["text"].map(text_hash).duplicated().sum())
    test_label_conflicts = int(
        _with_hash(test_out).groupby(HASH_COLUMN)["label"].nunique().gt(1).sum()
    )

    # uid phía TEST của các review trùng với train/val — lưu lại để phase EVAL báo được
    # "test acc khi bỏ các review trùng train" (DATA.md §4) dù parquet đã bỏ bản train/val.
    holdout_hashes = set(train_dedup["text"].map(text_hash)) | set(val_dedup["text"].map(text_hash))
    test_uids_dup_train = sorted(
        test_out.loc[test_out["text"].map(text_hash).isin(holdout_hashes), "uid"].tolist()
    )

    train_out = train_dedup[~train_dedup["text"].map(text_hash).isin(test_hashes)].copy()
    val_out = val_dedup[~val_dedup["text"].map(text_hash).isin(test_hashes)].copy()
    cross_train = len(train_dedup) - len(train_out)
    cross_val = len(val_dedup) - len(val_out)

    train_out, val_out, movie_leak = drop_movie_leak(train_out, val_out, test_out)

    stats = {
        "rule": (
            "trùng trong train/val: giữ 1 bản (uid nhỏ nhất); trùng giữa train/val và test: "
            "bỏ bản ở train/val, giữ test; trùng trong test: chỉ đếm"
        ),
        "hash": "sha1(clean_text) — src.data.clean.text_hash",
        "inputs": {"train": len(train), "val": len(val), "test": len(test)},
        "removed_within_split": {"train": train_within, "val": val_within},
        "removed_cross_split": {"train": cross_train, "val": cross_val},
        "test_reviews_also_in_train": len(test_uids_dup_train),
        # Khoá nội bộ: `prepare_dataset` ghi ra file rồi bỏ khoá này khỏi JSON.
        "_test_uids_dup_train": test_uids_dup_train,
        "removed_movie_leak": movie_leak,
        "duplicates_in_test_kept": test_within,
        "label_conflicts_in_duplicate_groups": {
            "train": train_label_conflicts,
            "val": val_label_conflicts,
            "test": test_label_conflicts,
        },
        "outputs": {"train": len(train_out), "val": len(val_out), "test": len(test_out)},
        "test_unchanged": len(test_out) == len(test),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return {"train": train_out, "val": val_out, "test": test_out}, stats


# ---------------------------------------------------------------- hợp đồng dữ liệu
def add_clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm `text_clean` = `clean_text(text)` (bỏ `<br />`, gộp khoảng trắng; giữ hoa/thường)."""
    _check_required(df, "data")
    out = df.copy()
    out["text_clean"] = out["text"].map(clean_text)
    return out


def build_processed(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Ghép 3 split thành bảng bàn giao: đúng cột, đúng kiểu, sắp theo (split, uid)."""
    frames = []
    for name, df in parts.items():
        if name not in SPLIT_ORDER:
            raise ValueError(f"unknown split {name!r}; expected one of {tuple(SPLIT_ORDER)}")
        part = add_clean_text(df)
        part["split"] = name
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out["label"] = out["label"].astype("int64")
    out["rating"] = out["rating"].astype("int64")
    out = out.assign(_order=out["split"].map(SPLIT_ORDER))
    out = out.sort_values(["_order", "uid"], kind="stable").drop(columns="_order")
    out = out[COLUMNS].reset_index(drop=True)
    check_contract(out)
    return out


def check_contract(df: pd.DataFrame) -> dict:
    """Kiểm hợp đồng dữ liệu; lỗi nghiêm trọng thì raise, còn lại trả về dict số đo.

    - đúng cột / kiểu, không NaN ở text;
    - `label` ∈ {0, 1} và khớp `rating` (≤4 → 0, ≥7 → 1);
    - không hash nào chung giữa train/val và test (không rò rỉ nội dung);
    - không `movie_id` chung giữa train và val / test (split theo phim).
    """
    if list(df.columns) != COLUMNS:
        raise ValueError(f"columns must be {COLUMNS}, got {list(df.columns)}")
    for col in ("uid", "split", "text", "text_clean", "movie_id"):
        if df[col].isna().any():
            raise ValueError(f"{col} has null values")
    if not set(df["split"]) <= set(SPLIT_ORDER):
        raise ValueError(f"split values must be in {tuple(SPLIT_ORDER)}, got {sorted(set(df['split']))}")
    if not set(df["label"]) <= {0, 1}:
        raise ValueError(f"label must be 0/1, got {sorted(set(df['label']))}")

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"
    holdout_mask = train_mask | val_mask

    holdout_hashes = set(df.loc[holdout_mask, "text"].map(text_hash))
    test_hashes = set(df.loc[test_mask, "text"].map(text_hash))
    hash_overlap = len(holdout_hashes & test_hashes)
    if hash_overlap:
        raise ValueError(f"{hash_overlap} reviews shared between train/val and test (leakage)")

    label_conflict = int(
        (((df["rating"] <= 4) & (df["label"] == 1)) | ((df["rating"] >= 7) & (df["label"] == 0))).sum()
    )

    def _overlap(a: str, b: str) -> int:
        return len(set(df.loc[df["split"] == a, "movie_id"]) & set(df.loc[df["split"] == b, "movie_id"]))

    movie_overlaps = {
        "movie_overlap_train_val": _overlap("train", "val"),
        "movie_overlap_train_test": _overlap("train", "test"),
        "movie_overlap_val_test": _overlap("val", "test"),
    }
    if any(movie_overlaps.values()):
        raise ValueError(f"movie_id shared between splits (leakage): {movie_overlaps}")

    return {
        "n_rows": len(df),
        "n_by_split": df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0).to_dict(),
        "pos_share_by_split": df.groupby("split")["label"].mean().round(6).to_dict(),
        "hash_overlap_trainval_test": hash_overlap,
        **movie_overlaps,
        "label_rating_conflicts": label_conflict,
        "empty_text_clean": int((df["text_clean"].str.len() == 0).sum()),
    }


# ---------------------------------------------------------------- chạy
def _display_path(path: Path) -> str:
    """Đường dẫn tương đối so với gốc repo nếu nằm trong repo, ngược lại giữ nguyên."""
    repo_root = Path(__file__).resolve().parents[2]
    resolved = Path(path).resolve()
    return str(resolved.relative_to(repo_root)) if resolved.is_relative_to(repo_root) else str(path)


def committed_stats(stats: dict) -> dict:
    """Bản thống kê để commit vào `experiments/`: bỏ `generated_at` (chạy lại không đổi file)."""
    return {k: v for k, v in stats.items() if k != "generated_at"}


def prepare_dataset(
    root: Path | None = None,
    split: dict | None = None,
    out_path: Path | None = None,
    results_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Chạy toàn bộ B1: đọc dữ liệu gốc → áp split → làm sạch → loại trùng → ghi parquet.

    Trả về (bảng bàn giao, dict thống kê). Ghi `out_path` (mặc định
    `data/processed/reviews.parquet`), `prepare_stats.json` cạnh đó, và danh sách uid test
    trùng train vào `results_dir/test_uids_dup_train.txt` (mặc định `experiments/results/data_prep/`).
    """
    download_imdb(root)
    train_all, test = load_split("train", root), load_split("test", root)
    train, val = apply_split(train_all, split)

    parts, stats = remove_duplicates(train, val, test)
    processed = build_processed(parts)
    stats["contract"] = check_contract(processed)
    # Hash theo NỘI DUNG: giống nhau giữa các máy/phiên bản pandas (sha256 file parquet thì không).
    stats["content_hash"] = int(pd.util.hash_pandas_object(processed, index=False).sum())

    out_path = Path(out_path) if out_path is not None else data_dir("processed") / "reviews.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(out_path, index=False)
    stats["output_file"] = _display_path(out_path)

    # uid phía test của các review trùng train/val — phase EVAL dùng để báo "test acc khi bỏ
    # các review trùng train" (DATA.md §4) mà không cần đụng vào parquet 7 cột.
    uids = stats.pop("_test_uids_dup_train", [])
    results_dir = Path(results_dir) if results_dir is not None else experiment_dir("results", "data_prep")
    results_dir.mkdir(parents=True, exist_ok=True)
    uids_path = results_dir / "test_uids_dup_train.txt"
    uids_path.write_text("".join(f"{uid}\n" for uid in uids), encoding="utf-8")
    stats["test_uids_dup_train"] = {"n": len(uids), "file": _display_path(uids_path)}

    import json

    stats_path = out_path.parent / "prepare_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return processed, stats


def _print_report(stats: dict) -> None:
    print("Loại review trùng (B1) — docs/DATA.md §4")
    print(f"{'':28s}{'train':>8s}{'val':>8s}{'test':>8s}")
    print(f"{'đầu vào':28s}{stats['inputs']['train']:>8d}{stats['inputs']['val']:>8d}{stats['inputs']['test']:>8d}")
    print(f"{'bỏ: trùng trong split':28s}{stats['removed_within_split']['train']:>8d}"
          f"{stats['removed_within_split']['val']:>8d}{stats['duplicates_in_test_kept']:>8d}")
    print(f"{'bỏ: trùng với test':28s}{stats['removed_cross_split']['train']:>8d}"
          f"{stats['removed_cross_split']['val']:>8d}{0:>8d}")
    print(f"{'bỏ: phim có trong test':28s}{stats['removed_movie_leak']['train']:>8d}"
          f"{stats['removed_movie_leak']['val']:>8d}{0:>8d}")
    print(f"{'đầu ra':28s}{stats['outputs']['train']:>8d}{stats['outputs']['val']:>8d}{stats['outputs']['test']:>8d}")
    print("(cột test ở dòng 2 là số bản trùng trong test được GIỮ LẠI, không xoá)")
    print()
    dup = stats.get("test_uids_dup_train")
    if dup:
        print(f"  test review trùng train (giữ trong test): {stats['test_reviews_also_in_train']} → {dup['file']}")
    print(f"  content_hash: {stats['content_hash']}  (hash nội dung, so được giữa các máy)")
    for key, value in stats["contract"].items():
        print(f"  {key}: {value}")


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    processed, stats = prepare_dataset()
    _print_report(stats)
    print(f"\nĐã ghi {len(processed)} review, {len(processed.columns)} cột: "
          f"{', '.join(processed.columns)}")
    # Bản sao thống kê trong experiments/results/ (được commit) để trích dẫn số trong tài liệu.
    import json

    results = experiment_dir("results", "data_prep") / "prepare_stats.json"
    results.write_text(json.dumps(committed_stats(stats), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Thống kê (commit): {results}")
    print(f"uid test trùng train: {results.parent / 'test_uids_dup_train.txt'}")


if __name__ == "__main__":
    main()
