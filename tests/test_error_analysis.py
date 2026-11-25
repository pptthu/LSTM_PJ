"""Test phân tích lỗi (T2): gán nhóm, bảng theo nhóm, chọn mô hình, ví dụ lỗi."""
import json

import numpy as np
import pandas as pd

from src.eval.error_analysis import (
    add_case_columns, by_group, contrast_at_end, error_examples, manual_sample, pick_best_exp,
)


def make_preds(n=200, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    pred = np.where(rng.random(n) < 0.8, label, 1 - label)
    texts = []
    for i in range(n):
        base = "good film " * int(rng.integers(5, 400))
        texts.append(base + ("but the ending was awful" if i % 3 == 0 else "really") + (" not never" * (i % 4)))
    return pd.DataFrame({
        "uid": [f"test_{i}" for i in range(n)], "text": texts, "label": label,
        "rating": rng.choice([1, 2, 3, 4, 7, 8, 9, 10], n), "prob": rng.random(n), "pred": pred,
        "n_tokens": [len(t.split()) for t in texts], "n_unk": rng.integers(0, 20, n),
        "was_truncated": [len(t.split()) > 250 for t in texts],
    })


def test_contrast_at_end_only_counts_turn_near_end():
    assert contrast_at_end("Great cast, lovely music, sharp dialogue and beautiful photography all along, but awful end.")
    assert not contrast_at_end("But the start is slow; the rest is wonderful, moving, funny and sharp throughout.")


def test_case_columns_and_group_table_cover_every_review():
    df = add_case_columns(make_preds())
    assert set(df["length"]) <= {"<100", "100–300", "300–600", ">600"}
    assert df.loc[df["n_tokens"] > 600, "length"].eq(">600").all()
    bl = df[["uid", "label"]].assign(pred=df["label"])  # baseline đúng hết
    t = by_group(df, bl)
    for _, g in t.groupby("group"):
        assert g["n"].sum() == len(df)
    assert (t["acc_baseline"] == 1.0).all()
    assert np.isclose(t.loc[t["group"] == "length", "n_errors_model"].sum(), (~df["correct"]).sum())


def test_examples_unique_and_capped_and_manual_sample_size():
    df = add_case_columns(make_preds(600, seed=1))
    ex = error_examples(df, k=10)
    assert ex["uid"].is_unique
    assert ex.groupby("case").size().max() <= 10
    assert (ex["label"] != ex["pred"]).all()
    m = manual_sample(df, n=50)
    assert len(m) == 50 and m["uid"].is_unique and (m["label"] != m["pred"]).all()


def test_pick_best_exp_skips_single_seed_and_ablation(tmp_path):
    def put(name, mean, n_seeds, seeds):
        d = tmp_path / name
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"n_seeds": n_seeds, "best_val_acc": {"mean": mean}}))
        for s, acc in seeds.items():
            (d / f"seed_{s}").mkdir()
            (d / f"seed_{s}" / "metrics.json").write_text(json.dumps({"best_val_acc": acc}))

    put("lstm", 0.976, 1, {42: 0.976})                 # lần chạy thử 1 seed → bỏ
    put("maxlen_600_post", 0.89, 3, {42: 0.89})        # ablation → bỏ
    put("gru", 0.859, 3, {42: 0.86, 123: 0.862, 2025: 0.855})
    put("bilstm_attention", 0.861, 3, {42: 0.866, 123: 0.862, 2025: 0.856})
    assert pick_best_exp(tmp_path) == ("bilstm_attention", 42)
