"""Test baseline B5: majority, TF-IDF + LR, chọn C trên val, không fit trên val, tất định."""
import numpy as np
import pandas as pd
import pytest

from src.models.baseline import compute_metrics, evaluate_baseline_on_test, fit_tfidf_lr, majority_baseline


def make_corpus(n=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = int(rng.integers(0, 2))
        words = ["good", "great", "love", "best"] if label == 1 else ["bad", "awful", "hate", "worst"]
        other = ["film", "movie", "plot", "acting", "story"]
        text = " ".join(list(rng.choice(words, 3)) + list(rng.choice(other, 3)))
        rows.append({"uid": f"u{i}", "text": text, "text_clean": text, "label": label,
                     "rating": 9 if label else 2, "movie_id": f"tt{i}"})
    return pd.DataFrame(rows)


def test_compute_metrics_hand_computed():
    m = compute_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.6, 0.4, 0.9])
    assert m["accuracy"] == 0.5
    assert m["confusion_matrix"]["matrix"] == [[1, 1], [1, 1]]
    assert m["precision"] == {"negative": 0.5, "positive": 0.5}
    assert m["macro_f1"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)


def test_majority_baseline_predicts_majority_class():
    train = pd.DataFrame({"label": [0, 0, 0, 1]})
    val = pd.DataFrame({"label": [0, 1, 1, 0]})
    out = majority_baseline(train, val)
    assert out["majority_class"] == 0
    assert out["val"]["accuracy"] == 0.5          # 2/4 đoán đúng
    assert out["val"]["roc_auc"] == 0.5           # prob hằng → xếp hạng ngẫu nhiên


def test_tfidf_is_fitted_on_train_only():
    train = make_corpus()
    val = make_corpus(seed=1).copy()
    val.loc[0, "text_clean"] = "valonlyword appears only here"
    vectorizer, model, _, _, _, _ = fit_tfidf_lr(train, val, c_grid=(1.0,), verbose=False)
    assert "valonlyword" not in vectorizer.vocabulary_
    assert model.classes_.tolist() == [0, 1]


def test_c_is_selected_on_validation_accuracy():
    train, val = make_corpus(200), make_corpus(60, seed=7)
    vectorizer, model, table, best_c, timing, best_metrics = fit_tfidf_lr(
        train, val, c_grid=(0.001, 0.1, 10.0), verbose=False
    )
    assert best_c in {0.001, 0.1, 10.0}
    assert table["C"].tolist() == [0.001, 0.1, 10.0]
    assert best_metrics["accuracy"] == table["val_accuracy"].max()
    # model lưu lại dùng đúng C đã chọn
    assert model.C == best_c
    assert timing["inference_ms"] >= 0 and len(vectorizer.vocabulary_) > 0


def test_baseline_is_deterministic():
    train, val = make_corpus(150), make_corpus(50, seed=3)
    first = fit_tfidf_lr(train, val, c_grid=(0.1, 1.0), verbose=False)
    second = fit_tfidf_lr(train, val, c_grid=(0.1, 1.0), verbose=False)
    assert first[3] == second[3]
    metric_cols = ["C", "val_accuracy", "val_macro_f1", "val_roc_auc"]
    pd.testing.assert_frame_equal(first[2][metric_cols], second[2][metric_cols])
    prob1 = first[1].predict_proba(first[0].transform(val["text_clean"]))[:, 1]
    prob2 = second[1].predict_proba(second[0].transform(val["text_clean"]))[:, 1]
    np.testing.assert_allclose(prob1, prob2)


def test_missing_text_column_raises():
    train = make_corpus().drop(columns=["text_clean"])
    with pytest.raises(ValueError, match="missing"):
        fit_tfidf_lr(train, make_corpus(), verbose=False)


def test_evaluate_baseline_on_test_isolated(tmp_path):
    tr = make_corpus(60, seed=1).assign(split="train")
    va = make_corpus(20, seed=2).assign(split="val")
    te = make_corpus(20, seed=3).assign(split="test")
    pq = tmp_path / "reviews.parquet"
    pd.concat([tr, va, te], ignore_index=True).to_parquet(pq, index=False)

    res_dir = tmp_path / "results"
    ckpt = tmp_path / "baseline_tfidf_lr.joblib"
    out = evaluate_baseline_on_test(parquet_path=str(pq), checkpoint_path=str(ckpt), results_dir=str(res_dir))
    assert "M0a_majority" in out and "M0b_tfidf_lr" in out
    pred_csv = res_dir / "predictions_test.csv"
    assert pred_csv.exists() and (res_dir / "test_metrics.json").exists()
    pred_df = pd.read_csv(pred_csv)
    assert len(pred_df) == 20
    assert {"uid", "text", "label", "rating", "prob", "pred", "n_tokens", "n_unk", "was_truncated"} <= set(pred_df.columns)
