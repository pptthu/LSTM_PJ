"""Baseline B5: M0a Majority class và M0b TF-IDF (1–2 gram) + Logistic Regression.

Quy tắc (docs/phases/BASELINE.md):
- TF-IDF fit **chỉ** trên train; val chỉ `transform`.
- Chọn `C` theo val accuracy (hoà → macro-F1 cao hơn → `C` nhỏ hơn, để tất định).
- Mô hình lưu lại vẫn train trên train với `C` đã chọn, **không** fit lại trên val.
- Không dùng test: số test chỉ chạy một lần ở phase EVAL.

Chạy từ gốc repo:
    python -m src.models.baseline     # cần data/processed/reviews.parquet (B1)

Ghi ra:
    experiments/checkpoints/baseline_tfidf_lr.joblib   (không commit)
    experiments/results/baseline/{metrics.json,config.json,c_grid.csv,predictions_val.csv}
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.utils.paths import data_dir, experiment_dir

RANDOM_STATE = 42
C_GRID = (0.01, 0.1, 1.0, 10.0, 30.0, 100.0)
TEXT_COLUMN = "text_clean"  # cùng đầu vào với tokenizer/LSTM (đã bỏ <br />, giữ hoa/thường)


# ---------------------------------------------------------------- metric
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Accuracy, precision/recall/F1 từng lớp, macro/weighted-F1, ROC-AUC, confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:  # chỉ có một lớp trong y_true, hoặc prob hằng số (Majority)
        auc = None
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": {"negative": float(precision[0]), "positive": float(precision[1])},
        "recall": {"negative": float(recall[0]), "positive": float(recall[1])},
        "f1": {"negative": float(f1[0]), "positive": float(f1[1])},
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "support": {"negative": int(support[0]), "positive": int(support[1])},
        "roc_auc": auc,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),  # [[TN, FP], [FN, TP]]
        },
    }


# ---------------------------------------------------------------- M0a
def majority_baseline(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """M0a: luôn dự đoán lớp nhiều nhất trong train; đo trên val."""
    majority = int(train["label"].value_counts().idxmax())
    pred = np.full(len(val), majority, dtype=int)
    prob = np.full(len(val), float(majority), dtype=float)  # prob = nhãn hằng → ROC-AUC không xác định
    return {
        "model": "majority",
        "majority_class": majority,
        "train_label_share": {str(k): float(v) for k, v in train["label"].value_counts(normalize=True).items()},
        "val": compute_metrics(val["label"], pred, prob),
    }


# ---------------------------------------------------------------- M0b
def fit_tfidf_lr(
    train: pd.DataFrame,
    val: pd.DataFrame,
    c_grid=C_GRID,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
) -> tuple[TfidfVectorizer, LogisticRegression, pd.DataFrame, float, dict, dict]:
    """Fit TF-IDF trên train, quét `C` trên val.

    Trả về (vectorizer, model tốt nhất, bảng C, C chọn, timing, metric val của C chọn).

    Vectorizer và mô hình cuối cùng chỉ nhìn thấy train; val chỉ dùng `transform`/`predict`.
    """
    for name, df in (("train", train), ("val", val)):
        if TEXT_COLUMN not in df.columns:
            raise ValueError(f"{name}: missing {TEXT_COLUMN!r}")
        if not set(df["label"]) <= {0, 1}:
            raise ValueError(f"{name}: label must be 0/1")
    if train.empty or val.empty:
        raise ValueError("train and val must be non-empty")

    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=False)
    x_train = vectorizer.fit_transform(train[TEXT_COLUMN])
    x_val = vectorizer.transform(val[TEXT_COLUMN])
    vectorize_time = time.perf_counter() - t0

    rows, fitted = [], {}
    for c_value in c_grid:
        t_fit = time.perf_counter()
        model = LogisticRegression(C=float(c_value), solver="liblinear", max_iter=1000, random_state=seed)
        model.fit(x_train, train["label"])
        train_time = time.perf_counter() - t_fit

        t_pred = time.perf_counter()
        prob = model.predict_proba(x_val)[:, 1]
        pred = (prob >= 0.5).astype(int)
        infer_time = time.perf_counter() - t_pred

        metrics = compute_metrics(val["label"], pred, prob)
        # accuracy trên train để thấy dấu hiệu overfitting (khoảng cách train–val tăng dần theo C)
        train_pred = (model.predict_proba(x_train)[:, 1] >= 0.5).astype(int)
        train_accuracy = float(accuracy_score(train["label"], train_pred))
        rows.append({
            "C": float(c_value),
            "train_accuracy": round(train_accuracy, 4),
            "val_accuracy": metrics["accuracy"],
            "train_minus_val_accuracy": round(train_accuracy - metrics["accuracy"], 4),
            "val_macro_f1": metrics["macro_f1"],
            "val_roc_auc": metrics["roc_auc"],
            "train_time_s": round(train_time, 3),
            "inference_time_ms": round(infer_time * 1000, 3),
        })
        fitted[float(c_value)] = (model, metrics)
        if verbose:
            print(f"  C={c_value:<6g} val acc={metrics['accuracy']:.4f}  macro-F1={metrics['macro_f1']:.4f}"
                  f"  AUC={metrics['roc_auc']:.4f}  ({train_time:.1f}s)")

    table = pd.DataFrame(rows)
    # Chọn theo accuracy; hoà → macro-F1; hoà nữa → C nhỏ hơn (tất định, ít overfit hơn).
    best_row = table.sort_values(
        ["val_accuracy", "val_macro_f1", "C"], ascending=[False, False, True], kind="stable"
    ).iloc[0]
    best_c = float(best_row["C"])
    best_model, best_metrics = fitted[best_c]
    timing = {
        "vectorize_s": round(vectorize_time, 3),
        "train_s": float(best_row["train_time_s"]),
        "inference_ms": float(best_row["inference_time_ms"]),
        "inference_ms_per_review": round(float(best_row["inference_time_ms"]) / len(val), 4),
    }
    return vectorizer, best_model, table, best_c, timing, best_metrics


# ---------------------------------------------------------------- chạy
def build_config(vectorizer: TfidfVectorizer, model: LogisticRegression, seed: int) -> dict:
    """Cấu hình đủ để dựng lại mô hình (không phụ thuộc object sklearn)."""
    import sklearn

    return {
        "text_column": TEXT_COLUMN,
        "tfidf": {
            "ngram_range": list(vectorizer.ngram_range),
            "min_df": vectorizer.min_df,
            "sublinear_tf": vectorizer.sublinear_tf,
            "lowercase": vectorizer.lowercase,
            "n_features": int(len(vectorizer.vocabulary_)),
        },
        "logistic_regression": {
            "C": float(model.C),
            "solver": model.solver,
            "max_iter": model.max_iter,
            "random_state": seed,
            "class_weight": model.class_weight,
        },
        "c_grid": list(C_GRID),
        "sklearn_version": sklearn.__version__,
    }


def _predict_frame(df: pd.DataFrame, prob, pred) -> pd.DataFrame:
    out = pd.DataFrame({
        "uid": df["uid"].values,
        "text": df["text"].values,
        "label": df["label"].astype(int).values,
        "rating": df["rating"].astype(int).values,
        "prob": np.round(np.asarray(prob, dtype=float), 6),
        "pred": np.asarray(pred, dtype=int),
    })
    return out


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parquet = data_dir("processed") / "reviews.parquet"
    if not parquet.exists():
        raise SystemExit(f"Chưa có {parquet} — chạy `python -m src.data.prepare` trước (B1).")
    df = pd.read_parquet(parquet)
    # Chỉ đọc train/val. Không đọc test: baseline không được đánh giá trên test (BASELINE.md).
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    print(f"train: {len(train)} review · val: {len(val)} review · test: KHÔNG đọc")

    print("\nM0a Majority:")
    majority = majority_baseline(train, val)
    print(f"  lớp nhiều nhất = {majority['majority_class']} · val acc = {majority['val']['accuracy']:.4f}"
          f" · macro-F1 = {majority['val']['macro_f1']:.4f}")

    print("\nM0b TF-IDF (1–2 gram) + LogisticRegression, quét C trên val:")
    vectorizer, model, table, best_c, timing, best = fit_tfidf_lr(train, val)
    print(f"  chọn C = {best_c:g} · val acc = {best['accuracy']:.4f} · macro-F1 = {best['macro_f1']:.4f}"
          f" · ROC-AUC = {best['roc_auc']:.4f}")
    print(f"  {len(vectorizer.vocabulary_):,} đặc trưng · vectorize {timing['vectorize_s']} s"
          f" · train {timing['train_s']} s · inference {timing['inference_ms_per_review']} ms/review")

    results_dir = experiment_dir("results", "baseline")
    checkpoints = experiment_dir("checkpoints")
    joblib.dump({"vectorizer": vectorizer, "model": model, "config": build_config(vectorizer, model, RANDOM_STATE)},
                checkpoints / "baseline_tfidf_lr.joblib")

    prob = model.predict_proba(vectorizer.transform(val[TEXT_COLUMN]))[:, 1]
    pred = (prob >= 0.5).astype(int)
    predictions = _predict_frame(val, prob, pred)
    predictions.to_csv(results_dir / "predictions_val.csv", index=False)
    table.to_csv(results_dir / "c_grid.csv", index=False)

    config = build_config(vectorizer, model, RANDOM_STATE)
    config["selected_C"] = best_c
    config["timing"] = timing
    config["n_train"], config["n_val"] = len(train), len(val)
    (results_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    metrics = {"M0a_majority": majority, "M0b_tfidf_lr": {"selected_C": best_c, "val": best, "c_grid": table.to_dict("records")}}
    (results_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nĐã ghi: {results_dir}/{{metrics.json,config.json,c_grid.csv,predictions_val.csv}}")
    print(f"Checkpoint: {checkpoints / 'baseline_tfidf_lr.joblib'} (không commit)")


if __name__ == "__main__":
    main()
