"""Test cho Task D4: Vòng huấn luyện (run_training), cân bằng mẫu con và Monitor."""
import tempfile
from pathlib import Path
import pandas as pd
import yaml

from src.train import _stratified_subset, run_training


def test_stratified_subset_balances_sorted_dataframe():
    # Mô phỏng reviews.parquet xếp toàn bộ nhãn 0 trước, nhãn 1 sau
    df = pd.DataFrame({
        "uid": [f"u_{i}" for i in range(100)],
        "label": [0] * 50 + [1] * 50,
        "text_clean": ["bad movie"] * 50 + ["great movie"] * 50,
    })
    sub = _stratified_subset(df, n=20, seed=42)
    assert len(sub) == 20
    counts = sub["label"].value_counts().to_dict()
    assert counts == {0: 10, 1: 10}


def test_run_training_one_epoch_isolated():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        rows = []
        for i in range(40):
            lbl = i % 2
            txt = "great wonderful film!" if lbl == 1 else "awful boring movie!"
            rows.append({
                "uid": f"tr_{i}",
                "split": "train" if i < 28 else "val",
                "text": txt,
                "text_clean": txt,
                "label": lbl,
                "rating": 9 if lbl else 2,
                "movie_id": f"tt{i}",
            })
        pq = tmp / "reviews.parquet"
        pd.DataFrame(rows).to_parquet(pq, index=False)

        cfg = {
            "model": {"rnn_type": "lstm", "bidirectional": False, "embed_dim": 16, "hidden_dim": 16, "dropout": 0.2},
            "data": {"max_len": 12, "batch_size": 8, "min_freq": 1, "max_vocab_size": 100, "truncation": "post"},
            "training": {"lr": 0.01, "weight_decay": 0.0, "max_epochs": 2, "patience": 2, "clip_grad_norm": 5.0},
        }
        cfg_file = tmp / "lstm_mini.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        exp_root = tmp / "experiments"
        res = run_training(
            config_path=cfg_file,
            seed=42,
            sample_size=20,
            epochs_override=2,
            device_name="cpu",
            data_path=pq,
            experiments_root=exp_root,
        )

        assert Path(res["checkpoint_path"]).exists()
        assert Path(res["vocab_path"]).exists()

        # Kiểm tra history.csv có mặt ở cả runs/ và results/ với đủ 9 cột bắt buộc
        run_hist = exp_root / "runs" / "lstm_mini" / "seed_42" / "history.csv"
        res_hist = exp_root / "results" / "lstm_mini" / "seed_42" / "history.csv"
        assert run_hist.exists() and res_hist.exists()

        hist_df = pd.read_csv(res_hist)
        expected_cols = {
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "val_f1", "grad_norm_mean", "lr", "epoch_seconds",
        }
        assert expected_cols == set(hist_df.columns)
        assert (exp_root / "results" / "lstm_mini" / "summary.json").exists()
        assert (exp_root / "plots" / "training" / "lstm_mini_curves.png").exists()
