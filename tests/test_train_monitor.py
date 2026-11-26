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
        run_hist = exp_root / "runs" / "lstm_mini_sample20" / "seed_42" / "history.csv"
        res_hist = exp_root / "results" / "lstm_mini_sample20" / "seed_42" / "history.csv"
        assert run_hist.exists() and res_hist.exists()

        hist_df = pd.read_csv(res_hist)
        expected_cols = {
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "val_f1", "grad_norm_mean", "lr", "epoch_seconds",
        }
        assert expected_cols == set(hist_df.columns)
        assert (exp_root / "results" / "lstm_mini_sample20" / "summary.json").exists()
        assert (exp_root / "plots" / "training" / "lstm_mini_sample20_curves.png").exists()

        # Đảm bảo chạy sample_size KHÔNG ghi vào checkpoints/<exp>/vocab.json
        ckpt_vocab = exp_root / "checkpoints" / "lstm_mini" / "vocab.json"
        assert not ckpt_vocab.exists()
        # ... và không ghi bất kỳ file nào vào thư mục của lần chạy thật
        for sub in ("checkpoints", "runs", "results"):
            assert not (exp_root / sub / "lstm_mini").exists()


def test_sample_size_does_not_pollute_full_vocab_and_summary_uses_sample_std():
    import json
    from src.data.vocab import Vocab
    from src.training.monitor import aggregate_experiment_seeds

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        rows = []
        for i in range(24):
            lbl = i % 2
            # Thêm từ đặc trưng riêng cho từng mẫu để vocab đầy đủ lớn hơn vocab của tập con
            txt = f"great film token_{i}" if lbl == 1 else f"bad movie token_{i}"
            rows.append({
                "uid": f"tr_{i}",
                "split": "train" if i < 16 else "val",
                "text": txt,
                "text_clean": txt,
                "label": lbl,
                "rating": 9 if lbl else 2,
                "movie_id": f"tt{i}",
            })
        pq = tmp / "reviews.parquet"
        pd.DataFrame(rows).to_parquet(pq, index=False)

        cfg = {
            "model": {"rnn_type": "lstm", "bidirectional": False, "embed_dim": 8, "hidden_dim": 8, "dropout": 0.0},
            "data": {"max_len": 8, "batch_size": 4, "min_freq": 1, "max_vocab_size": 200, "truncation": "post"},
            "training": {"lr": 0.01, "weight_decay": 0.0, "max_epochs": 1, "patience": 2, "clip_grad_norm": 5.0},
        }
        cfg_file = tmp / "exp_vocab.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        exp_root = tmp / "experiments"

        # 1. Chạy thử với sample_size=4 trước
        res_sample = run_training(
            config_path=cfg_file,
            seed=42,
            sample_size=4,
            epochs_override=1,
            device_name="cpu",
            data_path=pq,
            experiments_root=exp_root,
        )
        sample_vocab = Vocab.load(res_sample["vocab_path"])
        ckpt_vocab_path = exp_root / "checkpoints" / "exp_vocab" / "vocab.json"
        assert not ckpt_vocab_path.exists()

        # 2. Chạy đầy đủ (sample_size=None) sau đó -> phải xây vocab từ toàn bộ 16 mẫu train
        res_full = run_training(
            config_path=cfg_file,
            seed=42,
            sample_size=None,
            epochs_override=1,
            device_name="cpu",
            data_path=pq,
            experiments_root=exp_root,
        )
        assert ckpt_vocab_path.exists()
        full_vocab = Vocab.load(res_full["vocab_path"])
        assert len(full_vocab) > len(sample_vocab)

        # 3. Kiểm tra độ lệch chuẩn mẫu (ddof=1) cho 3 seed: std([0.80, 0.85, 0.90], ddof=1) = 0.05
        res_base = tmp / "std_check"
        for s, acc in [(42, 0.80), (123, 0.85), (2025, 0.90)]:
            s_dir = res_base / "lstm" / f"seed_{s}"
            s_dir.mkdir(parents=True)
            (s_dir / "metrics.json").write_text(
                json.dumps({"seed": s, "best_val_acc": acc, "parameters": {"total": 100}}),
                encoding="utf-8",
            )
        summary = aggregate_experiment_seeds("lstm", seeds=(42, 123, 2025), results_base=res_base)
        assert summary["best_val_acc"]["std"] == 0.05
        assert summary["best_val_acc"]["formatted"] == "85.00 ± 5.00%"
