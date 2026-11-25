"""Test cho Task D8: Metrics, McNemar test chính xác, compare_all_models và evaluate_split cô lập."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.vocab import Vocab
from src.eval.metrics import compute_classification_metrics, mcnemar_test
from src.evaluate import compare_all_models, evaluate_split, run_mcnemar_cli
from src.models.rnn import SentimentRNN


def test_compute_classification_metrics():
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
    y_prob = [0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.85, 0.3]

    res = compute_classification_metrics(y_true, y_pred, y_prob)
    assert res["accuracy"] == 6 / 8
    assert "macro_f1" in res
    assert res["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3, "total": 8}
    assert res["roc_auc"] is not None


def test_mcnemar_test_exact_and_chi2():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred_a = np.array([1, 1, 1, 1, 0, 0, 0, 1])
    y_pred_b = np.array([1, 0, 0, 1, 1, 1, 0, 0])

    res = mcnemar_test(y_true, y_pred_a, y_pred_b, "Model A", "Model B")
    assert res["model_a"] == "Model A"
    assert res["model_b"] == "Model B"
    assert "statistic" in res
    assert "exact_p_value" in res and "chi2_p_value" in res
    assert res["contingency_table"]["total"] == 8


def test_evaluate_split_pipeline_isolated_in_tmp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        vocab = Vocab({"<pad>": 0, "<unk>": 1, "good": 2, "bad": 3})
        vocab_file = tmp_path / "vocab.json"
        vocab.save(vocab_file)

        model = SentimentRNN(vocab_size=len(vocab), embed_dim=8, hidden_dim=16, rnn_type="lstm")
        ckpt_file = tmp_path / "best.pt"
        model.save_checkpoint(ckpt_file, config={"data": {"max_len": 10}})

        df = pd.DataFrame({
            "uid": ["test_0", "test_1", "test_2", "test_3"],
            "split": ["test", "test", "test", "test"],
            "text": ["good film", "bad film", "very good", "so bad"],
            "text_clean": ["good film", "bad film", "very good", "so bad"],
            "label": [1, 0, 1, 0],
            "rating": [9, 2, 8, 1],
            "movie_id": ["m1", "m2", "m3", "m4"],
        })
        parquet_file = tmp_path / "reviews.parquet"
        df.to_parquet(parquet_file)

        out_dir = tmp_path / "results"
        results = evaluate_split(
            checkpoint_path=ckpt_file,
            vocab_path=vocab_file,
            split="test",
            data_path=parquet_file,
            output_dir=out_dir,
            device_name="cpu",
            batch_size=2,
        )

        assert results["split"] == "test"
        assert (out_dir / "predictions_test.csv").exists()
        assert (out_dir / "test_metrics.json").exists()
        # Đảm bảo hình được ghi trong out_dir/plots, không ghi vào experiments/plots/eval của repo
        assert any((out_dir / "plots").glob("*.png"))

        pred_df = pd.read_csv(out_dir / "predictions_test.csv")
        assert len(pred_df) == 4
        expected_cols = {"uid", "text", "label", "rating", "prob", "pred", "n_tokens", "n_unk", "was_truncated"}
        assert expected_cols.issubset(set(pred_df.columns))


def test_compare_all_models_reads_b5_nested_baseline_and_mcnemar_sorts_uids():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res_base = Path(tmp_dir)
        bl_dir = res_base / "baseline"
        bl_dir.mkdir(parents=True)
        (bl_dir / "config.json").write_text(
            json.dumps({"selected_C": 10.0, "tfidf": {"n_features": 100}, "timing": {"train_s": 1.2, "inference_ms_per_review": 0.01}}),
            encoding="utf-8",
        )
        (bl_dir / "metrics.json").write_text(
            json.dumps({
                "M0a_majority": {"val": {"accuracy": 0.5, "macro_f1": 0.3333, "roc_auc": 0.5}},
                "M0b_tfidf_lr": {"selected_C": 10.0, "val": {"accuracy": 0.8985, "macro_f1": 0.8985, "roc_auc": 0.9623}},
            }),
            encoding="utf-8",
        )

        lstm_s42 = res_base / "lstm" / "seed_42"
        lstm_s42.mkdir(parents=True)
        (lstm_s42 / "metrics.json").write_text(
            json.dumps({"best_val_acc": 0.86, "val_f1_at_best": 0.86, "total_train_seconds": 12.0, "parameters": {"total": 1000, "trainable": 1000}}),
            encoding="utf-8",
        )

        comp = compare_all_models(results_base=res_base, split="val", include_lab3_reference=False)
        models = comp["model"].tolist()
        assert "M0a_majority" in models
        assert "M0b_tfidf_lr (C=10)" in models
        assert "lstm" in models

        # Kiểm tra run_mcnemar_cli tự khớp uid dù thứ tự dòng ngược nhau
        f_a = res_base / "a.csv"
        f_b = res_base / "b.csv"
        pd.DataFrame({"uid": ["u1", "u2", "u3"], "label": [0, 1, 1], "pred": [0, 1, 1]}).to_csv(f_a, index=False)
        pd.DataFrame({"uid": ["u3", "u1", "u2"], "label": [1, 0, 1], "pred": [0, 0, 1]}).to_csv(f_b, index=False)
        mc_out = res_base / "mcnemar.json"
        mc = run_mcnemar_cli(f_a, f_b, "RNN", "BL", output_path=mc_out)
        assert mc["contingency_table"]["a_correct_b_wrong"] == 1
        assert mc_out.exists()
