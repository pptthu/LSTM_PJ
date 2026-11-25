"""Đánh giá mô hình trên tập kiểm thử (Validation / Test Evaluation) và xuất kết quả.

Theo đặc tả trong docs/phases/EVAL.md và docs/TASKS.md §3 D8:
- Nạp checkpoint cuối (best.pt) và vocab.json.
- Chạy đánh giá trên tập val hoặc test.
- Xuất file dự đoán predictions_<split>.csv với đầy đủ 9 cột bàn giao:
  uid, text, label, rating, prob, pred, n_tokens, n_unk, was_truncated
- Xuất <split>_metrics.json (kèm số đo khi bỏ 123 review test trùng train ở DATA.md §4),
  đo latency, vẽ confusion matrix và ROC curve.
- Xuất bảng so sánh tổng hợp comparison_*.csv (tự động đọc cả M0a/M0b từ baseline) và mcnemar.json.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
import torch

from src.data.dataset import IMDbDataset, create_dataloader, load_imdb_split_data
from src.data.vocab import Vocab
from src.eval.metrics import (
    compute_classification_metrics,
    mcnemar_test,
    measure_inference_speed,
)
from src.models.rnn import SentimentRNN
from src.utils.paths import REPO_ROOT, experiment_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def _default_results_dir_for_ckpt(ckpt_file: Path) -> Path:
    """Chuyển đường dẫn .../checkpoints/<exp>/seed_<s>/best.pt sang .../results/<exp>/seed_<s>."""
    parts = ckpt_file.resolve().parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        rel_after = parts[idx + 1 : -1]
        if rel_after:
            return experiment_dir("results", *rel_after)
    return ckpt_file.parent


def evaluate_split(
    checkpoint_path: str | Path,
    vocab_path: str | Path,
    split: str = "test",
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    device_name: str | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Đánh giá checkpoint trên split chỉ định và xuất đầy đủ artifacts."""
    ckpt_file = Path(checkpoint_path)
    vocab_file = Path(vocab_path)

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại {ckpt_file}")
    if not vocab_file.exists():
        raise FileNotFoundError(f"Không tìm thấy vocab tại {vocab_file}")

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab.load(vocab_file)
    model, ckpt = SentimentRNN.load_checkpoint(ckpt_file, device=device)
    model.eval()

    df = load_imdb_split_data(parquet_path=data_path, split=split)
    logger.info(f"Đánh giá {len(df)} mẫu từ split '{split}'...")

    data_cfg = ckpt.get("config", {}).get("data", {})
    max_len = data_cfg.get("max_len", 250)
    truncation = data_cfg.get("truncation", "post")
    dataset = IMDbDataset(
        texts=df["text_clean"],
        labels=df["label"],
        uids=df["uid"],
        ratings=df["rating"],
        vocab=vocab,
        max_len=max_len,
        truncation=truncation,
    )
    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

    all_uids: list[str] = []
    all_ratings: list[int] = []
    all_targets: list[int] = []
    all_probs: list[float] = []
    all_preds: list[int] = []
    all_n_tokens: list[int] = []
    all_n_unk: list[int] = []
    all_was_truncated: list[bool] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"]

            probs = model.predict_proba(input_ids, lengths).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_uids.extend(batch["uids"])
            all_ratings.extend(batch["ratings"])
            all_targets.extend(batch["labels"].numpy().astype(int).tolist())
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_n_tokens.extend(batch["n_tokens"])
            all_n_unk.extend(batch["n_unk"])
            all_was_truncated.extend(batch["was_truncated"])

    # Giữ cột text gốc (nếu có trong parquet) hoặc text_clean để nhất quán với baseline
    raw_text_map = dict(zip(df["uid"], df["text"] if "text" in df.columns else df["text_clean"]))
    all_texts = [raw_text_map[u] for u in all_uids]

    metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
    speed_metrics = measure_inference_speed(model, dataloader, device)
    param_counts = model.count_parameters()

    results_data: dict[str, Any] = {
        "split": split,
        "checkpoint": ckpt_file.as_posix(),
        "parameters": param_counts,
        "metrics": metrics,
        "speed": speed_metrics,
    }

    # Báo thêm test metrics khi bỏ 123 review test trùng train (DATA.md §4)
    dup_uids_file = REPO_ROOT / "experiments" / "results" / "data_prep" / "test_uids_dup_train.txt"
    if split == "test" and dup_uids_file.exists():
        dup_uids = set(dup_uids_file.read_text(encoding="utf-8").split())
        keep_idx = [i for i, u in enumerate(all_uids) if u not in dup_uids]
        if 0 < len(keep_idx) < len(all_uids):
            results_data["metrics_without_train_dups"] = compute_classification_metrics(
                [all_targets[i] for i in keep_idx],
                [all_preds[i] for i in keep_idx],
                [all_probs[i] for i in keep_idx],
            )
            results_data["n_removed_train_dups"] = len(all_uids) - len(keep_idx)

    out_path = Path(output_dir) if output_dir is not None else _default_results_dir_for_ckpt(ckpt_file)
    out_path.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame(
        {
            "uid": all_uids,
            "text": all_texts,
            "label": all_targets,
            "rating": all_ratings,
            "prob": [round(p, 6) for p in all_probs],
            "pred": all_preds,
            "n_tokens": all_n_tokens,
            "n_unk": all_n_unk,
            "was_truncated": all_was_truncated,
        }
    )
    pred_csv_path = out_path / f"predictions_{split}.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    logger.info(f"Đã lưu bảng dự đoán chi tiết ({len(pred_df)} dòng) tại {pred_csv_path}")

    metrics_json_path = out_path / f"{split}_metrics.json"
    metrics_json_path.write_text(json.dumps(results_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Đã lưu metrics tại {metrics_json_path}")

    # Xác định nơi lưu hình: nếu chạy trong repo experiments/ thì lưu vào experiments/plots/eval/,
    # ngược lại (vd: unit test trong tmp_dir) lưu cạnh out_path/plots để không làm bẩn repo thật.
    if plots_dir is not None:
        target_plots_dir = Path(plots_dir)
    elif out_path.resolve().is_relative_to((REPO_ROOT / "experiments").resolve()):
        target_plots_dir = experiment_dir("plots", "eval")
    else:
        target_plots_dir = out_path / "plots"
    target_plots_dir.mkdir(parents=True, exist_ok=True)

    seed_stem = out_path.name if out_path.name.startswith("seed_") else (
        ckpt_file.parent.name if ckpt_file.parent.name.startswith("seed_") else None
    )
    exp_stem = out_path.parent.name if seed_stem and out_path.name.startswith("seed_") else (
        ckpt_file.parts[-3] if len(ckpt_file.parts) >= 3 else "model"
    )
    plot_prefix = f"{exp_stem}_{seed_stem}" if seed_stem else exp_stem
    title_suffix = f" ({seed_stem}, {split})" if seed_stem else f" ({split})"

    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array([
        [metrics["confusion_matrix"]["tn"], metrics["confusion_matrix"]["fp"]],
        [metrics["confusion_matrix"]["fn"], metrics["confusion_matrix"]["tp"]],
    ])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_arr, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix: {exp_stem}{title_suffix}", fontweight="bold")
    cm_plot_path = target_plots_dir / f"{plot_prefix}_{split}_cm.png"
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    auc_val = metrics.get("roc_auc") or 0.0
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_title(f"ROC Curve: {exp_stem}{title_suffix}", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    roc_plot_path = target_plots_dir / f"{plot_prefix}_{split}_roc.png"
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=150)
    plt.close(fig)

    logger.info(f"Đã xuất đồ thị đánh giá tại {target_plots_dir}")
    return results_data


def _extract_baseline_rows(exp_dir: Path, split: str) -> list[dict[str, Any]]:
    """Đọc cấu trúc metrics của B5 (src/models/baseline.py) cho cả M0a và M0b."""
    rows: list[dict[str, Any]] = []
    cfg_file = exp_dir / "config.json"
    cfg = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
    n_features = cfg.get("tfidf", {}).get("n_features")
    m0b_params = f"{int(n_features) + 1:,}" if n_features is not None else "—"
    m0b_train_s = cfg.get("timing", {}).get("train_s")
    m0b_val_lat = cfg.get("timing", {}).get("inference_ms_per_review")

    nested_file = exp_dir / ("metrics.json" if split == "val" else "test_metrics.json")
    if nested_file.exists():
        raw = json.loads(nested_file.read_text(encoding="utf-8"))
        if "M0a_majority" in raw or "M0b_tfidf_lr" in raw:
            m0a = raw.get("M0a_majority", {}).get(split)
            if m0a and m0a.get("accuracy") is not None:
                rows.append({
                    "model": "M0a_majority",
                    "accuracy": f"{m0a['accuracy'] * 100:.2f}%",
                    "macro_f1": f"{m0a['macro_f1']:.4f}" if m0a.get("macro_f1") is not None else "—",
                    "roc_auc": f"{m0a['roc_auc']:.4f}" if m0a.get("roc_auc") is not None else "—",
                    "total_params": "0",
                    "trainable_params": "0",
                    "train_time_s": "—",
                    "latency_ms_sample": "—",
                })
            m0b_root = raw.get("M0b_tfidf_lr", {})
            m0b = m0b_root.get(split)
            if m0b and m0b.get("accuracy") is not None:
                lat = m0b_root.get("inference_ms_per_review", m0b_val_lat if split == "val" else None)
                c_sel = m0b_root.get("selected_C", cfg.get("selected_C"))
                label = f"M0b_tfidf_lr (C={c_sel:g})" if c_sel is not None else "M0b_tfidf_lr"
                rows.append({
                    "model": label,
                    "accuracy": f"{m0b['accuracy'] * 100:.2f}%",
                    "macro_f1": f"{m0b['macro_f1']:.4f}" if m0b.get("macro_f1") is not None else "—",
                    "roc_auc": f"{m0b['roc_auc']:.4f}" if m0b.get("roc_auc") is not None else "—",
                    "total_params": m0b_params,
                    "trainable_params": m0b_params,
                    "train_time_s": f"{m0b_train_s:.1f}" if m0b_train_s is not None else "—",
                    "latency_ms_sample": lat if lat is not None else "—",
                })
            return rows

    direct_metric_file = exp_dir / f"{split}_metrics.json"
    if direct_metric_file.exists():
        raw = json.loads(direct_metric_file.read_text(encoding="utf-8"))
        m = raw.get("metrics", raw)
        acc = m.get("accuracy")
        if acc is not None:
            rows.append({
                "model": exp_dir.name,
                "accuracy": f"{acc * 100:.2f}%",
                "macro_f1": f"{m['macro_f1']:.4f}" if m.get("macro_f1") is not None else "—",
                "roc_auc": f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "—",
                "total_params": "0" if exp_dir.name == "majority" else m0b_params,
                "trainable_params": "0" if exp_dir.name == "majority" else m0b_params,
                "train_time_s": "—",
                "latency_ms_sample": "—",
            })
    return rows


def compare_all_models(
    results_base: Path | str | None = None,
    split: str = "test",
    include_lab3_reference: bool = True,
) -> pd.DataFrame:
    """Tạo bảng so sánh tổng hợp mọi mô hình (comparison_<split>.csv), gồm Baseline, RNN 3 seed và tham chiếu Lab 3."""
    base = Path(results_base) if results_base is not None else experiment_dir("results")
    rows: list[dict[str, Any]] = []
    skip_dirs = {"data_prep", "eda", "error_analysis"}

    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in skip_dirs:
            continue
        exp_name = exp_dir.name
        seed_dirs = sorted(exp_dir.glob("seed_*"))

        if not seed_dirs:
            rows.extend(_extract_baseline_rows(exp_dir, split))
            continue

        seed_accs: list[float] = []
        seed_f1s: list[float] = []
        seed_aucs: list[float] = []
        seed_latencies: list[float] = []
        seed_train_times: list[float] = []
        total_p = None
        trainable_p = None

        for s_dir in seed_dirs:
            m_file = s_dir / f"{split}_metrics.json"
            train_m_file = s_dir / "metrics.json"
            if train_m_file.exists():
                t_data = json.loads(train_m_file.read_text(encoding="utf-8"))
                if t_data.get("total_train_seconds") is not None:
                    seed_train_times.append(float(t_data["total_train_seconds"]))

            if m_file.exists():
                data = json.loads(m_file.read_text(encoding="utf-8"))
                m = data["metrics"]
                seed_accs.append(float(m["accuracy"]))
                seed_f1s.append(float(m["macro_f1"]))
                if m.get("roc_auc") is not None:
                    seed_aucs.append(float(m["roc_auc"]))
                lat = data.get("speed", {}).get("latency_ms_per_sample")
                if lat is not None:
                    seed_latencies.append(float(lat))
                if total_p is None:
                    total_p = data.get("parameters", {}).get("total")
                    trainable_p = data.get("parameters", {}).get("trainable")
            elif split == "val" and train_m_file.exists():
                t_data = json.loads(train_m_file.read_text(encoding="utf-8"))
                if t_data.get("best_val_acc") is not None:
                    seed_accs.append(float(t_data["best_val_acc"]))
                if t_data.get("val_f1_at_best") is not None:
                    seed_f1s.append(float(t_data["val_f1_at_best"]))
                if total_p is None:
                    total_p = t_data.get("parameters", {}).get("total")
                    trainable_p = t_data.get("parameters", {}).get("trainable")

        if seed_accs:
            acc_mean = float(np.mean(seed_accs))
            acc_std = float(np.std(seed_accs))
            f1_mean = float(np.mean(seed_f1s)) if seed_f1s else None
            f1_std = float(np.std(seed_f1s)) if seed_f1s else None
            auc_mean = float(np.mean(seed_aucs)) if seed_aucs else None
            lat_mean = round(float(np.mean(seed_latencies)), 3) if seed_latencies else None
            train_time_mean = round(float(np.mean(seed_train_times)), 1) if seed_train_times else None

            if split == "test":
                summary_payload = {
                    "exp_name": exp_name,
                    "split": split,
                    "n_seeds": len(seed_accs),
                    "accuracy": {"mean": round(acc_mean, 4), "std": round(acc_std, 4)},
                    "macro_f1": {"mean": round(f1_mean, 4), "std": round(f1_std, 4)} if f1_mean is not None else None,
                    "roc_auc": {"mean": round(auc_mean, 4)} if auc_mean is not None else None,
                    "total_params": total_p,
                    "trainable_params": trainable_p,
                    "train_time_s_mean": train_time_mean,
                    "latency_ms_sample_mean": lat_mean,
                }
                (exp_dir / f"{split}_summary.json").write_text(
                    json.dumps(summary_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            rows.append({
                "model": exp_name,
                "accuracy": f"{acc_mean * 100:.2f} ± {acc_std * 100:.2f}%",
                "macro_f1": f"{f1_mean:.4f} ± {f1_std:.4f}" if f1_mean is not None else "—",
                "roc_auc": f"{auc_mean:.4f}" if auc_mean is not None else "—",
                "total_params": f"{total_p:,}" if total_p is not None else "—",
                "trainable_params": f"{trainable_p:,}" if trainable_p is not None else "—",
                "train_time_s": f"{train_time_mean:.1f}" if train_time_mean is not None else "—",
                "latency_ms_sample": lat_mean if lat_mean is not None else "—",
            })

    if include_lab3_reference and split == "test" and rows:
        rows.append({
            "model": "DistilBERT (tham chiếu Lab 3, không train lại)",
            "accuracy": "93.00–93.40%",
            "macro_f1": "—",
            "roc_auc": "—",
            "total_params": "66,955,010",
            "trainable_params": "66,955,010",
            "train_time_s": "—",
            "latency_ms_sample": "—",
        })

    df_comp = pd.DataFrame(rows)
    out_csv = base / f"comparison_{split}.csv"
    df_comp.to_csv(out_csv, index=False)
    logger.info(f"Đã lưu bảng so sánh tổng hợp vào {out_csv}")
    return df_comp


def run_mcnemar_cli(
    pred_file_a: str | Path,
    pred_file_b: str | Path,
    model_a_name: str,
    model_b_name: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Chạy kiểm định McNemar giữa 2 file predictions và xuất ra JSON."""
    df_a = pd.read_csv(pred_file_a)
    df_b = pd.read_csv(pred_file_b)

    assert len(df_a) == len(df_b), f"Hai file dự đoán khác số dòng ({len(df_a)} vs {len(df_b)})!"
    if not (df_a["uid"].values == df_b["uid"].values).all():
        df_a = df_a.sort_values("uid", kind="stable").reset_index(drop=True)
        df_b = df_b.sort_values("uid", kind="stable").reset_index(drop=True)
    assert (df_a["uid"].values == df_b["uid"].values).all(), "Danh sách uid trong hai file không khớp!"

    res = mcnemar_test(
        y_true=df_a["label"].values,
        y_pred_a=df_a["pred"].values,
        y_pred_b=df_b["pred"].values,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
    )

    out_file = Path(output_path) if output_path is not None else (experiment_dir("results") / "mcnemar.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Đã lưu kết quả kiểm định McNemar vào {out_file}")
    return res


def evaluate_experiment_seeds(
    exp_name: str,
    split: str = "val",
    seeds: tuple[int, ...] = (42, 123, 2025),
    data_path: str | Path | None = None,
    batch_size: int = 64,
    device_name: str | None = None,
) -> list[dict[str, Any]]:
    """Đánh giá mọi seed của một cấu hình `<exp>`, đồng thời chép dự đoán của seed tốt nhất
    (chọn theo val_acc) ra `experiments/results/<exp>/predictions_<split>.csv` cho notebook trình bày."""
    ckpt_exp_dir = experiment_dir("checkpoints", exp_name)
    vocab_file = ckpt_exp_dir / "vocab.json"
    res_exp_dir = experiment_dir("results", exp_name)

    outputs: list[dict[str, Any]] = []
    best_seed: int | None = None
    best_val_acc = -1.0

    for s in seeds:
        ckpt_file = ckpt_exp_dir / f"seed_{s}" / "best.pt"
        if not ckpt_file.exists():
            continue
        out_dir = res_exp_dir / f"seed_{s}"
        res = evaluate_split(
            checkpoint_path=ckpt_file,
            vocab_path=vocab_file,
            split=split,
            data_path=data_path,
            output_dir=out_dir,
            device_name=device_name,
            batch_size=batch_size,
        )
        outputs.append(res)

        train_m = out_dir / "metrics.json"
        if train_m.exists():
            val_acc = float(json.loads(train_m.read_text(encoding="utf-8")).get("best_val_acc", -1.0))
        else:
            val_acc = float(res["metrics"]["accuracy"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_seed = s

    if best_seed is not None:
        src_csv = res_exp_dir / f"seed_{best_seed}" / f"predictions_{split}.csv"
        dst_csv = res_exp_dir / f"predictions_{split}.csv"
        if src_csv.exists():
            shutil.copyfile(src_csv, dst_csv)
            logger.info(f"Đã chép dự đoán đại diện (seed_{best_seed}) sang {dst_csv}")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình, lập bảng so sánh và kiểm định McNemar (Task D8).")
    parser.add_argument("--exp", type=str, default=None, help="Tên thí nghiệm trong experiments/checkpoints/<exp>")
    parser.add_argument("--seed", type=int, default=None, help="Seed cụ thể (bỏ trống để chạy mọi seed có sẵn của --exp)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Đường dẫn trực tiếp tới best.pt")
    parser.add_argument("--vocab", type=Path, default=None, help="Đường dẫn trực tiếp tới vocab.json")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Split cần đánh giá")
    parser.add_argument("--data-path", type=Path, default=None, help="Đường dẫn reviews.parquet tuỳ chọn")
    parser.add_argument("--output-dir", type=Path, default=None, help="Thư mục lưu kết quả")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size khi đánh giá")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị ('cpu' hoặc 'cuda')")
    parser.add_argument("--compare", action="store_true", help="Tổng hợp bảng comparison_<split>.csv")
    parser.add_argument("--mcnemar-a", type=Path, default=None, help="File predictions CSV của mô hình A")
    parser.add_argument("--mcnemar-b", type=Path, default=None, help="File predictions CSV của mô hình B")
    parser.add_argument("--name-a", type=str, default="Best_RNN", help="Tên mô hình A trong McNemar")
    parser.add_argument("--name-b", type=str, default="M0b_TFIDF_LR", help="Tên mô hình B trong McNemar")
    args = parser.parse_args()

    if args.checkpoint and args.vocab:
        evaluate_split(
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab,
            split=args.split,
            data_path=args.data_path,
            output_dir=args.output_dir,
            device_name=args.device,
            batch_size=args.batch_size,
        )
    elif args.exp:
        seeds = (args.seed,) if args.seed is not None else (42, 123, 2025)
        evaluate_experiment_seeds(
            exp_name=args.exp,
            split=args.split,
            seeds=seeds,
            data_path=args.data_path,
            batch_size=args.batch_size,
            device_name=args.device,
        )

    if args.compare or args.exp:
        df_comp = compare_all_models(split=args.split)
        if not df_comp.empty:
            print(df_comp.to_string(index=False))

    if args.mcnemar_a and args.mcnemar_b:
        run_mcnemar_cli(
            pred_file_a=args.mcnemar_a,
            pred_file_b=args.mcnemar_b,
            model_a_name=args.name_a,
            model_b_name=args.name_b,
        )


if __name__ == "__main__":
    main()
