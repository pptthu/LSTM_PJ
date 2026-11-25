"""Theo dõi tiến độ huấn luyện (Monitor training progress).

Theo đặc tả trong docs/phases/TRAINING.md § Monitor:
1. Log mỗi epoch vào history.csv (ở cả experiments/runs/ và bản commit experiments/results/):
   epoch, train_loss, train_acc, val_loss, val_acc, val_f1, grad_norm_mean, lr, epoch_seconds
2. Early stopping theo val_accuracy (lưu checkpoint tốt nhất).
3. Vẽ đồ thị Learning curve (loss, acc) theo epoch, lưu vào experiments/plots/training/<exp>_curves.png.
4. Bảng tóm tắt metrics.json cho từng seed và summary.json (mean ± std) cho 3 seed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.paths import experiment_dir


class TrainingMonitor:
    def __init__(
        self,
        exp_name: str,
        seed: int,
        patience: int = 3,
        run_dir: Path | str | None = None,
        results_dir: Path | str | None = None,
    ) -> None:
        self.exp_name = exp_name
        self.seed = seed
        self.patience = patience

        self.run_dir = (
            Path(run_dir)
            if run_dir is not None
            else experiment_dir("runs", exp_name, f"seed_{seed}")
        )
        self.results_dir = (
            Path(results_dir)
            if results_dir is not None
            else experiment_dir("results", exp_name, f"seed_{seed}")
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.history_csv_path = self.run_dir / "history.csv"
        self.results_history_csv_path = self.results_dir / "history.csv"
        self.history: list[dict[str, Any]] = []

        self.best_val_acc = -1.0
        self.best_epoch = 0
        self.patience_counter = 0

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        val_f1: float,
        grad_norm_mean: float,
        lr: float,
        epoch_seconds: float,
    ) -> tuple[bool, bool]:
        """Ghi nhận log của một epoch (lưu vào cả runs/ và results/ theo TRAINING.md)."""
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "grad_norm_mean": round(grad_norm_mean, 4),
            "lr": float(f"{lr:.6e}"),
            "epoch_seconds": round(epoch_seconds, 2),
        }
        self.history.append(row)

        df = pd.DataFrame(self.history)
        df.to_csv(self.history_csv_path, index=False)
        if self.results_history_csv_path != self.history_csv_path:
            df.to_csv(self.results_history_csv_path, index=False)

        is_best = False
        should_stop = False
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.patience_counter = 0
            is_best = True
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                should_stop = True

        return is_best, should_stop

    def save_run_summary(
        self,
        total_time: float,
        hardware_info: str = "CPU",
        model_params: dict[str, Any] | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ) -> Path:
        """Lưu tóm tắt kết quả chạy ra file metrics.json."""
        summary: dict[str, Any] = {
            "exp_name": self.exp_name,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "stopped_epoch": len(self.history),
            "best_val_acc": round(self.best_val_acc, 4),
            "total_train_seconds": round(total_time, 2),
            "hardware": hardware_info,
            "parameters": model_params or {},
        }
        if extra_metrics:
            summary.update(extra_metrics)

        if self.history:
            best_records = [r for r in self.history if r["epoch"] == self.best_epoch]
            if best_records:
                best_rec = best_records[0]
                summary["val_loss_at_best"] = best_rec["val_loss"]
                summary["val_f1_at_best"] = best_rec["val_f1"]
                summary["train_acc_at_best"] = best_rec["train_acc"]
                summary["train_loss_at_best"] = best_rec["train_loss"]
                summary["overfit_gap_at_best"] = round(best_rec["train_acc"] - best_rec["val_acc"], 4)
            min_loss_rec = min(self.history, key=lambda r: r["val_loss"])
            summary["min_val_loss_epoch"] = int(min_loss_rec["epoch"])
            summary["min_val_loss"] = min_loss_rec["val_loss"]

        metrics_file = self.results_dir / "metrics.json"
        metrics_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return metrics_file


def plot_learning_curves(
    history_dfs_by_seed: dict[int, pd.DataFrame],
    save_path: Path | str,
    title: str = "Training and Validation Curves",
) -> Path:
    """Vẽ đồ thị Loss và Accuracy của train/val qua các epoch, hỗ trợ hiển thị nhiều seed."""
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_loss, ax_acc = axes[0], axes[1]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (seed, df) in enumerate(sorted(history_dfs_by_seed.items())):
        if df.empty:
            continue
        color = colors[i % len(colors)]
        epochs = df["epoch"]

        ax_loss.plot(epochs, df["train_loss"], linestyle="--", alpha=0.7, color=color, label=f"Train (seed {seed})")
        ax_loss.plot(epochs, df["val_loss"], linestyle="-", linewidth=2, color=color, label=f"Val (seed {seed})")

        ax_acc.plot(epochs, df["train_acc"], linestyle="--", alpha=0.7, color=color, label=f"Train (seed {seed})")
        ax_acc.plot(epochs, df["val_acc"], linestyle="-", linewidth=2, color=color, label=f"Val (seed {seed})")

        best_idx = df["val_acc"].idxmax()
        best_epoch = df.loc[best_idx, "epoch"]
        best_acc = df.loc[best_idx, "val_acc"]
        ax_acc.scatter([best_epoch], [best_acc], color=color, marker="*", s=110, zorder=5)

    ax_loss.set_title("Loss vs. Epoch", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(fontsize=9)

    ax_acc.set_title("Accuracy vs. Epoch (★ = Best Val Acc)", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_file


def aggregate_experiment_seeds(
    exp_name: str,
    seeds: list[int] | tuple[int, ...] = (42, 123, 2025),
    results_base: Path | str | None = None,
) -> dict[str, Any]:
    """Tổng hợp kết quả giữa các seed thành summary.json (mean ± std)."""
    base_root = Path(results_base) if results_base is not None else experiment_dir("results")
    base = base_root / exp_name
    metrics_list: list[dict[str, Any]] = []

    for s in seeds:
        metric_file = base / f"seed_{s}" / "metrics.json"
        if metric_file.exists():
            data = json.loads(metric_file.read_text(encoding="utf-8"))
            metrics_list.append(data)

    if not metrics_list:
        return {}

    keys_to_agg = ["best_val_acc", "val_loss_at_best", "val_f1_at_best", "train_acc_at_best", "total_train_seconds"]
    params = metrics_list[0].get("parameters", {})
    summary: dict[str, Any] = {
        "exp_name": exp_name,
        "n_seeds": len(metrics_list),
        "seeds": [m["seed"] for m in metrics_list],
        "parameters": params,
    }

    for k in keys_to_agg:
        vals = [m[k] for m in metrics_list if k in m and m[k] is not None]
        if vals:
            mean_val = float(np.mean(vals))
            ddof = 1 if len(vals) > 1 else 0
            std_val = float(np.std(vals, ddof=ddof))
            summary[k] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "formatted": (
                    f"{mean_val * 100:.2f} ± {std_val * 100:.2f}%"
                    if ("acc" in k or "f1" in k)
                    else f"{mean_val:.4f} ± {std_val:.4f}"
                ),
            }

    out_file = base / "summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
