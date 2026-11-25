"""Huấn luyện mô hình phân loại cảm xúc SentimentRNN.

Theo đặc tả trong docs/phases/TRAINING.md và docs/TASKS.md §3 D4:
- Nhận cấu hình từ configs/<exp>.yaml và seed (42, 123, 2025).
- Xây dựng từ vựng CHỈ từ train và lưu experiments/checkpoints/<exp>/vocab.json.
- Vòng train có gradient clipping (5.0) và BCEWithLogitsLoss.
- Early stopping và chọn checkpoint best.pt theo VAL ACCURACY.
- Ghi log epoch vào cả experiments/runs/<exp>/seed_<s>/history.csv và experiments/results/<exp>/seed_<s>/history.csv.
- Lưu metrics.json và summary.json (mean ± std 3 seed).
- Tự động vẽ learning curve sau mỗi lần huấn luyện.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.optim import Adam
import yaml

from src.data.dataset import IMDbDataset, create_dataloader, load_imdb_split_data
from src.data.tokenizer import tokenize
from src.data.vocab import Vocab
from src.models.glove import build_glove_matrix
from src.models.rnn import SentimentRNN
from src.training.monitor import (
    TrainingMonitor,
    aggregate_experiment_seeds,
    plot_learning_curves,
)
from src.utils.paths import data_dir, experiment_dir
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def _stratified_subset(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Lấy tập con cân bằng theo nhãn (tránh lỗi .head(n) chỉ lấy toàn review nhãn 0 ở đầu file)."""
    if n >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    parts: list[pd.DataFrame] = []
    labels = sorted(df["label"].unique())
    per_class = max(1, n // max(1, len(labels)))
    taken = 0
    for idx, lbl in enumerate(labels):
        sub = df[df["label"] == lbl]
        k = per_class if idx < len(labels) - 1 else max(1, n - taken)
        k = min(k, len(sub))
        chosen_idx = rng.choice(sub.index.to_numpy(), size=k, replace=False)
        parts.append(df.loc[sorted(chosen_idx)])
        taken += k
    return pd.concat(parts, ignore_index=True)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_norm: float = 5.0,
) -> tuple[float, float, float]:
    """Chạy 1 epoch huấn luyện, trả về (avg_loss, accuracy, mean_grad_norm)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    grad_norms: list[float] = []
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()

        if trainable_params:
            gnorm = nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_norm)
            grad_norms.append(float(gnorm.item() if hasattr(gnorm, "item") else gnorm))

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    acc = correct / max(1, total_samples)
    mean_grad = float(sum(grad_norms) / max(1, len(grad_norms)))
    return avg_loss, acc, mean_grad


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Đánh giá trên tập val, trả về (avg_loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"]
            labels = batch["labels"].to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().tolist()
            targets = labels.long().cpu().tolist()

            all_preds.extend(preds)
            all_targets.extend(targets)
            total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    acc = correct / max(1, total_samples)
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, macro_f1


def run_training(
    config_path: str | Path,
    seed: int = 42,
    exp_name_override: str | None = None,
    sample_size: int | None = None,
    epochs_override: int | None = None,
    device_name: str | None = None,
    data_path: str | Path | None = None,
    experiments_root: str | Path | None = None,
    allow_random_glove: bool = False,
) -> dict[str, Any]:
    """Hàm chạy quy trình huấn luyện đầy đủ theo config."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file config tại {cfg_file}")

    with cfg_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_name = exp_name_override or cfg_file.stem
    logger.info(f"Bắt đầu huấn luyện thí nghiệm '{exp_name}' với seed {seed}...")

    # 1. Cố định seed
    set_seed(seed)

    # 2. Thiết bị phần cứng
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    logger.info(f"Sử dụng thiết bị: {device} ({hardware_name})")

    # 3. Đọc dữ liệu train và val (TUYỆT ĐỐI không đọc test trong vòng train)
    train_df = load_imdb_split_data(parquet_path=data_path, split="train")
    val_df = load_imdb_split_data(parquet_path=data_path, split="val")

    if sample_size is not None and sample_size > 0:
        logger.info(f"Chạy chế độ thử nghiệm cân bằng lớp (sample_size={sample_size} mẫu)")
        train_df = _stratified_subset(train_df, sample_size, seed=seed)
        val_df = _stratified_subset(val_df, min(sample_size, len(val_df)), seed=seed)

    logger.info(f"Số lượng mẫu: train={len(train_df)}, val={len(val_df)}")

    # 4. Đường dẫn thư mục thí nghiệm
    if experiments_root is not None:
        exp_root = Path(experiments_root)
        ckpt_dir = exp_root / "checkpoints" / exp_name
        run_dir = exp_root / "runs" / exp_name / f"seed_{seed}"
        res_base = exp_root / "results"
        res_dir = res_base / exp_name / f"seed_{seed}"
        plots_dir = exp_root / "plots" / "training"
    else:
        ckpt_dir = experiment_dir("checkpoints", exp_name)
        run_dir = experiment_dir("runs", exp_name, f"seed_{seed}")
        res_base = experiment_dir("results")
        res_dir = experiment_dir("results", exp_name, f"seed_{seed}")
        plots_dir = experiment_dir("plots", "training")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 5. Quản lý Từ vựng (Vocab) — Xây dựng CHỈ từ train
    vocab_path = ckpt_dir / "vocab.json"
    data_cfg = config.get("data", {})
    min_freq = data_cfg.get("min_freq", 2)
    max_vocab_size = data_cfg.get("max_vocab_size", 25000)

    if vocab_path.exists() and sample_size is None:
        logger.info(f"Tải từ vựng có sẵn từ {vocab_path}")
        vocab = Vocab.load(vocab_path)
    else:
        logger.info(f"Xây dựng từ vựng từ {len(train_df)} review train...")
        train_tokens = [tokenize(text) for text in train_df["text_clean"]]
        vocab = Vocab.build_from_tokens(train_tokens, min_freq=min_freq, max_size=max_vocab_size)
        vocab.save(vocab_path)
        logger.info(f"Đã lưu từ vựng ({len(vocab)} từ) vào {vocab_path}")

    # 6. Xử lý Embedding (GloVe nếu có yêu cầu)
    model_cfg = config.get("model", {})
    embed_dim = model_cfg.get("embed_dim", 100)
    pretrained_weights = None
    glove_coverage: float | None = None

    if model_cfg.get("use_pretrained_embeddings", False):
        glove_path = model_cfg.get("glove_path", None)
        npy_cache = data_dir("processed") / "glove_100d_vocab.npy" if experiments_root is None and sample_size is None else None
        if npy_cache is not None and npy_cache.exists() and np.load(npy_cache).shape == (len(vocab), embed_dim):
            logger.info(f"Nạp ma trận GloVe đã lưu từ {npy_cache}")
            pretrained_weights = np.load(npy_cache)
        else:
            matrix, cov = build_glove_matrix(
                vocab,
                glove_path=glove_path,
                embed_dim=embed_dim,
                output_path=npy_cache,
                allow_random=allow_random_glove,
            )
            pretrained_weights = matrix
            glove_coverage = cov
            logger.info(f"Nạp ma trận GloVe hoàn tất (độ phủ: {cov * 100:.2f}%)")

    # 7. Tạo Dataset và DataLoader
    max_len = data_cfg.get("max_len", 250)
    truncation = data_cfg.get("truncation", "post")
    batch_size = data_cfg.get("batch_size", 64)

    train_ds = IMDbDataset(
        texts=train_df["text_clean"],
        labels=train_df["label"],
        uids=train_df["uid"],
        ratings=train_df["rating"],
        vocab=vocab,
        max_len=max_len,
        truncation=truncation,
    )
    val_ds = IMDbDataset(
        texts=val_df["text_clean"],
        labels=val_df["label"],
        uids=val_df["uid"],
        ratings=val_df["rating"],
        vocab=vocab,
        max_len=max_len,
        truncation=truncation,
    )

    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    # 8. Khởi tạo mô hình
    model = SentimentRNN(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        rnn_type=model_cfg.get("rnn_type", "lstm"),
        bidirectional=model_cfg.get("bidirectional", False),
        num_layers=model_cfg.get("num_layers", 1),
        dropout=model_cfg.get("dropout", 0.5),
        padding_idx=vocab.pad_idx,
        pretrained_embeddings=pretrained_weights,
        freeze=model_cfg.get("freeze_embeddings", False),
        pooling=model_cfg.get("pooling", "last"),
    ).to(device)

    param_counts = model.count_parameters()
    logger.info(f"Mô hình: {param_counts['total']:,} tham số (trainable: {param_counts['trainable']:,})")

    # 9. Thiết lập tối ưu và monitor
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    max_epochs = epochs_override or train_cfg.get("max_epochs", 10)
    patience = train_cfg.get("patience", 3)
    clip_norm = float(train_cfg.get("clip_grad_norm", 5.0))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    monitor = TrainingMonitor(
        exp_name=exp_name,
        seed=seed,
        patience=patience,
        run_dir=run_dir,
        results_dir=res_dir,
    )

    best_checkpoint_path = ckpt_dir / f"seed_{seed}" / "best.pt"
    start_time = time.time()

    # 10. Vòng lặp huấn luyện
    logger.info(f"Bắt đầu huấn luyện tối đa {max_epochs} epochs (patience={patience})...")

    for epoch in range(1, max_epochs + 1):
        ep_start = time.time()
        tr_loss, tr_acc, grad_norm = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            clip_norm=clip_norm,
        )
        va_loss, va_acc, va_f1 = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        ep_duration = time.time() - ep_start

        is_best, should_stop = monitor.log_epoch(
            epoch=epoch,
            train_loss=tr_loss,
            train_acc=tr_acc,
            val_loss=va_loss,
            val_acc=va_acc,
            val_f1=va_f1,
            grad_norm_mean=grad_norm,
            lr=lr,
            epoch_seconds=ep_duration,
        )

        star = " ★ (Best)" if is_best else ""
        logger.info(
            f"Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
            f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} Macro-F1: {va_f1:.4f}{star} | "
            f"Grad: {grad_norm:.2f} | Time: {ep_duration:.1f}s"
        )

        if is_best:
            model.save_checkpoint(
                best_checkpoint_path,
                config=config,
                extra={"val_acc": va_acc, "val_loss": va_loss, "val_f1": va_f1, "epoch": epoch, "seed": seed},
            )

        if should_stop:
            logger.info(f"Early stopping kích hoạt tại epoch {epoch} (val acc không tăng sau {patience} epochs).")
            break

    total_training_time = time.time() - start_time
    logger.info(
        f"Hoàn thành huấn luyện seed {seed} trong {total_training_time:.1f}s. "
        f"Val Acc tốt nhất: {monitor.best_val_acc * 100:.2f}% (Epoch {monitor.best_epoch})"
    )

    extra_info: dict[str, Any] = {}
    if glove_coverage is not None:
        extra_info["glove_coverage"] = round(glove_coverage, 6)

    monitor.save_run_summary(
        total_time=total_training_time,
        hardware_info=hardware_name,
        model_params=param_counts,
        extra_metrics=extra_info or None,
    )

    # 11. Đọc lại các seed đã chạy để vẽ learning curve tổng hợp
    exp_res_dir = res_base / exp_name
    history_dfs: dict[int, pd.DataFrame] = {}
    for s_dir in sorted(exp_res_dir.glob("seed_*")):
        try:
            s_num = int(s_dir.name.replace("seed_", ""))
            h_csv = s_dir / "history.csv"
            if h_csv.exists():
                history_dfs[s_num] = pd.read_csv(h_csv)
        except ValueError:
            continue

    if history_dfs:
        curve_plot_path = plots_dir / f"{exp_name}_curves.png"
        plot_learning_curves(history_dfs, curve_plot_path, title=f"Learning Curves: {exp_name.upper()}")
        logger.info(f"Đã cập nhật đồ thị learning curve tại {curve_plot_path}")

    summary = aggregate_experiment_seeds(exp_name, results_base=res_base)
    return {
        "exp_name": exp_name,
        "seed": seed,
        "best_val_acc": monitor.best_val_acc,
        "best_epoch": monitor.best_epoch,
        "total_time": total_training_time,
        "checkpoint_path": str(best_checkpoint_path),
        "vocab_path": str(vocab_path),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình SentimentRNN (Task D4-D7).")
    parser.add_argument("--config", type=Path, required=True, help="Đường dẫn file YAML cấu hình (vd: configs/lstm.yaml)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên (vd: 42, 123, 2025)")
    parser.add_argument("--exp-name", type=str, default=None, help="Tên thí nghiệm ghi đè (mặc định theo tên file config)")
    parser.add_argument("--sample-size", type=int, default=None, help="Số mẫu chạy thử nhanh cân bằng lớp (vd: 500)")
    parser.add_argument("--epochs", type=int, default=None, help="Số epoch ghi đè (vd: 1 khi kiểm thử)")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị ('cpu' hoặc 'cuda')")
    parser.add_argument("--data-path", type=Path, default=None, help="Đường dẫn reviews.parquet tuỳ chọn")
    parser.add_argument("--allow-random-glove", action="store_true", help="Cho phép dùng vector ngẫu nhiên nếu chưa có GloVe")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        seed=args.seed,
        exp_name_override=args.exp_name,
        sample_size=args.sample_size,
        epochs_override=args.epochs,
        device_name=args.device,
        data_path=args.data_path,
        allow_random_glove=args.allow_random_glove,
    )


if __name__ == "__main__":
    main()
