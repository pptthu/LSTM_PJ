"""Cố định seed cho tính tái lập (torch, numpy, random).

Theo docs/phases/TRAINING.md: Split không đổi theo seed; chỉ cố định seed khởi tạo mô hình và shuffle.
"""
from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Thiết lập seed cho random, numpy, torch CPU & GPU."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
