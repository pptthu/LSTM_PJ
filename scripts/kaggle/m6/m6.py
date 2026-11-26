"""M6: clone repo sạch, chạy đúng lệnh README, train lại GRU seed 42, so với số đã commit."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
BRANCH = "feat/eval-explain"
ROOT = Path("/kaggle/tmp/LSTM_PJ_m6")
OUT = Path("/kaggle/working")
log = []


def run(cmd):
    t0 = time.time()
    print(">>", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=ROOT if ROOT.exists() else None, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    log.append({"cmd": cmd, "rc": r.returncode, "seconds": round(time.time() - t0), "tail": (r.stdout + r.stderr)[-800:]})
    (OUT / "m6_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    if r.returncode:
        raise SystemExit(f"FAILED: {cmd}")


subprocess.run(f"git clone -q --branch {BRANCH} https://github.com/pptthu/LSTM_PJ.git {ROOT}", shell=True, check=True)
head = subprocess.run("git rev-parse HEAD", shell=True, cwd=ROOT, capture_output=True, text=True).stdout.strip()
R = ROOT / "experiments" / "results"
ref = {
    "val_acc": json.loads((R / "gru/seed_42/metrics.json").read_text(encoding="utf-8"))["best_val_acc"],
    "test_acc": json.loads((R / "gru/seed_42/test_metrics.json").read_text(encoding="utf-8"))["metrics"]["accuracy"],
    "content_hash": json.loads((R / "data_prep/prepare_stats.json").read_text(encoding="utf-8"))["content_hash"],
}

# Đúng thứ tự README §4
run("python -m src.data.download")
run("python -m src.data.prepare")
run("python -m src.train --config configs/gru.yaml --seed 42")
run("python -m src.evaluate --exp gru --seed 42 --split val")
run("python -m src.evaluate --exp gru --seed 42 --split test")

new = {
    "val_acc": json.loads((R / "gru/seed_42/metrics.json").read_text(encoding="utf-8"))["best_val_acc"],
    "test_acc": json.loads((R / "gru/seed_42/test_metrics.json").read_text(encoding="utf-8"))["metrics"]["accuracy"],
    "content_hash": json.loads((R / "data_prep/prepare_stats.json").read_text(encoding="utf-8"))["content_hash"],
}
res = {
    "commit": head, "ref": ref, "new": new,
    "diff_val_pts": round((new["val_acc"] - ref["val_acc"]) * 100, 2),
    "diff_test_pts": round((new["test_acc"] - ref["test_acc"]) * 100, 2),
    "hash_same": new["content_hash"] == ref["content_hash"],
}
res["m6_pass"] = abs(res["diff_val_pts"]) <= 1 and abs(res["diff_test_pts"]) <= 1 and res["hash_same"]
(OUT / "m6_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
sys.exit(0)
