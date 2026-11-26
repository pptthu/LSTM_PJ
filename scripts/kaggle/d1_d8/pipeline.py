"""Chạy D1–D8 của LSTM_PJ trên Kaggle GPU — cùng các bước với notebooks/colab_d1_d8.ipynb (Đại).

Khác notebook: mỗi giai đoạn ghi trạng thái vào stages.json và chép kết quả ra /kaggle/working ngay,
để lỗi giữa chừng không mất phần đã chạy. LSTMPJ_SMOKE=1: chạy thử 1 epoch trên dữ liệu nhỏ (máy CPU).
"""
import copy
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 1 GPU, tránh DataParallel trên T4 x2

SMOKE = os.environ.get("LSTMPJ_SMOKE") == "1"
COMMIT = os.environ.get("LSTMPJ_COMMIT", "e1f38ecb2b1c7658b7cc5c14851818a1ea7cccbe")
WORK = Path(os.environ.get("LSTMPJ_OUT", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)
STAGES = WORK / "stages.json"
state = json.loads(STAGES.read_text(encoding="utf-8")) if STAGES.exists() and STAGES.stat().st_size else {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mark(stage, status, **extra):
    state[stage] = {"status": status, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    STAGES.write_text(json.dumps(state, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


# ------------------------------------------------------------------ code
if os.environ.get("LSTMPJ_REPO"):
    ROOT = Path(os.environ["LSTMPJ_REPO"]).resolve()
else:
    ROOT = Path("/kaggle/tmp/LSTM_PJ")
    if not ROOT.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/pptthu/LSTM_PJ.git", str(ROOT)], check=True)
    subprocess.run(["git", "checkout", "-q", COMMIT], cwd=ROOT, check=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

env = {"code_commit": head, "smoke": SMOKE, "python": sys.version.split()[0], "torch": torch.__version__,
       "pandas": pd.__version__, "cuda": torch.cuda.is_available(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
log("ENV", env)
mark("env", "ok", **env)

RES = ROOT / "experiments" / "results"
CONFIG_LIST = ["lstm", "gru", "bilstm", "bilstm_glove_frozen", "bilstm_glove_ft", "bilstm_attention"]
SEEDS = (42, 123, 2025)
CFG_TMP = ROOT / "experiments" / "runs" / "_cfg"
CFG_TMP.mkdir(parents=True, exist_ok=True)


def cfg_path(name: str, cfg: dict | None = None) -> Path:
    """Config gốc; khi SMOKE thì chép sang thư mục tạm (cùng tên → cùng exp_name) với max_epochs=1."""
    src = ROOT / "configs" / f"{name}.yaml"
    if cfg is None and not SMOKE:
        return src
    c = copy.deepcopy(cfg) if cfg is not None else yaml.safe_load(src.read_text(encoding="utf-8"))
    if SMOKE:
        c["training"]["max_epochs"] = 1
    p = CFG_TMP / f"{name}.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    return p


def save_progress(tag: str) -> None:
    """Chép results + plots (nhỏ) ra WORK sau mỗi giai đoạn."""
    for sub in ("results", "plots"):
        src = ROOT / "experiments" / sub
        if src.exists():
            shutil.copytree(src, WORK / "experiments" / sub, dirs_exist_ok=True)
    log("saved progress after", tag)


def stage(name):
    def deco(fn):
        def run(*a, **k):
            if state.get(name, {}).get("status") == "ok" and not os.environ.get("LSTMPJ_FORCE"):
                log("skip (done)", name)
                return state[name].get("value")
            log(f"===== {name} =====")
            t0 = time.time()
            mark(name, "running")
            try:
                v = fn(*a, **k)
            except Exception:
                tb = traceback.format_exc()
                log("FAILED", name, "\n" + tb)
                mark(name, "error", traceback=tb[-4000:], seconds=round(time.time() - t0))
                save_progress(name + " (error)")
                raise
            mark(name, "ok", seconds=round(time.time() - t0), value=v)
            save_progress(name)
            return v
        return run
    return deco


# ------------------------------------------------------------------ B1
@stage("b1_data")
def b1():
    from src.data.prepare import check_contract, prepare_dataset

    pq = ROOT / "data" / "processed" / "reviews.parquet"
    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df, _ = prepare_dataset()
    c = check_contract(df)
    h = int(pd.util.hash_pandas_object(df, index=False).sum())
    expected = json.loads((RES / "data_prep" / "prepare_stats.json").read_text(encoding="utf-8"))["content_hash"]
    log("split", c["n_by_split"], "hash", h, "expected", expected)
    if not SMOKE:
        assert c["n_by_split"] == {"train": 22304, "val": 2473, "test": 25000}, c["n_by_split"]
        assert h == expected, f"content_hash {h} != {expected}"
    return {"n_by_split": c["n_by_split"], "content_hash": h, "hash_ok": h == expected}


# ------------------------------------------------------------------ D1–D3
@stage("d3_glove")
def d3():
    from src.data.tokenizer import tokenize
    from src.data.vocab import Vocab
    from src.models.glove import build_glove_matrix

    df = pd.read_parquet(ROOT / "data" / "processed" / "reviews.parquet")
    train = df[df["split"] == "train"]
    vocab = Vocab.build_from_tokens((tokenize(t) for t in train["text_clean"]), min_freq=2, max_size=25000)
    vocab.save(ROOT / "data" / "processed" / "vocab.json")
    gdir = ROOT / "data" / "glove"
    gdir.mkdir(parents=True, exist_ok=True)
    gtxt = gdir / "glove.6B.100d.txt"
    if not gtxt.exists():
        found = [p for p in glob.glob("/kaggle/input/**/*", recursive=True)
                 if os.path.isfile(p) and os.path.getsize(p) == 347116733]
        if found:
            os.symlink(found[0], gtxt)
            log("GloVe from Kaggle dataset", found[0])
        elif SMOKE:
            rng = np.random.default_rng(0)
            with open(gtxt, "w", encoding="utf-8") as f:
                for w in list(vocab.token_to_idx)[2:]:
                    f.write(w + " " + " ".join(f"{x:.4f}" for x in rng.normal(size=100)) + "\n")
        else:
            import urllib.request
            import zipfile
            z = gdir / "glove.6B.zip"
            urllib.request.urlretrieve("https://nlp.stanford.edu/data/glove.6B.zip", z)
            zipfile.ZipFile(z).extract("glove.6B.100d.txt", path=gdir)
    _, cov = build_glove_matrix(vocab=vocab, glove_path=gtxt, embed_dim=100,
                                output_path=ROOT / "data" / "processed" / "glove_100d_vocab.npy")
    log("vocab", len(vocab), "glove coverage", round(cov, 4))
    return {"vocab": len(vocab), "glove_coverage": round(float(cov), 4)}


# ------------------------------------------------------------------ D5
PILOT_MAXLEN = (200, 250, 400, 600)
TIE = 0.005  # quy tắc chốt trước khi chạy: chênh < 0,5 điểm val acc thì chọn max_len nhỏ hơn (nhanh hơn)


def with_maxlen(name: str, ml: int, tr: str = "post") -> dict:
    c = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
    c["data"]["max_len"], c["data"]["truncation"] = ml, tr
    return c


@stage("d5_pilot")
def d5():
    """Plan D5: pilot M1 (LSTM, seed 42) trên val → chốt max_len; kiểm trần epoch / patience."""
    from src.train import run_training

    rows = []
    for ml in PILOT_MAXLEN:
        v = f"pilot_lstm_ml{ml}"
        r = run_training(config_path=cfg_path(v, with_maxlen("lstm", ml)), seed=42, exp_name_override=v)
        rows.append({"max_len": ml, "best_val_acc": round(r["best_val_acc"], 4), "best_epoch": r["best_epoch"],
                     "train_seconds": round(r["total_time"], 1)})
        log("pilot", rows[-1])
        pd.DataFrame(rows).to_csv(RES / "pilot_maxlen.csv", index=False)
    top = max(x["best_val_acc"] for x in rows)
    chosen = min(x["max_len"] for x in rows if x["best_val_acc"] >= top - TIE)
    max_ep = yaml.safe_load((ROOT / "configs" / "lstm.yaml").read_text(encoding="utf-8"))["training"]["max_epochs"]
    note = {"rows": rows, "rule": f"val acc cao nhất; chênh < {TIE*100:.1f} điểm thì chọn max_len nhỏ hơn",
            "chosen_max_len": chosen, "max_epochs": max_ep,
            "epoch_cap_ok": all(x["best_epoch"] < max_ep for x in rows)}
    (RES / "pilot_decision.json").write_text(json.dumps(note, indent=1, ensure_ascii=False), encoding="utf-8")
    log("D5 chốt max_len =", chosen, note)
    return note


# ------------------------------------------------------------------ D6
def d6_one(name, ml):
    @stage(f"d6_{name}")
    def run():
        from src.evaluate import evaluate_experiment_seeds
        from src.train import run_training

        out = {}
        for s in SEEDS:
            r = run_training(config_path=cfg_path(name, with_maxlen(name, ml)), seed=s)
            out[s] = round(r["best_val_acc"], 4)
        evaluate_experiment_seeds(exp_name=name, split="val", seeds=SEEDS)
        return out
    return run()


@stage("d6_compare_val")
def d6_compare():
    from src.evaluate import compare_all_models

    compare_all_models(split="val")
    best, best_acc = None, -1.0
    for name in CONFIG_LIST:
        s = RES / name / "summary.json"
        if s.exists():
            acc = float(json.loads(s.read_text(encoding="utf-8"))["best_val_acc"]["mean"])
            if acc > best_acc:
                best, best_acc = name, acc
    log("BEST by mean val acc:", best, best_acc)
    return {"best": best, "best_val_mean": best_acc}


# ------------------------------------------------------------------ D7
@stage("d7_maxlen")
def d7(best, ml_chosen):
    from src.data.tokenizer import tokenize
    from src.evaluate import evaluate_split
    from src.train import run_training

    base = with_maxlen(best, ml_chosen)
    train = pd.read_parquet(ROOT / "data" / "processed" / "reviews.parquet").query("split == 'train'")
    lens = np.array([len(tokenize(t)) for t in train["text_clean"]])
    rows = []
    grid = [(f"maxlen_{m}_post", m, "post") for m in (200, 400, 600) if m != ml_chosen]
    grid += [(f"maxlen_{ml_chosen}_{tr}", ml_chosen, tr) for tr in ("pre", "head_tail")]
    for v, ml, tr in grid:
        c = copy.deepcopy(base)
        c["data"]["max_len"], c["data"]["truncation"] = ml, tr
        r = run_training(config_path=cfg_path(v, c), seed=42, exp_name_override=v)
        e = evaluate_split(checkpoint_path=r["checkpoint_path"], vocab_path=r["vocab_path"], split="val",
                           output_dir=RES / v / "seed_42")
        rows.append({"variant": v, "base_model": best, "max_len": ml, "truncation": tr,
                     "pct_train_truncated": round(float((lens > ml).mean() * 100), 2),
                     "val_accuracy": e["metrics"]["accuracy"], "val_macro_f1": e["metrics"]["macro_f1"],
                     "val_roc_auc": e["metrics"]["roc_auc"], "train_seconds": round(r["total_time"], 1),
                     "latency_ms_per_sample": e["speed"]["latency_ms_per_sample"]})
        pd.DataFrame(rows).to_csv(RES / "maxlen_comparison.csv", index=False)
        save_progress(v)
    return rows


# ------------------------------------------------------------------ D8
@stage("d8_test")
def d8(best):
    from src.evaluate import compare_all_models, evaluate_experiment_seeds, run_mcnemar_cli
    from src.models.baseline import evaluate_baseline_on_test

    bl = evaluate_baseline_on_test()
    for name in CONFIG_LIST:
        evaluate_experiment_seeds(exp_name=name, split="test", seeds=SEEDS)
    compare_all_models(split="test")
    m = run_mcnemar_cli(pred_file_a=RES / best / "predictions_test.csv",
                        pred_file_b=RES / "baseline" / "predictions_test.csv",
                        model_a_name=best, model_b_name="M0b_TFIDF_LR")
    log("McNemar", m.get("interpretation"))
    return {"baseline_test_acc": bl["M0b_tfidf_lr"]["test"]["accuracy"], "mcnemar": m.get("interpretation")}


def bundle():
    ck = ROOT / "experiments" / "checkpoints"
    if ck.exists():
        shutil.copytree(ck, WORK / "experiments" / "checkpoints", dirs_exist_ok=True)
    runs = ROOT / "experiments" / "runs"
    if runs.exists():
        shutil.copytree(runs, WORK / "experiments" / "runs", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.pt"))
    save_progress("bundle")


if __name__ == "__main__":
    ok = True
    try:
        b1()
        d3()
        ml = d5()["chosen_max_len"]
        for n in CONFIG_LIST:
            d6_one(n, ml)
        best = d6_compare()["best"]
        d7(best, ml)
        d8(best)
    except Exception:
        ok = False
    finally:
        bundle()
        mark("done", "ok" if ok else "error")
        log("PIPELINE", "OK" if ok else "ERROR — xem stages.json")
    sys.exit(0 if ok else 1)
