"""Phân tích lỗi theo 7 nhóm tình huống (T2, docs/phases/EVAL.md §3).

Chỉ đọc file dự đoán đã có (không chạy lại mô hình trên test):
- mô hình tốt nhất: `experiments/results/<exp>/predictions_test.csv` (chọn theo mean val acc 3 seed)
- baseline:         `experiments/results/baseline/predictions_test.csv` (nếu có, để so cùng nhóm)

Nhóm tình huống (quy tắc gán ghi ở `RULES`, đưa nguyên vào báo cáo):
1. Độ dài (số token) và review bị cắt ở `max_len`.
2. Phủ định: số từ phủ định trong review.
3. Khen chê lẫn lộn: có từ đổi ý (but, however, ...) ở 30% cuối review.
4. Mỉa mai: ứng viên theo từ khoá / dấu nháy quanh từ khen (heuristic, cần gán tay xác nhận).
5. Điểm sát ngưỡng (4, 7) so với cực trị (1, 10).
6. Tỷ lệ từ ngoài vocab (OOV).
7. 30 câu tự viết `data/sample/handwritten_30.csv` (cần checkpoint, chạy trên CPU).

Chạy từ gốc repo:
    python -m src.eval.error_analysis                     # tự chọn mô hình tốt nhất
    python -m src.eval.error_analysis --exp bilstm_attention
    python -m src.eval.error_analysis --results-dir <thư mục results khác> --out <thư mục ra>

Đầu ra (`experiments/results/error_analysis/`):
    by_group.csv            accuracy từng nhóm con: mô hình vs baseline, số mẫu, số lỗi
    error_examples.csv      ≤ 10 lỗi tự tin nhất mỗi nhóm khó, cột `explanation` để Thư viết
    manual_label_50.csv     50 lỗi để gán tay (ưu tiên ứng viên khen chê lẫn lộn / mỉa mai)
    handwritten_30_predictions.csv   (nếu có checkpoint)
    summary.json
    experiments/plots/error_analysis/acc_by_group.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import REPO_ROOT, data_dir, experiment_dir

MAIN_SKIP_PREFIX = ("pilot_", "maxlen_")
NON_MODEL_DIRS = {"baseline", "majority", "data_prep", "eda", "error_analysis", "explain", "demo"}

NEG_RE = re.compile(r"\b(?:not|no|never|nothing|nobody|none|neither|nor|hardly|barely|without)\b|n't\b")
CONTRAST_RE = re.compile(
    r"\b(?:but|however|although|though|yet|nevertheless|nonetheless|despite|whereas|on the other hand)\b"
)
SARCASM_RE = re.compile(
    r"\b(?:yeah,? right|oh,? great|oh,? joy|so bad (?:it'?s|that it'?s) good|thanks a lot|"
    r"what a (?:masterpiece|gem|treat)|bravo|pure genius)\b"
    r"|[\"'](?:masterpiece|brilliant|genius|great|classic|best|amazing|funny|scary|hilarious|"
    r"acting|plot|story|actors?|film|movie)[\"']"
)
TURN_AT = 0.70  # từ đổi ý nằm trong 30% cuối review

RULES = {
    "length": "số token (tokenizer của mô hình): <100, 100–300, 300–600, >600; và was_truncated (dài hơn max_len)",
    "negation": f"số lần khớp {NEG_RE.pattern!r}: 0–2, 3–5, 6+",
    "contrast_end": f"có từ khớp {CONTRAST_RE.pattern!r} bắt đầu sau {int(TURN_AT * 100)}% độ dài review",
    "sarcasm_candidate": "khớp từ khoá mỉa mai hoặc dấu nháy quanh từ khen (heuristic; xác nhận bằng gán tay)",
    "rating": "sát ngưỡng = 4 hoặc 7; cực trị = 1 hoặc 10; còn lại = 2, 3, 8, 9",
    "oov": "n_unk / n_tokens: <1%, 1–3%, >3%",
}
HARD_SUBGROUPS = [
    ("length", ">600"), ("truncated", "True"), ("negation", "6+"), ("contrast_end", "True"),
    ("sarcasm_candidate", "True"), ("rating", "sát ngưỡng (4, 7)"), ("oov", ">3%"),
]


# ------------------------------------------------------------------ chọn mô hình
def pick_best_exp(results_dir: Path) -> tuple[str, int]:
    """Mô hình chính có mean val acc 3 seed cao nhất; seed đại diện = seed có val acc cao nhất."""
    best, best_acc = None, -1.0
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name in NON_MODEL_DIRS or d.name.startswith(MAIN_SKIP_PREFIX):
            continue
        s = d / "summary.json"
        if not s.exists():
            continue
        info = json.loads(s.read_text(encoding="utf-8"))
        if "best_val_acc" not in info or int(info.get("n_seeds", 3)) < 2:  # bỏ thư mục không phải mô hình, chạy thử 1 seed
            continue
        acc = float(info["best_val_acc"]["mean"])
        if acc > best_acc:
            best, best_acc = d.name, acc
    if best is None:
        raise FileNotFoundError(f"Không có summary.json của mô hình chính trong {results_dir}")
    return best, pick_seed(results_dir / best)


def pick_seed(exp_dir: Path) -> int:
    accs = {}
    for m in exp_dir.glob("seed_*/metrics.json"):
        accs[int(m.parent.name.split("_")[1])] = float(json.loads(m.read_text(encoding="utf-8"))["best_val_acc"])
    if not accs:
        raise FileNotFoundError(f"Không có seed_*/metrics.json trong {exp_dir}")
    return max(accs, key=accs.get)


def load_predictions(exp_dir: Path, seed: int | None) -> pd.DataFrame:
    p = exp_dir / f"seed_{seed}" / "predictions_test.csv" if seed is not None else exp_dir / "predictions_test.csv"
    if not p.exists():
        p = exp_dir / "predictions_test.csv"
    df = pd.read_csv(p)
    need = {"uid", "text", "label", "rating", "prob", "pred", "n_tokens", "n_unk", "was_truncated"}
    if not need <= set(df.columns):
        raise ValueError(f"{p} thiếu cột {sorted(need - set(df.columns))}")
    return df


# ------------------------------------------------------------------ gán nhóm
def _norm(text: str) -> str:
    return text.replace("<br />", " ").lower()


def contrast_at_end(text: str) -> bool:
    t = _norm(text)
    return any(m.start() >= TURN_AT * len(t) for m in CONTRAST_RE.finditer(t))


def add_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    norm = out["text"].map(_norm)
    out["length"] = pd.cut(out["n_tokens"], [-1, 99, 300, 600, np.inf],
                           labels=["<100", "100–300", "300–600", ">600"]).astype(str)
    out["truncated"] = out["was_truncated"].astype(bool).astype(str)
    n_neg = norm.map(lambda t: len(NEG_RE.findall(t)))
    out["n_negation"] = n_neg
    out["negation"] = pd.cut(n_neg, [-1, 2, 5, np.inf], labels=["0–2", "3–5", "6+"]).astype(str)
    out["contrast_end"] = out["text"].map(contrast_at_end).astype(str)
    out["sarcasm_candidate"] = norm.map(lambda t: bool(SARCASM_RE.search(t))).astype(str)
    out["rating_group"] = out["rating"].map(
        lambda r: "sát ngưỡng (4, 7)" if r in (4, 7) else ("cực trị (1, 10)" if r in (1, 10) else "giữa (2, 3, 8, 9)"))
    oov = out["n_unk"] / out["n_tokens"].clip(lower=1)
    out["oov"] = pd.cut(oov, [-1, 0.01, 0.03, np.inf], labels=["<1%", "1–3%", ">3%"]).astype(str)
    out["correct"] = out["label"] == out["pred"]
    return out


GROUP_COLS = {"length": "length", "truncated": "truncated", "negation": "negation",
              "contrast_end": "contrast_end", "sarcasm_candidate": "sarcasm_candidate",
              "rating": "rating_group", "oov": "oov"}


def by_group(df: pd.DataFrame, baseline: pd.DataFrame | None = None) -> pd.DataFrame:
    """Accuracy từng nhóm con; nếu có baseline thì so trên đúng cùng các review (khớp theo uid)."""
    d = df.copy()
    if baseline is not None:
        d = d.merge(baseline[["uid", "pred"]].rename(columns={"pred": "pred_baseline"}), on="uid", how="left")
        d["correct_baseline"] = d["label"] == d["pred_baseline"]
    rows = []
    for group, col in GROUP_COLS.items():
        for sub, g in d.groupby(col, sort=False):
            row = {"group": group, "subgroup": sub, "n": len(g), "share": len(g) / len(d),
                   "acc_model": g["correct"].mean(), "n_errors_model": int((~g["correct"]).sum())}
            if baseline is not None:
                row["acc_baseline"] = g["correct_baseline"].mean()
                row["delta_model_minus_baseline"] = row["acc_model"] - row["acc_baseline"]
            rows.append(row)
    out = pd.DataFrame(rows)
    order = {"length": ["<100", "100–300", "300–600", ">600"], "negation": ["0–2", "3–5", "6+"],
             "oov": ["<1%", "1–3%", ">3%"], "truncated": ["False", "True"],
             "contrast_end": ["False", "True"], "sarcasm_candidate": ["False", "True"],
             "rating": ["cực trị (1, 10)", "giữa (2, 3, 8, 9)", "sát ngưỡng (4, 7)"]}
    out["_o"] = [order[g].index(s) if s in order[g] else 99 for g, s in zip(out["group"], out["subgroup"])]
    out["_g"] = out["group"].map(list(GROUP_COLS).index)
    return out.sort_values(["_g", "_o"]).drop(columns=["_g", "_o"]).reset_index(drop=True)


def error_examples(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """≤ k lỗi tự tin nhất (xác suất xa 0,5 nhất) trong mỗi nhóm khó; không lặp review giữa các nhóm."""
    err = df[~df["correct"]].assign(confidence=lambda d: (d["prob"] - 0.5).abs())
    used, parts = set(), []
    for group, sub in HARD_SUBGROUPS:
        g = err[(err[GROUP_COLS[group]] == sub) & ~err["uid"].isin(used)].nlargest(k, "confidence")
        used |= set(g["uid"])
        parts.append(g.assign(case=f"{group}={sub}"))
    cols = ["case", "uid", "label", "pred", "prob", "rating", "n_tokens", "was_truncated", "n_negation",
            "contrast_end", "sarcasm_candidate", "text"]
    return pd.concat(parts)[cols].assign(explanation="").reset_index(drop=True)


def manual_sample(df: pd.DataFrame, n: int = 50, seed: int = 42) -> pd.DataFrame:
    """50 lỗi để gán tay: ưu tiên ứng viên khen chê lẫn lộn / mỉa mai, bù bằng lỗi ngẫu nhiên."""
    err = df[~df["correct"]]
    pri = err[(err["contrast_end"] == "True") | (err["sarcasm_candidate"] == "True")]
    pick = pri.sample(min(n, len(pri)), random_state=seed)
    if len(pick) < n:
        rest = err.drop(pick.index)
        pick = pd.concat([pick, rest.sample(min(n - len(pick), len(rest)), random_state=seed)])
    cols = ["uid", "label", "pred", "prob", "rating", "n_tokens", "was_truncated", "contrast_end",
            "sarcasm_candidate", "text"]
    return pick[cols].assign(
        manual_type="",  # mixed | sarcasm | negation | truncated_conclusion | target_shift | label_noise | other
        note="",
    ).reset_index(drop=True)


# ------------------------------------------------------------------ 30 câu tự viết
def handwritten_predictions(ckpt: Path, vocab: Path, baseline_ckpt: Path | None = None) -> pd.DataFrame:
    from src.predict import SentimentPredictor

    hw = pd.read_csv(data_dir("sample") / "handwritten_30.csv")
    pred = SentimentPredictor(checkpoint_path=ckpt, vocab_path=vocab, device="cpu")
    hw["prob"] = [round(p, 4) for p in pred.predict_batch(hw["text"].tolist())]
    hw["pred"] = (hw["prob"] >= 0.5).astype(int)
    hw["correct"] = hw["pred"] == hw["label"]
    if baseline_ckpt is not None and baseline_ckpt.exists():
        import joblib

        from src.data.clean import clean_text

        bl = joblib.load(baseline_ckpt)
        clf = bl.get("model") or bl.get("classifier")
        p = clf.predict_proba(bl["vectorizer"].transform([clean_text(t) for t in hw["text"]]))[:, 1]
        hw["prob_baseline"] = np.round(p, 4)
        hw["pred_baseline"] = (p >= 0.5).astype(int)
        hw["correct_baseline"] = hw["pred_baseline"] == hw["label"]
    return hw


# ------------------------------------------------------------------ hình
def plot_by_group(table: pd.DataFrame, path: Path, model_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(GROUP_COLS)
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups), 3.8), sharey=True)
    for ax, g in zip(axes, groups):
        t = table[table["group"] == g]
        x = np.arange(len(t))
        has_bl = "acc_baseline" in t
        w = 0.4 if has_bl else 0.7
        ax.bar(x - (w / 2 if has_bl else 0), t["acc_model"], w, label=model_name)
        if has_bl:
            ax.bar(x + w / 2, t["acc_baseline"], w, label="TF-IDF + LR")
        for xi, (a, n) in enumerate(zip(t["acc_model"], t["n"])):
            ax.text(xi, 0.52, f"n={n}", ha="center", fontsize=7, rotation=90)
        ax.set_xticks(x, t["subgroup"], rotation=35, ha="right", fontsize=8)
        ax.set_title(g, fontsize=9)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy (test)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def run(results_dir: Path, out_dir: Path, plots_dir: Path, exp: str | None = None,
        ckpt_root: Path | None = None) -> dict:
    if exp is None:
        exp, seed = pick_best_exp(results_dir)
    else:
        seed = pick_seed(results_dir / exp)
    df = add_case_columns(load_predictions(results_dir / exp, seed))
    bl_path = results_dir / "baseline" / "predictions_test.csv"
    baseline = pd.read_csv(bl_path) if bl_path.exists() else None

    out_dir.mkdir(parents=True, exist_ok=True)
    table = by_group(df, baseline)
    table.round(4).to_csv(out_dir / "by_group.csv", index=False)
    error_examples(df).to_csv(out_dir / "error_examples.csv", index=False)
    manual_sample(df).to_csv(out_dir / "manual_label_50.csv", index=False)
    plot_by_group(table, plots_dir / "acc_by_group.png", f"{exp} (seed {seed})")

    summary = {
        "model": exp, "seed": seed, "n_test": len(df), "accuracy": round(float(df["correct"].mean()), 4),
        "n_errors": int((~df["correct"]).sum()),
        "false_positive": int(((df["pred"] == 1) & (df["label"] == 0)).sum()),
        "false_negative": int(((df["pred"] == 0) & (df["label"] == 1)).sum()),
        "baseline_file": _rel(bl_path) if baseline is not None else None,
        "rules": RULES,
    }

    ckpt_root = ckpt_root or experiment_dir("checkpoints")
    ckpt, vocab = ckpt_root / exp / f"seed_{seed}" / "best.pt", ckpt_root / exp / "vocab.json"
    if ckpt.exists() and vocab.exists():
        hw = handwritten_predictions(ckpt, vocab, ckpt_root / "baseline_tfidf_lr.joblib")
        hw.to_csv(out_dir / "handwritten_30_predictions.csv", index=False)
        summary["handwritten_30"] = {
            "accuracy_scored": round(float(hw.loc[hw["case_type"] != "out_of_domain", "correct"].mean()), 4),
            "by_case_type": hw.groupby("case_type")["correct"].mean().round(2).to_dict(),
        }
        if "correct_baseline" in hw:
            summary["handwritten_30"]["baseline_by_case_type"] = hw.groupby("case_type")["correct_baseline"].mean().round(2).to_dict()
    else:
        summary["handwritten_30"] = f"bỏ qua: chưa có {ckpt.name} / vocab.json trong {ckpt_root / exp}"

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Phân tích lỗi theo 7 nhóm tình huống (T2).")
    ap.add_argument("--exp", default=None, help="tên thí nghiệm; mặc định chọn theo mean val acc")
    ap.add_argument("--results-dir", type=Path, default=experiment_dir("results"))
    ap.add_argument("--checkpoints-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=experiment_dir("results", "error_analysis"))
    ap.add_argument("--plots", type=Path, default=experiment_dir("plots", "error_analysis"))
    a = ap.parse_args()
    s = run(a.results_dir, a.out, a.plots, a.exp, a.checkpoints_dir)
    print(json.dumps({k: v for k, v in s.items() if k != "rules"}, ensure_ascii=False, indent=1))
    print(pd.read_csv(a.out / "by_group.csv").to_string(index=False))


if __name__ == "__main__":
    main()
