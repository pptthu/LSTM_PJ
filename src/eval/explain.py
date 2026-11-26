"""T3 — Giải thích mô hình bằng occlusion, kiểm chứng bằng deletion test, kiểm "đường tắt".

Chạy (từ gốc repo, CPU là đủ):
    python -m src.eval.explain                       # tự chọn mô hình + seed tốt nhất theo val
    python -m src.eval.explain --exp bilstm_attention --checkpoints-dir <thư mục checkpoints>

Bốn phần:
1. Heatmap occlusion: che lần lượt từng từ (thay bằng <unk>), đo xác suất lớp dự đoán giảm bao nhiêu.
2. Deletion test (kiểm occlusion có "đúng" không): che k từ quan trọng nhất theo occlusion,
   so với che k từ ngẫu nhiên. Occlusion có ý nghĩa nếu che top-k làm độ tự tin giảm nhanh hơn hẳn.
3. Đường tắt tên riêng: che mọi tên riêng (từ viết hoa giữa câu) so với che ngẫu nhiên cùng số từ.
   Nếu che tên riêng làm đổi dự đoán nhiều hơn ngẫu nhiên → mô hình dựa vào tên diễn viên / phim.
4. Đường tắt độ dài: tương quan Spearman giữa số token và xác suất positive / độ tự tin.

Mọi thứ chạy trên val (chọn mô hình) và dữ liệu mẫu — không dùng test để quyết định gì.
Đầu ra: experiments/results/explain/, experiments/plots/explain/.
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.error_analysis import _rel, pick_best_exp, pick_seed
from src.utils.paths import data_dir, experiment_dir

KS = (1, 3, 5, 10)
N_VAL = 300          # số review val cho deletion test và kiểm đường tắt (CPU vài phút)
SEED = 42
PROPER_RE = re.compile(r"(?<=[a-z,;:)] )[A-Z][a-z]{2,}")  # từ viết hoa đứng sau chữ thường → không phải đầu câu
CHUNK = 256


# ------------------------------------------------------------------ lõi
def probs_for_ids(pred, id_lists: list[list[int]]) -> np.ndarray:
    """Xác suất positive cho nhiều chuỗi id đã mã hoá (không cắt lại)."""
    import torch

    from src.data.dataset import collate_imdb_batch

    out = []
    for i in range(0, len(id_lists), CHUNK):
        items = [{"uid": f"x{j}", "text": "", "input_ids": ids or [pred.vocab.unk_idx], "length": max(1, len(ids)),
                  "rating": 0, "n_tokens": len(ids), "n_unk": 0, "was_truncated": False}
                 for j, ids in enumerate(id_lists[i:i + CHUNK])]
        batch = collate_imdb_batch(items, pad_idx=pred.vocab.pad_idx)
        with torch.no_grad():
            p = pred.model.predict_proba(batch["input_ids"].to(pred.device), batch["lengths"]).cpu().numpy()
        out.append(np.atleast_1d(p))
    return np.concatenate(out)


def mask(ids: list[int], positions, unk: int) -> list[int]:
    pos = set(int(p) for p in positions)
    return [unk if i in pos else t for i, t in enumerate(ids)]


def occlusion(pred, text: str) -> dict:
    """Điểm của từng token theo hướng lớp dự đoán: > 0 là token ủng hộ dự đoán."""
    enc = pred._encode_one(text)
    ids, toks = enc["input_ids"], enc["tokens"]
    p = probs_for_ids(pred, [ids] + [mask(ids, [i], pred.vocab.unk_idx) for i in range(len(ids))])
    base, rest = float(p[0]), p[1:]
    label = int(base >= 0.5)
    sign = 1.0 if label == 1 else -1.0
    return {"tokens": toks, "ids": ids, "prob": base, "label": label, "scores": sign * (base - rest)}


def confidence(prob: np.ndarray | float, label: int) -> np.ndarray | float:
    return prob if label == 1 else 1 - prob


def deletion_test(pred, texts: list[str], ks=KS, seed: int = SEED) -> pd.DataFrame:
    """Độ tự tin vào lớp dự đoán giảm bao nhiêu khi che top-k (occlusion) so với k ngẫu nhiên."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in texts:
        o = occlusion(pred, t)
        n = len(o["ids"])
        order = np.argsort(-o["scores"])
        variants, meta = [], []
        for k in ks:
            if k > n:
                continue
            variants.append(mask(o["ids"], order[:k], pred.vocab.unk_idx))
            meta.append((k, "top_occlusion"))
            variants.append(mask(o["ids"], rng.choice(n, k, replace=False), pred.vocab.unk_idx))
            meta.append((k, "random"))
        if not variants:
            continue
        p = probs_for_ids(pred, variants)
        c0 = confidence(o["prob"], o["label"])
        for (k, how), pk in zip(meta, p):
            rows.append({"k": k, "how": how, "conf_drop": c0 - confidence(float(pk), o["label"]),
                         "flipped": int(pk >= 0.5) != o["label"]})
    df = pd.DataFrame(rows)
    return (df.groupby(["k", "how"]).agg(n=("conf_drop", "size"), mean_conf_drop=("conf_drop", "mean"),
                                         flip_rate=("flipped", "mean"))
              .reset_index().round(4))


def proper_nouns(text: str) -> set[str]:
    return {w.lower() for w in PROPER_RE.findall(text)}


def shortcut_names(pred, texts: list[str], seed: int = SEED) -> dict:
    """Che tên riêng vs che ngẫu nhiên cùng số token."""
    rng = np.random.default_rng(seed)
    base_ids, name_ids, rand_ids, n_masked = [], [], [], []
    for t in texts:
        enc = pred._encode_one(t)
        ids, toks = enc["input_ids"], enc["tokens"]
        names = proper_nouns(t)
        pos = [i for i, w in enumerate(toks) if w in names]
        if not pos:
            continue
        base_ids.append(ids)
        name_ids.append(mask(ids, pos, pred.vocab.unk_idx))
        rand_ids.append(mask(ids, rng.choice(len(ids), len(pos), replace=False), pred.vocab.unk_idx))
        n_masked.append(len(pos))
    if not base_ids:
        return {"n_reviews": 0}
    pb, pn, pr = (probs_for_ids(pred, x) for x in (base_ids, name_ids, rand_ids))
    flip = lambda a, b: float(np.mean((a >= 0.5) != (b >= 0.5)))  # noqa: E731
    return {
        "n_reviews": len(base_ids), "mean_tokens_masked": round(float(np.mean(n_masked)), 2),
        "names_mean_abs_dprob": round(float(np.mean(np.abs(pb - pn))), 4),
        "random_mean_abs_dprob": round(float(np.mean(np.abs(pb - pr))), 4),
        "names_flip_rate": round(flip(pb, pn), 4), "random_flip_rate": round(flip(pb, pr), 4),
    }


def shortcut_length(pred, texts: list[str]) -> dict:
    from scipy.stats import spearmanr

    probs = np.array(pred.predict_batch(texts))
    n_tok = np.array([pred._encode_one(t)["n_tokens"] for t in texts])
    r_p, p_p = spearmanr(n_tok, probs)
    r_c, p_c = spearmanr(n_tok, np.abs(probs - 0.5))
    return {"n": len(texts), "spearman_len_vs_prob_pos": round(float(r_p), 4), "p_value_prob": float(p_p),
            "spearman_len_vs_confidence": round(float(r_c), 4), "p_value_conf": float(p_c)}


def top_tokens(occs: list[dict], min_count: int = 3, k: int = 20) -> pd.DataFrame:
    """Token ảnh hưởng mạnh nhất trung bình (điểm quy về hướng positive)."""
    rows = []
    for o in occs:
        sign = 1.0 if o["label"] == 1 else -1.0
        for w, s in zip(o["tokens"], o["scores"]):
            rows.append((w, sign * float(s)))
    df = pd.DataFrame(rows, columns=["token", "score_to_positive"])
    g = df.groupby("token")["score_to_positive"].agg(["mean", "size"]).query("size >= @min_count")
    pos = g.nlargest(k, "mean").assign(direction="positive")
    neg = g.nsmallest(k, "mean").assign(direction="negative")
    return pd.concat([pos, neg]).rename(columns={"mean": "mean_score", "size": "count"}).reset_index().round(4)


# ------------------------------------------------------------------ hình
def plot_heatmap(o: dict, title: str, path: Path, max_tokens: int = 220, width: int = 140) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    toks, sc = o["tokens"][:max_tokens], np.asarray(o["scores"][:max_tokens])
    lim = max(float(np.abs(sc).max()), 1e-6)
    cmap = plt.get_cmap("RdBu" if o["label"] == 1 else "RdBu_r")  # xanh = ủng hộ positive, đỏ = negative
    lines, cur, n = [], [], 0
    for i, w in enumerate(toks):
        if n + len(w) + 1 > width and cur:
            lines.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += len(w) + 1
    if cur:
        lines.append(cur)
    fig, ax = plt.subplots(figsize=(12, 0.32 * len(lines) + 0.9))
    ax.axis("off")
    for r, idx in enumerate(lines):
        x = 0.0
        for i in idx:
            c = cmap(0.5 + 0.5 * sc[i] / lim)
            ax.text(x, -r, toks[i], fontsize=9, family="monospace", va="top",
                    bbox={"facecolor": c, "edgecolor": "none", "pad": 1.2})
            x += (len(toks[i]) + 1) / width
    ax.set_xlim(0, 1)
    ax.set_ylim(-len(lines), 0.6)
    more = f" (hiện {max_tokens}/{len(o['tokens'])} token)" if len(o["tokens"]) > max_tokens else ""
    ax.set_title("\n".join(textwrap.wrap(title + more, 120)), fontsize=9, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_deletion(table: pd.DataFrame, path: Path, model_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for how, g in table.groupby("how"):
        ax.plot(g["k"], g["mean_conf_drop"], marker="o", label=how)
    ax.set(xlabel="số từ bị che (k)", ylabel="độ tự tin giảm (TB)", title=f"Deletion test — {model_name}, val")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ chạy
def pick_examples(pred, sample: pd.DataFrame, hand: pd.DataFrame) -> list[tuple[str, str]]:
    """2 review mẫu đúng + 2 sai (sample_100) + 1 câu tự viết mỗi loại phủ định / mỉa mai / lẫn lộn."""
    p = np.array(pred.predict_batch(sample["text"].tolist()))
    s = sample.assign(prob=p, pred=(p >= 0.5).astype(int), conf=np.abs(p - 0.5))
    right = s[s["pred"] == s["label"]].nlargest(2, "conf")
    wrong = s[s["pred"] != s["label"]].nlargest(2, "conf")
    out = [(f"sample_right_{r.uid}", r.text, f"ĐÚNG · nhãn {r.label}") for r in right.itertuples()]
    out += [(f"sample_wrong_{r.uid}", r.text, f"SAI · nhãn {r.label}") for r in wrong.itertuples()]
    for ct in ("negation", "sarcasm", "mixed"):
        h = hand[hand["case_type"] == ct].head(1)
        out += [(f"hand_{ct}_{i}", r.text, f"tự viết · {ct} · nhãn {r.label}") for i, r in zip(h.index, h.itertuples())]
    return out


def run(results_dir: Path, ckpt_root: Path, out_dir: Path, plots_dir: Path, exp: str | None = None,
        n_val: int = N_VAL) -> dict:
    from src.predict import SentimentPredictor

    exp, seed = (exp, pick_seed(results_dir / exp)) if exp else pick_best_exp(results_dir)
    ckpt = ckpt_root / exp / f"seed_{seed}" / "best.pt"
    vocab = ckpt_root / exp / "vocab.json"
    pred = SentimentPredictor(checkpoint_path=ckpt, vocab_path=vocab, device="cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    val = pd.read_parquet(data_dir("processed") / "reviews.parquet").query("split == 'val'")
    val = val.sample(min(n_val, len(val)), random_state=SEED)
    texts = val["text_clean"].tolist()

    dele = deletion_test(pred, texts)
    dele.to_csv(out_dir / "deletion_test.csv", index=False)
    plot_deletion(dele, plots_dir / "deletion_test.png", exp)

    occs = [occlusion(pred, t) for t in texts]
    top_tokens(occs).to_csv(out_dir / "top_tokens_val.csv", index=False)

    names = shortcut_names(pred, texts)
    length = shortcut_length(pred, texts)

    sample = pd.read_csv(data_dir("sample") / "sample_100.csv")
    hand = pd.read_csv(data_dir("sample") / "handwritten_30.csv")
    shown = []
    for key, text, tag in pick_examples(pred, sample, hand):
        o = occlusion(pred, text)
        title = f"{exp} seed {seed} · {tag} · dự đoán {o['label']} (p_pos = {o['prob']:.3f}) · xanh = đẩy về positive, đỏ = negative"
        plot_heatmap(o, title, plots_dir / f"occlusion_{key}.png")
        top = np.argsort(-np.abs(o["scores"]))[:8]
        shown.append({"key": key, "tag": tag, "prob_pos": round(o["prob"], 4),
                      "top_tokens": [(o["tokens"][i], round(float(o["scores"][i]), 4)) for i in top]})

    summary = {"model": exp, "seed": seed, "checkpoint": _rel(ckpt), "n_val_reviews": len(texts),
               "deletion_test": dele.to_dict("records"), "shortcut_names": names, "shortcut_length": length,
               "examples": shown,
               "rules": {"occlusion": "thay token bằng <unk>; điểm = độ tự tin vào lớp dự đoán giảm",
                         "proper_noun": PROPER_RE.pattern, "ks": list(KS), "seed": SEED}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", default=None)
    ap.add_argument("--results-dir", type=Path, default=experiment_dir("results"))
    ap.add_argument("--checkpoints-dir", type=Path, default=experiment_dir("checkpoints"))
    ap.add_argument("--out", type=Path, default=experiment_dir("results", "explain"))
    ap.add_argument("--plots", type=Path, default=experiment_dir("plots", "explain"))
    ap.add_argument("--n-val", type=int, default=N_VAL)
    a = ap.parse_args()
    s = run(a.results_dir, a.checkpoints_dir, a.out, a.plots, exp=a.exp, n_val=a.n_val)
    print(json.dumps({k: s[k] for k in ("model", "seed", "shortcut_names", "shortcut_length")},
                     indent=1, ensure_ascii=False))
    print(pd.DataFrame(s["deletion_test"]).to_string(index=False))


if __name__ == "__main__":
    main()
