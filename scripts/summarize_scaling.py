#!/usr/bin/env python3
"""Сводка + кривая масштабирования по дереву чекпойнтов scaling.

Читает <root>/k{K}/ctx_{cond}/report.json (поле eval) и строит:
  - таблицу в stdout,
  - CSV,
  - фигуру MRR / Hits@10 / tokens vs K.

Работает и по iso-тесту (graphvis_ckpts_scaling), и по общему тесту
(graphvis_ckpts_scaling_common). Пример:
    python scripts/summarize_scaling.py --root graphvis_ckpts_scaling_common \
        --out_prefix tmp/common
"""
import argparse
import glob
import json
import os

_ORDER = {"text": 0, "text_adjlist": 1, "text_matrix": 2, "text_feat": 3,
          "visual": 4, "visual_aug": 5, "both": 6, "both_feat": 7}
_COL = {"text": "#1f77b4", "text_adjlist": "#17becf", "text_matrix": "#9467bd",
        "text_feat": "#2ca02c", "visual": "#d62728", "visual_aug": "#ff7f0e",
        "both": "#8c564b", "both_feat": "#e377c2"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="дерево чекпойнтов (…/k{K}/ctx_{cond})")
    ap.add_argument("--out_prefix", default="tmp/scaling",
                    help="префикс для .csv и .png")
    args = ap.parse_args()

    rows = []
    for rep in sorted(glob.glob(os.path.join(args.root, "k*", "ctx_*", "report.json"))):
        d = json.load(open(rep))
        parts = rep.replace("\\", "/").split("/")
        k = int([p for p in parts if p.startswith("k") and p[1:].isdigit()][-1][1:])
        cond = [p for p in parts if p.startswith("ctx_")][-1].replace("ctx_", "")
        e = d.get("eval", {})
        rows.append(dict(k=k, cond=cond, MRR=e.get("MRR"), H1=e.get("Hits@1"),
                         H3=e.get("Hits@3"), H10=e.get("Hits@10"),
                         tot=e.get("mean_total_tokens"), txt=e.get("mean_text_tokens"),
                         vis=e.get("mean_visual_tokens"), nev=e.get("n_evaluated"),
                         trTok=(d.get("train", {}) or {}).get("train_tokens_total")))
    if not rows:
        print(f"Нет report.json в {args.root}")
        return
    rows.sort(key=lambda r: (r["k"], _ORDER.get(r["cond"], 9)))

    print(f"{'K':>2} {'cond':<10} {'MRR':>6} {'H@1':>5} {'H@3':>5} {'H@10':>5} "
          f"{'tot':>6} {'txt':>4} {'vis':>4} {'nEval':>6} {'trainTok':>9}")
    for r in rows:
        print(f"{r['k']:>2} {r['cond']:<10} {r['MRR']:.3f} {r['H1']:5.1f} {r['H3']:5.1f} "
              f"{r['H10']:5.1f} {r['tot']:6.0f} {r['txt']:4.0f} {r['vis']:4.0f} "
              f"{r['nev']:6} {(r['trTok'] or 0) / 1e6:8.2f}M")

    csv_path = args.out_prefix + "_results.csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "condition", "MRR", "Hits@1", "Hits@3", "Hits@10",
                    "tot_tokens", "text_tokens", "visual_tokens", "n_eval", "train_tokens_M"])
        for r in rows:
            w.writerow([r["k"], r["cond"], f"{r['MRR']:.4f}", f"{r['H1']:.2f}",
                        f"{r['H3']:.2f}", f"{r['H10']:.2f}", f"{r['tot']:.0f}",
                        f"{r['txt']:.0f}", f"{r['vis']:.0f}", r["nev"],
                        f"{(r['trTok'] or 0) / 1e6:.2f}"])
    print("saved", csv_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ks = sorted({r["k"] for r in rows})
    conds = sorted({r["cond"] for r in rows}, key=lambda c: _ORDER.get(c, 9))

    def series(c, f):
        out = []
        for k in Ks:
            m = [r[f] for r in rows if r["k"] == k and r["cond"] == c]
            out.append(m[0] if m else None)
        return out

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for c in conds:
        col = _COL.get(c)
        ax[0].plot(Ks, series(c, "MRR"), marker="o", color=col, label=c)
        ax[1].plot(Ks, series(c, "H10"), marker="o", color=col, label=c)
        ax[2].plot(Ks, series(c, "tot"), marker="s", color=col, label=c)
    ax[0].set_title("MRR vs #objects (K)"); ax[0].set_ylabel("MRR")
    ax[1].set_title("Hits@10 vs K"); ax[1].set_ylabel("Hits@10 (%)")
    ax[2].set_title("Input tokens/query vs K"); ax[2].set_ylabel("mean total tokens")
    for a in ax:
        a.set_xlabel("K (neighbor budget)"); a.set_xticks(Ks); a.grid(alpha=.3); a.legend()
    fig_path = args.out_prefix + "_curve.png"
    plt.tight_layout(); plt.savefig(fig_path, dpi=130)
    print("saved", fig_path)


if __name__ == "__main__":
    main()
