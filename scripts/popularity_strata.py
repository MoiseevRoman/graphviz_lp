"""Вклад контекста по популярности сущностей (без переобучения).

Гипотеза: на известных модели сущностях окрестность дублирует параметрические
знания, на редких — вклад text/visual над none больше. Запросы K=16 (n=971)
делятся на терцили по степени ГОЛОВЫ (и отдельно ответа) в FB15k-237 train;
в каждой терцили — ΔMRR условия над none, парный бутстрап по запросам.

    python scripts/popularity_strata.py [--out graphvis_ckpts_s1/k16/popularity_strata.json]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "graphvis_ckpts_s1", "k16")
CONDS = {   # имя → (папка, тег файла)
    "none": ("ctx_none", "blank"),
    "text": ("ctx_text", "blank"),
    "visual_vbf16": ("ctx_visual_vbf16", ""),
    "visual_s1fv": ("ctx_visual_s1fv", ""),
}


def load_details(folder, tag):
    name = "test_details_rank" + (f"_{tag}" if tag else "") + ".json"
    return {d["idx"]: d for d in json.load(open(os.path.join(CK, folder, name), encoding="utf-8"))}


def degree_by_name(data_dir):
    deg = collections.Counter()
    with open(os.path.join(data_dir, "train.txt"), encoding="utf-8") as f:
        for ln in f:
            h, _, t = ln.rstrip("\n").split("\t")
            deg[h] += 1
            deg[t] += 1
    by_name = collections.defaultdict(int)
    with open(os.path.join(data_dir, "FB15k_mid2name.txt"), encoding="utf-8") as f:
        for ln in f:
            mid, name = ln.rstrip("\n").split("\t", 1)
            if mid in deg:      # омонимы: берём самую популярную сущность с этим именем
                key = name.replace("_", " ").strip().lower()
                by_name[key] = max(by_name[key], deg[mid])
    return by_name


def boot_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "graphvis_scaling", "k16", "datasets", "stage2_test"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "FB15k-237"))
    ap.add_argument("--out", default=os.path.join(CK, "popularity_strata.json"))
    args = ap.parse_args()

    from datasets import load_from_disk
    ds = load_from_disk(args.dataset)
    det = {c: load_details(*v) for c, v in CONDS.items()}
    common = sorted(set.intersection(*(set(d) for d in det.values())))
    deg = degree_by_name(args.data_dir)

    rows, miss = [], 0
    for i in common:
        s = ds[int(i)]
        assert s["tail"] == det["none"][i]["gold"], f"idx {i}: несовпадение gold"
        dh = deg.get(s["head"].replace("_", " ").strip().lower())
        dt = deg.get(s["tail"].replace("_", " ").strip().lower())
        if dh is None or dt is None:
            miss += 1
            continue
        rows.append((i, dh, dt))
    print(f"запросов: {len(common)}, со степенью: {len(rows)} (без совпадения имени: {miss})")

    rr = {c: np.array([1.0 / det[c][i]["rank"] for i, _, _ in rows]) for c in CONDS}
    result = {"n": len(rows), "strata": {}}
    for key, col in (("head_degree", 1), ("tail_degree", 2)):
        d = np.array([r[col] for r in rows])
        cuts = np.quantile(d, [1 / 3, 2 / 3])
        strata = np.digitize(d, cuts, right=True)
        print(f"\n=== терцили по {key} (границы {cuts[0]:.0f} / {cuts[1]:.0f}) ===")
        print(f"{'терциль':22s} {'n':>4s} {'none':>6s}  " + "  ".join(f"{c + ' - none':>30s}" for c in CONDS if c != "none"))
        res = []
        for s_id, lab in enumerate(("редкие", "средние", "популярные")):
            m = strata == s_id
            rng_ = f"{lab} [{d[m].min()}-{d[m].max()}]"
            line = {"stratum": lab, "deg_min": int(d[m].min()), "deg_max": int(d[m].max()),
                    "n": int(m.sum()), "mrr_none": float(rr["none"][m].mean())}
            cells = []
            for c in CONDS:
                if c == "none":
                    continue
                mu, lo, hi = boot_ci(rr[c][m] - rr["none"][m])
                line[c] = {"mrr": float(rr[c][m].mean()), "d_vs_none": mu, "ci95": [lo, hi]}
                cells.append(f"{mu:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}")
            print(f"{rng_:22s} {int(m.sum()):4d} {rr['none'][m].mean():6.3f}  " + "  ".join(f"{x:>30s}" for x in cells))
            res.append(line)
        # редкие − популярные: растёт ли вклад контекста на «хвосте»
        rare, pop = strata == 0, strata == 2
        diff = {}
        for c in CONDS:
            if c == "none":
                continue
            a, b = rr[c][rare] - rr["none"][rare], rr[c][pop] - rr["none"][pop]
            rng = np.random.default_rng(1)
            bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
                  for _ in range(2000)]
            diff[c] = [float(a.mean() - b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            print(f"  Δ(редкие) − Δ(популярные), {c}: {diff[c][0]:+.3f} [{diff[c][1]:+.3f}, {diff[c][2]:+.3f}]")
        result["strata"][key] = {"cuts": [float(x) for x in cuts], "rows": res, "rare_minus_popular": diff}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
