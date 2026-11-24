"""Когда помогает контекст: вклад text/visual над none по типам отношений (K=16, n≈970).

Группировки (без переобучения, по готовым test_details_rank*.json):
  domain       — первый сегмент пути отношения Freebase (/film, /people, …);
  cardinality  — 1-1 / 1-N / N-1 / N-N по Bordes et al. (порог 1.5 на среднее число
                 хвостов на голову и голов на хвост в train);
  rel_freq     — терцили по частоте отношения в train.
В датасете хранится только последний сегмент отношения → полный путь берём как самый
частый в train с таким окончанием.

    python scripts/relation_strata.py [--out graphvis_ckpts_s1/k16/relation_strata.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

from popularity_strata import CK, CONDS, ROOT, boot_ci, load_details


def relation_info(data_dir):
    triples = []
    with open(os.path.join(data_dir, "train.txt"), encoding="utf-8") as f:
        for ln in f:
            triples.append(ln.rstrip("\n").split("\t"))
    freq = collections.Counter(r for _, r, _ in triples)
    tails, heads = collections.defaultdict(set), collections.defaultdict(set)
    for h, r, t in triples:
        tails[(r, h)].add(t)
        heads[(r, t)].add(h)
    tph, hpt = collections.defaultdict(list), collections.defaultdict(list)
    for (r, _), s in tails.items():
        tph[r].append(len(s))
    for (r, _), s in heads.items():
        hpt[r].append(len(s))
    info = {}
    for r in freq:
        a, b = np.mean(tph[r]), np.mean(hpt[r])
        card = ("1" if b < 1.5 else "N") + "-" + ("1" if a < 1.5 else "N")
        info[r] = {"freq": freq[r], "card": card, "domain": r.split("/")[1]}
    by_last = collections.defaultdict(list)
    for r in freq:
        by_last[r.rsplit("/", 1)[-1]].append(r)
    last2full = {k: max(v, key=lambda r: freq[r]) for k, v in by_last.items()}
    return info, last2full


def table(name, groups, rr, min_n):
    print(f"\n=== {name} ===")
    hdr = f"{'группа':22s} {'n':>4s} {'none':>6s}  " + "  ".join(f"{c + ' - none':>28s}" for c in CONDS if c != "none")
    print(hdr)
    out = []
    order = sorted(set(groups), key=lambda g: -np.sum(groups == g))
    for g in order:
        m = groups == g
        if m.sum() < min_n:
            continue
        line = {"group": str(g), "n": int(m.sum()), "mrr_none": float(rr["none"][m].mean())}
        cells = []
        for c in CONDS:
            if c == "none":
                continue
            mu, lo, hi = boot_ci(rr[c][m] - rr["none"][m])
            line[c] = {"mrr": float(rr[c][m].mean()), "d_vs_none": mu, "ci95": [lo, hi]}
            cells.append(f"{mu:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"{str(g):22s} {int(m.sum()):4d} {rr['none'][m].mean():6.3f}  " + "  ".join(f"{x:>28s}" for x in cells))
        out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "graphvis_scaling", "k16", "datasets", "stage2_test"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "FB15k-237"))
    ap.add_argument("--min_n", type=int, default=40, help="не показывать группы меньше")
    ap.add_argument("--out", default=os.path.join(CK, "relation_strata.json"))
    args = ap.parse_args()

    from datasets import load_from_disk
    ds = load_from_disk(args.dataset)
    det = {c: load_details(*v) for c, v in CONDS.items()}
    common = sorted(set.intersection(*(set(d) for d in det.values())))
    info, last2full = relation_info(args.data_dir)

    rows = []
    for i in common:
        full = last2full.get(ds[int(i)]["relation"])
        if full is not None:
            rows.append((i, info[full]))
    print(f"запросов: {len(common)}, с найденным отношением: {len(rows)}")
    rr = {c: np.array([1.0 / det[c][i]["rank"] for i, _ in rows]) for c in CONDS}

    dom = np.array([x["domain"] for _, x in rows])
    top = {d for d, _ in collections.Counter(dom).most_common() if np.sum(dom == d) >= args.min_n}
    dom = np.array([d if d in top else "other" for d in dom])
    card = np.array([x["card"] for _, x in rows])
    f = np.array([x["freq"] for _, x in rows])
    cuts = np.quantile(f, [1 / 3, 2 / 3])
    fq = np.array(["редкие", "средние", "частые"])[np.digitize(f, cuts, right=True)]
    fq = np.array([f"{x} отн." for x in fq])

    res = {"n": len(rows), "freq_cuts": [float(x) for x in cuts],
           "domain": table("по домену отношения", dom, rr, args.min_n),
           "cardinality": table("по кардинальности (голова-хвост)", card, rr, args.min_n),
           "rel_freq": table(f"по частоте отношения в train (границы {cuts[0]:.0f} / {cuts[1]:.0f})", fq, rr, args.min_n)}
    with open(args.out, "w", encoding="utf-8") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=1)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
