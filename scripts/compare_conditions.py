"""Сводит прогоны блока 1 (ctx_none/ctx_text/ctx_visual) в парную таблицу с
ΔMRR 95%-CI (парный bootstrap) и McNemar по Hits@k против базового условия.

Читает graphvis_ckpts/ctx_<cond>/test_details_rank_<tag>.json (поквери {idx,gold,
rank,n_cand}), парит по idx. Условия, для которых деталей ещё нет (прогон не
доехал), молча пропускает — можно запускать до готовности visual и после.

    python scripts/compare_conditions.py
    python scripts/compare_conditions.py --conditions none,text,visual --baseline none
"""

from __future__ import annotations

# Windows-консоль/редирект в файл по умолчанию cp1251 → печать 'Δ' роняла процесс
# UnicodeEncodeError ПОСЛЕ подсчёта, но ДО json.dump → результаты терялись.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import glob
import json
import math
import os

import numpy as np

_PREF = ["real", "blank", "noise", "shuffle"]   # какой image_mode брать как основной


def _tag(path: str) -> str:
    b = os.path.basename(path)[:-len(".json")]   # test_details_rank[_<tag>]
    rest = b[len("test_details_rank"):]          # "_blank" | "_real" | ""
    return rest[1:] if rest.startswith("_") else (rest or "real")


def find_details(ckpt_root: str, cond: str, kind: str = "rank"):
    d = os.path.join(ckpt_root, f"ctx_{cond}")
    if kind == "sample":
        # открытый (KGT5) прогон: test_details_sample_<label>.json, label = context_mode
        paths = glob.glob(os.path.join(d, "test_details_sample_*.json"))
        if not paths:
            return None, None
        p = sorted(paths)[0]
        return p, os.path.basename(p)[len("test_details_sample_"):-len(".json")]
    # rank: ловим и с тегом (test_details_rank_blank.json), и без (test_details_rank.json)
    paths = glob.glob(os.path.join(d, "test_details_rank*.json"))
    if not paths:
        return None, None
    paths.sort(key=lambda p: _PREF.index(_tag(p)) if _tag(p) in _PREF else 99)
    return paths[0], _tag(paths[0])


def load(path: str):
    return {r["idx"]: r["rank"] for r in json.load(open(path, encoding="utf-8"))}


def mrr(r):
    return float(np.mean(1.0 / r))


def hits(r, k):
    return 100.0 * float(np.mean(r <= k))


def mcnemar(rb, rc, k):
    a = rb <= k
    b = rc <= k
    n01 = int(np.sum(~a & b))   # base miss, cond hit  → cond выиграл
    n10 = int(np.sum(a & ~b))   # base hit, cond miss  → base выиграл
    if n01 + n10 == 0:
        return n01, n10, float("nan"), 1.0
    z = (abs(n01 - n10) - 1) / math.sqrt(n01 + n10)   # с поправкой на непрерывность
    p = math.erfc(abs(z) / math.sqrt(2))
    return n01, n10, z, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="graphvis_ckpts")
    ap.add_argument("--conditions", default="none,text,visual")
    ap.add_argument("--baseline", default="none")
    ap.add_argument("--kind", choices=["rank", "sample"], default="rank",
                    help="rank = type-constrained (test_details_rank*), "
                         "sample = открытый KGT5 (test_details_sample_*)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = ("graphvis_ckpts/compare_conditions.json" if args.kind == "rank"
                    else "graphvis_ckpts/compare_conditions_sample.json")

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(f"[protocol] {args.kind}")
    data, tags = {}, {}
    for c in conds:
        p, t = find_details(args.ckpt_root, c, args.kind)
        if p is None:
            print(f"[skip] {c}: нет test_details (прогон не готов?)")
            continue
        data[c] = load(p)
        tags[c] = t
        print(f"[ok]   {c}: image_mode={t}, n={len(data[c])}")

    report = {"conditions": {}, "pairwise": {}}
    print(f"\n=== Метрики ===\n{'cond':<8}{'n':>6}{'MRR':>9}{'H@1':>8}{'H@3':>8}{'H@10':>8}")
    for c in conds:
        if c not in data:
            continue
        r = np.array(list(data[c].values()), float)
        print(f"{c:<8}{len(r):>6}{mrr(r):>9.4f}{hits(r,1):>8.2f}{hits(r,3):>8.2f}{hits(r,10):>8.2f}")
        report["conditions"][c] = {"tag": tags[c], "n": len(r), "MRR": mrr(r),
                                   "H@1": hits(r, 1), "H@3": hits(r, 3), "H@10": hits(r, 10)}

    if args.baseline not in data:
        print(f"\n[!] базовое '{args.baseline}' недоступно — парные тесты пропущены")
    else:
        base = data[args.baseline]
        rng = np.random.RandomState(42)
        for c in conds:
            if c not in data or c == args.baseline:
                continue
            common = sorted(set(base) & set(data[c]))
            rb = np.array([base[i] for i in common], float)
            rc = np.array([data[c][i] for i in common], float)
            diff = 1.0 / rc - 1.0 / rb
            obs = float(diff.mean())
            n, B = len(common), args.bootstrap
            boot = np.empty(B)
            for b in range(B):
                boot[b] = diff[rng.randint(0, n, n)].mean()
            lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
            sig = lo > 0 or hi < 0
            print(f"\n=== {c} vs {args.baseline} (парных {n}) ===")
            print(f"  ΔMRR = {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"{'ЗНАЧИМО' if sig else 'не значимо'}")
            mc = {}
            for k in (1, 3, 10):
                n01, n10, z, p = mcnemar(rb, rc, k)
                print(f"  McNemar H@{k:<2}: {c}-win={n01:>3} {args.baseline}-win={n10:>3} "
                      f"z={z:5.2f} p={p:.4g} {'значимо' if p < 0.05 else 'н/з'}")
                mc[f"H@{k}"] = {"win_cond": n01, "win_base": n10, "z": z, "p": p}
            report["pairwise"][f"{c}_vs_{args.baseline}"] = {
                "n": n, "dMRR": obs, "CI95": [lo, hi], "significant": sig, "mcnemar": mc}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
