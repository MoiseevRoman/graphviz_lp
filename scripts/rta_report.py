"""Сводка по «прочитай, потом ответь» (RTA) и контрфактическому LP-тесту (drop300).

1) Качество чтения RTA на основном тесте (K=16, 971 запрос): точность/полнота
   выписанных рёбер против нарисованных; MRR по терцилям полноты; real против blank
   (на пустой картинке «чтение» = вспоминание соседей из параметрической памяти).
2) Контрфактика LP: те же 297 запросов, у ~30 % соседей рёбра убраны и с картинки,
   и из текста. Для text / visual_vbf16 / visual_rta: ΔMRR (drop − orig) с ДИ,
   доля запросов с изменившимся рангом, средний |Δ log p(gold)|. Для RTA ещё:
   упоминаются ли удалённые соседи в чтении (читает или вспоминает).

    python scripts/rta_report.py
"""

from __future__ import annotations

import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):   # Windows cp1251: печать «→»/«Δ» роняла скрипт
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
import rta_target as rta   # noqa: E402

RTA_DIR = os.path.join(_ROOT, "graphvis_ckpts_rta", "k16")
CF_DIR = os.path.join(_ROOT, "graphvis_ckpts_cf", "k16")
CF_DATA = os.path.join(_ROOT, "graphvis_cf", "k16", "datasets")
CF_CONDS = {"text": "blank", "visual_vbf16": "", "visual_rta": ""}


def _details(folder, tag=""):
    p = os.path.join(folder, "test_details_rank" + (f"_{tag}" if tag else "") + ".json")
    if not os.path.exists(p):
        return None
    return {d["idx"]: d for d in json.load(open(p, encoding="utf-8"))}


def _boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _ents(pairs):
    return {rta._n(e) for _, e, _ in pairs}


def reading_quality(det, ds):
    rows = []
    for i, d in det.items():
        r = ds[int(i)]
        gold = {(rta._n(a), rta._n(b), c) for a, b, c in
                rta.visible_pairs(r["head"], r["relation"], r["neighbors_text"])}
        got = rta.parse_reading(d.get("reading", ""))
        ge, oe = {g[1] for g in gold}, {g[1] for g in got}
        rows.append({"rr": 1 / d["rank"],
                     "p": len(got & gold) / max(1, len(got)), "r": len(got & gold) / max(1, len(gold)),
                     "ent_p": len(oe & ge) / max(1, len(oe)), "ent_r": len(oe & ge) / max(1, len(ge)),
                     "n_read": len(got), "n_gold": len(gold)})
    return rows


def part1(out):
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(_ROOT, "graphvis_scaling", "k16", "datasets", "stage2_test"))
    print("=== 1. RTA: качество чтения на основном тесте ===")
    for tag in ("", "blank"):
        det = _details(os.path.join(RTA_DIR, "ctx_visual_rta"), tag)
        if det is None:
            print(f"  [{tag or 'real'}] нет деталей — пропуск")
            continue
        rows = reading_quality(det, ds)
        m = {k: float(np.mean([x[k] for x in rows])) for k in ("rr", "p", "r", "ent_p", "ent_r", "n_read", "n_gold")}
        print(f"  [{tag or 'real':5s}] n={len(rows)} MRR={m['rr']:.3f} | рёбра P/R {m['p']:.3f}/{m['r']:.3f}"
              f" | соседи P/R {m['ent_p']:.3f}/{m['ent_r']:.3f} | выписано {m['n_read']:.1f} из {m['n_gold']:.1f}")
        rec = np.array([x["r"] for x in rows])
        rr = np.array([x["rr"] for x in rows])
        cuts = np.quantile(rec, [1 / 3, 2 / 3])
        st = np.digitize(rec, cuts, right=True)
        terc = [{"recall_range": [float(rec[st == s].min()), float(rec[st == s].max())],
                 "n": int((st == s).sum()), "mrr": float(rr[st == s].mean())} for s in range(3) if (st == s).any()]
        print("          MRR по терцилям полноты чтения: " +
              ", ".join(f"[{t['recall_range'][0]:.2f}–{t['recall_range'][1]:.2f}] {t['mrr']:.3f} (n={t['n']})" for t in terc))
        out[f"reading_{tag or 'real'}"] = {**m, "n": len(rows), "by_recall_tercile": terc}


def part2(out):
    from datasets import load_from_disk
    from visualizer import format_label
    print("\n=== 2. Контрфактика LP: ~30 % соседей убраны и с картинки, и из текста ===")
    try:
        dso = load_from_disk(os.path.join(CF_DATA, "orig300", "stage2_test"))
        dsd = load_from_disk(os.path.join(CF_DATA, "drop300", "stage2_test"))
    except Exception:
        print("  нет graphvis_cf датасетов — пропуск")
        return
    res = {}
    for c, tag in CF_CONDS.items():
        o = _details(os.path.join(CF_DIR, "orig300", f"ctx_{c}"), tag)
        d = _details(os.path.join(CF_DIR, "drop300", f"ctx_{c}"), tag)
        if o is None or d is None:
            print(f"  {c:13s} нет деталей — пропуск")
            continue
        ids = sorted(set(o) & set(d))
        ro = np.array([1 / o[i]["rank"] for i in ids])
        rd = np.array([1 / d[i]["rank"] for i in ids])
        mu, lo, hi = _boot(rd - ro)
        chg = float(np.mean([o[i]["rank"] != d[i]["rank"] for i in ids]))
        gs = [abs(d[i]["gold_score"] - o[i]["gold_score"]) for i in ids if "gold_score" in o[i] and "gold_score" in d[i]]
        line = {"n": len(ids), "mrr_orig": float(ro.mean()), "mrr_drop": float(rd.mean()),
                "d_mrr": mu, "ci95": [lo, hi], "rank_changed_%": round(100 * chg, 1),
                "mean_abs_d_gold_logp": float(np.mean(gs)) if gs else None}
        print(f"  {c:13s} n={len(ids)} MRR orig {ro.mean():.3f} → drop {rd.mean():.3f}  Δ={mu:+.3f} [{lo:+.3f},{hi:+.3f}]"
              f"{'*' if lo > 0 or hi < 0 else ' '}  ранг изменился {100 * chg:.0f}%"
              + (f"  |Δlog p(gold)|={np.mean(gs):.3f}" if gs else ""))
        if c == "visual_rta":
            ment_o, ment_d, kept_o, kept_d = [], [], [], []
            for i in ids:
                ro_ = dso[int(i)]
                dropped = {rta._n(format_label(x, 30)) for x in json.loads(dsd[int(i)]["dropped"])}
                kept = _ents(rta.visible_pairs(ro_["head"], ro_["relation"], ro_["neighbors_text"])) - dropped
                eo = {e for _, e, _ in rta.parse_reading(o[i].get("reading", ""))}
                ed = {e for _, e, _ in rta.parse_reading(d[i].get("reading", ""))}
                if dropped:
                    ment_o.append(len(eo & dropped) / len(dropped))
                    ment_d.append(len(ed & dropped) / len(dropped))
                if kept:
                    kept_o.append(len(eo & kept) / len(kept))
                    kept_d.append(len(ed & kept) / len(kept))
            line["reading"] = {"dropped_mentioned_orig": float(np.mean(ment_o)),
                               "dropped_mentioned_drop": float(np.mean(ment_d)),
                               "kept_mentioned_orig": float(np.mean(kept_o)),
                               "kept_mentioned_drop": float(np.mean(kept_d))}
            print(f"  {'':13s} чтение: удалённые соседи упомянуты {np.mean(ment_o):.2f} (orig) → {np.mean(ment_d):.2f} (drop);"
                  f" оставшиеся {np.mean(kept_o):.2f} → {np.mean(kept_d):.2f}")
        res[c] = line
    out["counterfactual_lp"] = res


def main():
    out = {}
    part1(out)
    part2(out)
    os.makedirs(RTA_DIR, exist_ok=True)
    p = os.path.join(RTA_DIR, "rta_report.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n→ {os.path.relpath(p, _ROOT)}")


if __name__ == "__main__":
    main()
