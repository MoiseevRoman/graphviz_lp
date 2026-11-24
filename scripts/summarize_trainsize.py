#!/usr/bin/env python3
"""Свести train-size ablation в кривую MRR(n) по модальностям.

Читает отчёты viability_gate из дерева <root>:
    <root>/ctx_<cond>/report.json          -> n = --final_n (финальный адаптер, 16k)
    <root>/ctx_<cond>/n<NNNNN>/report.json -> n = NNNNN (промежуточный чекпойнт)

Печатает таблицу n × условия по MRR (+ дельту text-visual) и пишет CSV.

Диагностика вопроса «не мало ли train_n»:
  - если у visual MRR ещё заметно растёт к n=16k, а у text уже вышел на плато
    -> visual недообучен, сравнение при малом n нечестное (нужно ещё больше n);
  - если обе кривые выположились и text держится выше -> вывод «text >= visual»
    робастен к бюджету обучения, а не артефакт малого n.

Пример:
    python scripts/summarize_trainsize.py --root graphvis_ckpts_trainsize/k16 \
        --final_n 16000 --out graphvis_ckpts_trainsize/k16/trainsize_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re


def _eval_block(report_path: str):
    """Вернуть dict eval-метрик из report.json или None."""
    try:
        with open(report_path, encoding="utf-8") as f:
            rep = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] не прочитан {report_path}: {exc}")
        return None
    ev = rep.get("eval") or {}
    return ev if ev.get("MRR") is not None else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="дерево train-size ablation (…/graphvis_ckpts_trainsize/k16)")
    ap.add_argument("--final_n", type=int, default=16000,
                    help="train_n финального адаптера (его report.json = верхняя точка)")
    ap.add_argument("--conditions", default="text,visual")
    ap.add_argument("--out", default="", help="куда писать CSV (пусто = не писать)")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    # data[cond][n] = eval-dict
    data = {c: {} for c in conds}
    for c in conds:
        base = os.path.join(args.root, f"ctx_{c}")
        fin = os.path.join(base, "report.json")
        if os.path.isfile(fin):
            ev = _eval_block(fin)
            if ev:
                data[c][args.final_n] = ev
        for d in sorted(glob.glob(os.path.join(base, "n*"))):
            m = re.search(r"n(\d+)$", os.path.basename(d))
            if not m:
                continue
            ev = _eval_block(os.path.join(d, "report.json"))
            if ev:
                data[c][int(m.group(1))] = ev

    ns = sorted({n for c in conds for n in data[c]})
    if not ns:
        print(f"[!] в {args.root} не найдено ни одного report.json с eval.MRR")
        return

    both = "text" in conds and "visual" in conds
    print()
    print(f"Train-size ablation ({os.path.basename(args.root)}) — MRR по n (общий rank-eval)")
    hdr = "  n       " + "".join(f"{c:>10}" for c in conds) + ("   gap(t-v)" if both else "")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for n in ns:
        line = f"  {n:<7}"
        vals = {}
        for c in conds:
            ev = data[c].get(n)
            mrr = ev.get("MRR") if ev else None
            vals[c] = mrr
            line += f"{mrr:>10.4f}" if isinstance(mrr, (int, float)) else f"{'--':>10}"
        if both and isinstance(vals.get("text"), (int, float)) and isinstance(vals.get("visual"), (int, float)):
            line += f"{vals['text'] - vals['visual']:>+11.4f}"
        print(line)

        row = {"n": n}
        for c in conds:
            ev = data[c].get(n) or {}
            row[f"{c}_MRR"] = ev.get("MRR")
            row[f"{c}_H1"] = ev.get("Hits@1")
            row[f"{c}_H3"] = ev.get("Hits@3")
            row[f"{c}_H10"] = ev.get("Hits@10")
            row[f"{c}_n_eval"] = ev.get("n_evaluated")
        rows.append(row)

    print()
    for c in conds:
        pts = [(n, data[c][n]["MRR"]) for n in ns if data[c].get(n)]
        if len(pts) >= 2:
            (n0, m0), (n1, m1) = pts[0], pts[-1]
            print(f"  {c:<7}: MRR {m0:.4f}@n={n0} -> {m1:.4f}@n={n1}  (delta {m1 - m0:+.4f})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cols = ["n"] + [f"{c}_{k}" for c in conds
                        for k in ("MRR", "H1", "H3", "H10", "n_eval")]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nCSV -> {args.out}")


if __name__ == "__main__":
    main()
