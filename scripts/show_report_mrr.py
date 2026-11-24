#!/usr/bin/env python3
"""Печать loss_last + MRR из одного или нескольких report.json (viability_gate).

Используется в bat-обёртках как читаемая сводка без кавычек-ада ``python -c``.
Дивёрдж виден сразу: loss_last ~1.5 = ок, 1e5+ = разошлось.

    python scripts/show_report_mrr.py a/report.json b/report.json
"""
from __future__ import annotations

import json
import math
import os
import sys


def one(path: str) -> None:
    tag = os.path.basename(os.path.dirname(path)) or path
    if not os.path.isfile(path):
        print(f"  {tag:<24} report not found")
        return
    try:
        r = json.load(open(path, encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  {tag:<24} unreadable ({exc.__class__.__name__})")
        return
    tr = r.get("train") or {}
    ev = r.get("eval") or {}
    loss = tr.get("loss_last")
    mrr = ev.get("MRR")
    h1 = ev.get("Hits@1")
    loss_s = f"{loss:.4g}" if isinstance(loss, (int, float)) else str(loss)
    mrr_s = f"{mrr:.4f}" if isinstance(mrr, (int, float)) else str(mrr)
    h1_s = f"{h1:.2f}" if isinstance(h1, (int, float)) else str(h1)
    flag = ""
    if isinstance(loss, (int, float)) and not math.isnan(loss):
        flag = "  <-- DIVERGED" if loss > 100 else "  ok"
    elif isinstance(loss, float) and math.isnan(loss):
        flag = "  (eval-only: no train loss)"
    print(f"  {tag:<24} loss_last={loss_s:<12} MRR={mrr_s}  Hits@1={h1_s}{flag}")


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print("usage: show_report_mrr.py <report.json> [report.json ...]")
        return
    for p in paths:
        one(p)


if __name__ == "__main__":
    main()
