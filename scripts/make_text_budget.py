"""Текст окрестности, ужатый до токенного бюджета картинки (iso-token для звезды).

Берёт датасет rerender_dataset (graphvis_scaling/k<K>/datasets) и пишет копию,
где neighbors_text обрезан ПО СТРОКАМ (целыми рёбрами, в порядке hybrid-ранжирования,
т.е. отбрасываются наименее релевантные соседи) до --budget токенов Qwen.
Бюджет = число визуальных токенов картинки того же K: 560 px → 400, 728 px → 676.
Остальные колонки не меняются → условие text на этой копии = «текст за ту же цену».

    python scripts/make_text_budget.py --src graphvis_scaling/k64/datasets \
        --out graphvis_scaling/k64_textiso/datasets --budget 676
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def truncate(text: str, budget: int, count) -> tuple[str, int, int]:
    lines = [x for x in text.splitlines() if x.strip()]
    kept = []
    for ln in lines:
        if count("\n".join(kept + [ln])) > budget:
            break
        kept.append(ln)
    return "\n".join(kept), len(kept), len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, required=True, help="токенов (= визуальных токенов картинки)")
    ap.add_argument("--splits", default="stage2_train,stage2_test")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def count(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    meta = {"src": args.src, "budget": args.budget, "splits": {}}
    for split in args.splits.split(","):
        ds = load_from_disk(os.path.join(args.src, split))
        stats = []

        def _cut(row):
            t, k, n = truncate(row["neighbors_text"], args.budget, count)
            stats.append((k, n, count(row["neighbors_text"])))
            return {"neighbors_text": t}

        out = ds.map(_cut, desc=f"budget {args.budget} [{split}]", load_from_cache_file=False)
        out.save_to_disk(os.path.join(args.out, split))
        k, n, full = (np.array(x) for x in zip(*stats))
        meta["splits"][split] = {"n": len(out), "edges_kept_mean": round(float(k.mean()), 1),
                                 "edges_total_mean": round(float(n.mean()), 1),
                                 "full_tokens_mean": round(float(full.mean()), 1),
                                 "share_truncated": round(float((k < n).mean()), 3)}
        print(split, meta["splits"][split], flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
