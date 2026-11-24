"""Собрать датасет candidate-aware подграфов с путями (path_retriever) в формате
конвейера (HF arrow: stage2_train / stage2_test) — без Neo4j.

Тройки берутся из уже собранного датасета основной статьи (по умолчанию
graphvis_scaling/k4/datasets): те же 4000 train и те же 4000 test в том же порядке,
поэтому seed-выборка eval_n=1000 в viability_gate даёт ТЕ ЖЕ 971 запрос общего теста.

Колонки = базовые (image_path, prompt, answer, task_type, query_id, head, relation,
tail) + новые:
    paths_text  — список рёбер подграфа (matched information с картинкой);
    paths_cn    — число 2-хоп путей head→кандидат по ПОЛНОМУ графу (признаковое
                  условие, аналог text_feat: несёт больше, чем картинка с ≤M путями);
    paths_text_iso — тот же edge-list, ужатый до --text_budget токенов (iso-token с
                  картинкой 560×560 = 400 визуальных токенов; см. encode_paths_budget);
                  пишется только при --text_budget > 0;
    gold_shown  — попал ли золотой хвост в показанные кандидаты (для анализа, в
                  промпт не идёт).
Режимы подачи (viability_gate --context_mode): text_cn / text_paths / visual_paths /
both_paths (+ none для базы).

Контекст по сплитам: train-запросы видят только train (и без ВСЕХ прямых рёбер
head–gold, см. PathSubgraphConfig.drop_gold_direct), test — train+valid.

    python scripts/build_path_dataset.py --splits stage2_train --workers 6
    python scripts/build_path_dataset.py --splits stage2_test --limit 2000 --workers 6
    (viability_gate читает 2*eval_n = 2000 первых тест-строк → общий тест целиком внутри)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from multiprocessing import Pool

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pyarrow as pa

from path_retriever import (PathGraph, PathSubgraphConfig, encode_cn,
                            encode_paths, encode_paths_budget, render)

_STR_COLS = ["image_path", "prompt", "answer", "task_type", "query_id", "head",
             "relation", "tail", "paths_text", "paths_cn"]
_SPLIT_GRAPH = {"stage2_train": "train", "stage2_test": "test"}

_G: dict = {}       # split → PathGraph (строится в каждом воркере: Windows не умеет fork)
_ARGS = None
_COUNT = None       # str → число токенов Qwen (нужен для бюджетов)


def load_counter(args):
    """Токенизатор Qwen: tokenizer.json (быстро, без transformers) или HF-кэш модели."""
    if os.path.exists(args.tokenizer):
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(args.tokenizer)
        return lambda s: len(t.encode(s).ids)
    try:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(args.hf_tokenizer)
        return lambda s: len(t(s, add_special_tokens=False)["input_ids"])
    except Exception as e:  # noqa: BLE001
        if args.text_budget or args.full_budget:
            raise SystemExit(f"Нет токенизатора ({args.tokenizer} / {args.hf_tokenizer}): {e}")
        return None


def _init(args, split: str) -> None:
    global _ARGS, _COUNT
    _ARGS = args
    _COUNT = load_counter(args)
    _G[split] = PathGraph(os.path.join(_ROOT, args.data_dir), split=_SPLIT_GRAPH[split])


def _visual_prompt(head: str, rel: str) -> str:
    h = head.replace("_", " ")
    return ("<image>\n"
            f"The image shows {h} (left), candidate answers (right) and 2-hop paths "
            f"connecting them.\n\nQuery: ({h}, {rel.replace('_', ' ')}, ?)\n"
            "Predict the missing tail entity. Answer with the entity name only.")


def _work(job):
    split, row = job
    a = _ARGS
    cfg = PathSubgraphConfig(n_show=a.n_show, max_paths_per_cand=a.max_paths,
                             preselect=a.preselect, cand_cap=a.cand_cap,
                             label_edges=a.label_edges,
                             drop_gold_direct=(split == "stage2_train"))
    g = _G[split]
    sg = g.build(row["head"], row["relation"], row["tail"], cfg)
    rel_img = f"{a.out_root_rel}/graphs/lp_{row['query_id']}.png"
    full = os.path.join(_ROOT, rel_img)
    if not os.path.exists(full):
        render(sg, cfg, full, target_size=a.target_size)
    if a.full_budget:
        paths_text = encode_paths_budget(sg, cfg, _COUNT, a.full_budget)[0]
    else:
        paths_text = encode_paths(sg, cfg)
    out = {
        "image_path": rel_img,
        "prompt": _visual_prompt(row["head"], row["relation"]),
        "answer": row["answer"],
        "task_type": "link_prediction",
        "query_id": row["query_id"],
        "head": row["head"],
        "relation": row["relation"],
        "tail": row["tail"],
        "paths_text": paths_text,
        "paths_cn": encode_cn(sg, full=True),
        "gold_shown": row["tail"] in sg.shown,
        "_n_inter": len(sg.inter),
        "_n_edges": len(sg.edges),
    }
    if a.text_budget:
        out["paths_text_iso"], out["_kept"], out["_total"] = encode_paths_budget(
            sg, cfg, _COUNT, a.text_budget)
    return out


def write_hf_arrow(rows, out_dir: str) -> None:
    """Сохранить как datasets.save_to_disk: arrow-stream + dataset_info + state."""
    os.makedirs(out_dir, exist_ok=True)
    str_cols = _STR_COLS + [c for c in ("paths_text_iso",) if c in rows[0]]
    features = {c: {"dtype": "string", "_type": "Value"} for c in str_cols}
    features["gold_shown"] = {"dtype": "bool", "_type": "Value"}
    info = {"citation": "", "description": "", "features": features, "homepage": "", "license": ""}
    cols = {c: pa.array([r[c] for r in rows], type=pa.string()) for c in str_cols}
    cols["gold_shown"] = pa.array([r["gold_shown"] for r in rows], type=pa.bool_())
    table = pa.table(cols).replace_schema_metadata(
        {"huggingface": json.dumps({"info": {"features": features}})})
    with pa.OSFile(os.path.join(out_dir, "data-00000-of-00001.arrow"), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as w:
            w.write_table(table)
    with open(os.path.join(out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    state = {"_data_files": [{"filename": "data-00000-of-00001.arrow"}],
             "_fingerprint": uuid.uuid4().hex[:16], "_format_columns": None,
             "_format_kwargs": {}, "_format_type": None,
             "_output_all_columns": False, "_split": None}
    with open(os.path.join(out_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def main():
    global _ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default="graphvis_scaling/k4/datasets",
                    help="откуда брать тройки (тот же порядок → тот же общий тест)")
    ap.add_argument("--out_root", default="graphvis_paths")
    ap.add_argument("--data_dir", default="FB15k-237")
    ap.add_argument("--splits", default="stage2_train,stage2_test")
    ap.add_argument("--n_show", type=int, default=10)
    ap.add_argument("--max_paths", type=int, default=2,
                    help="M=2: при M=3 у 5%% тест-подграфов текст > 800 токенов (лимит длины)")
    ap.add_argument("--preselect", default="cn_freq", choices=["freq", "cn_freq"])
    ap.add_argument("--cand_cap", type=int, default=50)
    ap.add_argument("--target_size", type=int, default=560)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--label_edges", action=argparse.BooleanOptionalAction, default=True,
                    help="подписи отношений на рёбрах (картинка и текст одинаково)")
    ap.add_argument("--text_budget", type=int, default=0,
                    help=">0: добавить paths_text_iso, ужатый до стольких токенов (400 = картинка 560x560)")
    ap.add_argument("--full_budget", type=int, default=0,
                    help=">0: потолок для paths_text (иначе редкие гигантские подграфы > max_length)")
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/qwen_tok/tokenizer.json"))
    ap.add_argument("--hf_tokenizer", default="Qwen/Qwen2.5-VL-3B-Instruct",
                    help="если --tokenizer не найден: AutoTokenizer из HF-кэша")
    args = ap.parse_args()
    args.out_root_rel = args.out_root.replace("\\", "/").rstrip("/")
    _ARGS = args
    os.makedirs(os.path.join(_ROOT, args.out_root, "graphs"), exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    count = load_counter(args)

    meta = {"src_root": args.src_root, "n_show": args.n_show, "max_paths": args.max_paths,
            "preselect": args.preselect, "cand_cap": args.cand_cap,
            "label_edges": args.label_edges, "text_budget": args.text_budget,
            "full_budget": args.full_budget,
            "target_size": args.target_size, "splits": {}}
    for split in splits:
        src = os.path.join(_ROOT, args.src_root, split, "data-00000-of-00001.arrow")
        rows = pa.ipc.open_stream(src).read_all().select(
            ["head", "relation", "tail", "answer", "query_id"]).to_pylist()
        if args.limit:
            rows = rows[:args.limit]
        t0 = time.time()
        with Pool(args.workers, initializer=_init, initargs=(args, split)) as pool:
            out = []
            for i, r in enumerate(pool.imap(_work, [(split, r) for r in rows], chunksize=8)):
                out.append(r)
                if (i + 1) % 500 == 0:
                    print(f"[{split}] {i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
        write_hf_arrow(out, os.path.join(_ROOT, args.out_root, "datasets", split))

        st = {"n": len(out),
              "gold_shown_%": round(float(np.mean([r["gold_shown"] for r in out])) * 100, 2),
              "intermediates_median/p90": [float(np.median([r["_n_inter"] for r in out])),
                                           float(np.percentile([r["_n_inter"] for r in out], 90))],
              "edges_median/p90": [float(np.median([r["_n_edges"] for r in out])),
                                   float(np.percentile([r["_n_edges"] for r in out], 90))]}
        if args.text_budget:
            cov = [r["_kept"] / r["_total"] for r in out if r["_total"]]
            st["iso_paths_kept_%_mean/median"] = [round(float(np.mean(cov)) * 100, 1),
                                                  round(float(np.median(cov)) * 100, 1)]
            st["iso_truncated_%"] = round(float(np.mean([r["_kept"] < r["_total"] for r in out])) * 100, 1)
        if count:
            for col in ("paths_text", "paths_cn", "paths_text_iso"):
                if col not in out[0]:
                    continue
                n = [count(r[col]) for r in out]
                st[f"tokens_{col}_mean/p95/max"] = [round(float(np.mean(n)), 1),
                                                     float(np.percentile(n, 95)), int(max(n))]
        meta["splits"][split] = st
        print(f"[{split}] готово: {json.dumps(st, ensure_ascii=False)}", flush=True)

    # дописываем к метаданным прошлых запусков (сплиты можно собирать по отдельности)
    mp = os.path.join(_ROOT, args.out_root, "datasets", "metadata.json")
    if os.path.exists(mp):
        old = json.load(open(mp, encoding="utf-8"))
        old["splits"].update(meta["splits"])
        meta["splits"] = old["splits"]
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
