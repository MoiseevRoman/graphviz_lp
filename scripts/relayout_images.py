#!/usr/bin/env python3
"""Layout-augmentation: пере-рендер обучающих картинок разными раскладками.

Мотивация (GITA, NeurIPS'24): layout augmentation — сильнейший рычаг качества
vision-графового ризонинга (там avg 38.9 → 63.4). Чтобы сравнение модальностей
было ЧЕСТНЫМ, визуальному условию нужно дать его лучшую версию: обучение на
разнообразных раскладках (dot/neato/fdp/circo/twopi), а не на одной.

Здесь мы НЕ ходим в Neo4j и НЕ меняем подграф: рёбра берём из готовой колонки
neighbors_text (те же рёбра, что в text/visual), реконструируем звёздный
подграф и рендерим его выбранным движком. Аугментация применяется только к
TRAIN (каждому примеру — свой движок, детерминированно по query_id); TEST
остаётся на фиксированном dot (детерминированная оценка), его картинки не
трогаем — датасет теста просто копируется с исходными image_path.

Пишет НОВОЕ дерево (--out_base, по умолч. graphvis_scaling_aug); исходное
graphvis_scaling не изменяется.

Пример:
    python scripts/relayout_images.py --budgets 4,16 \
        --engines dot,neato,fdp,circo,twopi
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import VisualizationConfig            # noqa: E402
from visualizer import visualize_task             # noqa: E402
from add_text_encodings import parse_neighbors_text  # noqa: E402


def _pick_engine(engines, seed, query_id):
    """Детерминированный выбор движка по query_id (воспроизводимо)."""
    h = hashlib.md5(f"{seed}:{query_id}".encode("utf-8")).hexdigest()
    return engines[int(h, 16) % len(engines)]


def _build_sg(head, relation, edges, order, head_desc):
    if order and order[0] != head:
        order = [head] + [n for n in order if n != head]
    return {
        "head": head,
        "rel": relation,
        "nodes": [{"name": n} for n in order],
        "edges": [{"source": s, "relation": r, "target": t} for (s, r, t) in edges],
        "head_description": head_desc or "",
        "stats": {"total_nodes": len(order), "total_edges": len(edges)},
    }


def _render_one(sg, qid, relation, engine, out_dir, target_size):
    """Рендер выбранным движком с откатом на dot, если движок недоступен."""
    for eng in (engine, "dot"):
        cfg = VisualizationConfig()
        cfg.output_dir = out_dir
        cfg.engine = eng
        cfg.target_size = target_size
        cfg.cache_enabled = False
        try:
            return visualize_task(sg, qid, relation, cfg), eng
        except Exception:
            if eng == "dot":
                raise
    return None, None


def process_train(path, out_ds_dir, graphs_dir, engines, seed, target_size):
    from datasets import load_from_disk
    ds = load_from_disk(path)
    used = {}

    def _relayout(row):
        head = row["head"]
        rel = row["relation"]
        edges, order = parse_neighbors_text(row.get("neighbors_text", ""))
        if edges is None or not edges:
            return {"image_path": row.get("image_path", "")}  # нечего рендерить
        sg = _build_sg(head, rel, edges, order, row.get("head_description", ""))
        eng = _pick_engine(engines, seed, row["query_id"])
        p, eng_used = _render_one(sg, row["query_id"], rel, eng, graphs_dir, target_size)
        used[eng_used] = used.get(eng_used, 0) + 1
        rel_p = os.path.relpath(Path(p).resolve(), _ROOT).replace("\\", "/")
        return {"image_path": rel_p}

    ds = ds.map(_relayout, desc="relayout[train]", keep_in_memory=True,
                load_from_cache_file=False)
    ds.save_to_disk(out_ds_dir)
    return len(ds), used


def copy_split(path, out_ds_dir):
    """Тест копируем как есть (image_path указывает на исходные dot-картинки)."""
    from datasets import load_from_disk
    ds = load_from_disk(path)
    ds.save_to_disk(out_ds_dir)
    return len(ds)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", default="4,16", help="K через запятую")
    ap.add_argument("--data_base", default=os.path.join(_ROOT, "graphvis_scaling"))
    ap.add_argument("--out_base", default=os.path.join(_ROOT, "graphvis_scaling_aug"))
    ap.add_argument("--engines", default="dot,neato,fdp,circo,twopi",
                    help="движки раскладки для аугментации TRAIN")
    ap.add_argument("--train_split", default="stage2_train")
    ap.add_argument("--test_split", default="stage2_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_size", type=int, default=560)
    ap.add_argument("--dry_run", action="store_true", help="показать пример без записи")
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    Ks = [int(x) for x in args.budgets.split(",") if x.strip()]
    print(f"layout-aug: движки={engines}  seed={args.seed}")
    print(f"src={args.data_base}  →  dst={args.out_base}\n")

    if args.dry_run:
        # распределение движков по первым query_id (проверка баланса)
        import collections
        dist = collections.Counter()
        for i in range(3000):
            dist[_pick_engine(engines, args.seed, f"q{i:06d}")] += 1
        print("проверка баланса движков на 3000 синтетических id:")
        for e in engines:
            print(f"  {e:8}: {dist[e]}")
        return

    for k in Ks:
        base = os.path.join(args.data_base, f"k{k}", "datasets")
        out_k = os.path.join(args.out_base, f"k{k}")
        graphs_dir = out_k  # visualize_task пишет в <output_dir>/graphs
        os.makedirs(out_k, exist_ok=True)
        tr = os.path.join(base, args.train_split)
        te = os.path.join(base, args.test_split)
        if os.path.isdir(tr):
            n, used = process_train(tr, os.path.join(out_k, "datasets", args.train_split),
                                    graphs_dir, engines, args.seed, args.target_size)
            print(f"[k={k}] train: {n} примеров пере-рендерено; движки={dict(used)}")
        else:
            print(f"[k={k}] нет {tr} — пропуск train")
        if os.path.isdir(te):
            m = copy_split(te, os.path.join(out_k, "datasets", args.test_split))
            print(f"[k={k}] test:  {m} примеров скопировано (dot, image_path без изменений)")
        else:
            print(f"[k={k}] нет {te} — пропуск test")


if __name__ == "__main__":
    main()
