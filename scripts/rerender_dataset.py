"""Перерендер датасета в целевом конфиге (hybrid@4, R=672) — контролируемая замена.

Берёт СУЩЕСТВУЮЩИЙ датасет (stage2_train/valid/test) и меняет ТОЛЬКО картинку:
те же тройки, те же сплиты, те же промпты/ответы — иначе сравнение «старый vs
новый рендер» было бы нечестным. Для каждого примера:

  1. тянем большой пул соседей головы из Neo4j (tail исключён);
  2. ранжируем hybrid (из neighbor_ranking_eval) и берём top-k;
  3. строим звёздный подграф (head + k соседей, только инцидентные голове рёбра);
  4. рендерим в target_size (672) новым визуализатором;
  5. пересобираем пример = старый пример с ПОДМЕНЁННЫМ image_path.

Параметризовано: --method / --k / --target_size — можно собрать любой вариант.
Требуется Neo4j; рендер — graphviz (dot). --dry_run проверяет сборку без рендера.

Usage:
    python scripts/rerender_dataset.py                       # hybrid@4, 672, все сплиты
    python scripts/rerender_dataset.py --limit 200 --dry_run # быстрая проверка сборки
    python scripts/rerender_dataset.py --splits stage2_test --limit 500
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import types
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if "graphvis_lp" not in sys.modules:
    _pkg = types.ModuleType("graphvis_lp"); _pkg.__path__ = [_ROOT]
    sys.modules["graphvis_lp"] = _pkg

import config as _cfg
from retriever import FB15k237GraphRetriever
from visualizer import visualize_task
from text_encoders import encode_features_block
import neighbor_ranking_eval as nre

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rerender")

_SPLIT_ARG = {"stage2_train": "train", "stage2_valid": "valid", "stage2_test": "test"}

# Блок описания головы в промпте Stage2 (шаблоны 1 и 3 генератора):
#   "Entity description: {desc}\n\n"  |  "Description: {desc}\n\n"
# Описание — одна строка (retriever режет до 200 симв., без внутренних \n),
# поэтому [^\n]* безопасно останавливается на границе строки и не задевает
# следующий блок "Query:". Вырезаем ХИРУРГИЧЕСКИ: всё после "Query:" остаётся
# байт-в-байт как было (проверено на всех трёх шаблонах) → старые и новые
# числа сравнимы, отличается только наличие описания.
_DESC_RE = re.compile(r"(?:Entity description|Description): [^\n]*\n\n")


def strip_description(prompt: str) -> str:
    """Убирает блок описания головы из промпта (утечка ответа: в 4.3% троек имя
    хвоста дословно встречается в описании головы). Промпты без описания
    (шаблон 2) не меняются."""
    return _DESC_RE.sub("", prompt)


def _desc_leaks_tail(tail: str, head_description: str) -> bool:
    """True, если имя хвоста дословно встречается в описании головы (то, что
    реально попадало в промпт). Маркер утечки для колонки ``desc_leak``."""
    t = (tail or "").strip().lower()
    d = (head_description or "").lower()
    return bool(t) and t in d


def _safe_render(sg, qid, trel, vis_cfg, retries=4, sleep_s=0.5):
    """Рендер одной картинки с ретраями — чтобы ОДНА заблокированная на Windows
    (открыта в просмотрщике / скан антивируса) не роняла весь joblib-батч.
    Исключение ловим ВНУТРИ воркера → возвращаем None, а не пробрасываем."""
    last = None
    for attempt in range(retries):
        try:
            return visualize_task(sg, qid, trel, vis_cfg)
        except Exception as e:   # transient file-lock (Defender / открытый файл)
            last = e
            time.sleep(sleep_s * (attempt + 1))
    logging.getLogger("rerender").warning("рендер не удался для %s: %s", qid, last)
    return None


def candidates_with_edges(sg, head):
    """Как nre.candidate_edges, но с ВЫРОВНЕННЫМИ исходными рёбрами (для рендера)."""
    cands, orig = [], []
    for e in sg.get("edges", []):
        if e["source"] == head:
            nb, rel, d = e["target"], nre._lastseg(e["relation"]), "out"
        elif e["target"] == head:
            nb, rel, d = e["source"], nre._lastseg(e["relation"]), "in"
        else:
            continue
        cands.append({"nb": nb, "rel": rel, "dir": d})
        orig.append(e)
    return cands, orig


def serialize_facts(head, selected_cands):
    """Те же отобранные рёбра, что и в картинке, но ТЕКСТОМ (формат nre.serialize).
    Пишется в колонку neighbors_text → matched текстовый контекст для сравнения
    модальностей (визуальный контекст = картинка vs текстовый = эта строка)."""
    lines = []
    for c in selected_cands:
        if c["dir"] == "out":
            lines.append(f"- {head} --{c['rel'].replace('_', ' ')}--> {c['nb']}")
        else:
            lines.append(f"- {c['nb']} --{c['rel'].replace('_', ' ')}--> {head}")
    return "\n".join(lines)


def build_star_subgraph(head, relation, head_desc, selected_edges):
    """Звёздный подграф из выбранных инцидентных рёбер (формат визуализатора)."""
    names = [head]
    for e in selected_edges:
        for n in (e["source"], e["target"]):
            if n not in names:
                names.append(n)
    return {
        "head": head,
        "rel": relation,
        "nodes": [{"name": n} for n in names],
        "edges": list(selected_edges),
        "head_description": head_desc or "",
        "stats": {"total_nodes": len(names), "total_edges": len(selected_edges)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="hybrid",
                    choices=["baseline_semantic", "relation_sim", "pmi", "mmr", "hybrid"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--target_size", type=int, default=672)
    ap.add_argument("--splits", default="stage2_train,stage2_valid,stage2_test")
    ap.add_argument("--limit", type=int, default=0, help="ограничить примеров на сплит (0 = все)")
    ap.add_argument("--workers", type=int, default=4, help="воркеров рендера (joblib)")
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "FB15k-237"))
    ap.add_argument("--src_root", default=os.path.join(_ROOT, "graphvis_fb15k_data", "datasets"))
    ap.add_argument("--out_root", default=os.path.join(_ROOT, "graphvis_fb15k_data_hybrid4_672"))
    ap.add_argument("--keep_desc", action="store_true",
                    help="НЕ вырезать описание головы из промпта (старое поведение с утечкой)")
    ap.add_argument("--drop_leak", action="store_true",
                    help="выкинуть тройки, где имя хвоста дословно есть в описании головы")
    ap.add_argument("--dry_run", action="store_true", help="без рендера — только проверить сборку")
    ap.add_argument("--feat_layout", action="store_true",
                    help="в колонку neighbors_feat добавить layout-координаты движка "
                         "(дороже: вызывает dot на подграф). Без флага — только degree/центральность.")
    args = ap.parse_args()

    splits = args.splits.split(",")

    logger.info("Статистики KG + эмбеддинги отношений …")
    stats = nre.build_stats(args.data_dir)
    needs_emb = args.method in ("relation_sim", "mmr", "hybrid")
    relemb = nre.build_rel_embeddings(stats["rels"]) if needs_emb else {}

    vis_cfg = _cfg.VisualizationConfig(
        output_dir=args.out_root, target_size=args.target_size, cache_enabled=False)
    out_graphs = Path(args.out_root) / "graphs"
    out_datasets = Path(args.out_root) / "datasets"
    out_graphs.mkdir(parents=True, exist_ok=True)
    out_datasets.mkdir(parents=True, exist_ok=True)

    from datasets import load_from_disk, Dataset
    neo = _cfg.Neo4jConfig.from_env()
    rcfg = _cfg.RetrieverConfig(neo4j=neo, k_neighbors=nre.BIG_K, include_neighbor_edges=False)

    summary = {}
    with FB15k237GraphRetriever(rcfg) as retr:
        for split in splits:
            ds = load_from_disk(os.path.join(args.src_root, split))
            n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
            split_arg = _SPLIT_ARG.get(split, "test")
            logger.info("=== %s: %d примеров (retriever split=%s) ===", split, n, split_arg)

            tasks, kept_rows, neigh_texts, neigh_feats, leak_flags, no_edges = [], [], [], [], [], 0
            for i in range(n):
                row = ds[i]
                head, rel, tail = row["head"], row["relation"], row["tail"]
                leak_flags.append(_desc_leaks_tail(tail, row.get("head_description", "")))
                rq = nre._lastseg(rel)
                sg = retr.get_head_subgraph(head, rel, tail, split=split_arg)
                if "error" in sg or not sg.get("edges"):
                    no_edges += 1
                    selected, selected_cands = [], []
                else:
                    cands, orig = candidates_with_edges(sg, head)
                    if cands:
                        order = nre.rank_indices(args.method, cands, rq, stats, relemb)
                        sel_idx = order[:args.k]
                        selected = [orig[j] for j in sel_idx]
                        selected_cands = [cands[j] for j in sel_idx]
                    else:
                        selected, selected_cands = [], []
                        no_edges += 1
                sg_new = build_star_subgraph(head, rel, row.get("head_description", ""), selected)
                tasks.append((sg_new, row["query_id"], rel))
                kept_rows.append(row)
                nt = serialize_facts(head, selected_cands)
                neigh_texts.append(nt)
                # Тот же текст + инженерные признаки движка (для условия text_feat).
                nf = ""
                if nt:
                    try:
                        fb = encode_features_block(sg_new, vis_cfg, args.feat_layout)
                        nf = f"{nt}\n\nNode features:\n{fb}"
                    except Exception:
                        nf = nt
                neigh_feats.append(nf)
                if (i + 1) % 1000 == 0:
                    logger.info("  …подграфы %d/%d (пустых рёбер: %d)", i + 1, n, no_edges)

            if args.dry_run:
                avg_edges = sum(len(t[0]["edges"]) for t in tasks) / max(len(tasks), 1)
                n_leak = sum(leak_flags)
                logger.info("[dry] %s: собрано %d подграфов, ср. рёбер=%.2f, без рёбер=%d; "
                            "утечка описания (имя хвоста в описании головы): %d/%d (%.1f%%)"
                            "%s",
                            split, len(tasks), avg_edges, no_edges,
                            n_leak, len(tasks), 100 * n_leak / max(len(tasks), 1),
                            " → будут выкинуты (--drop_leak)" if args.drop_leak else
                            (" → описание в промпте СОХРАНЕНО (--keep_desc)" if args.keep_desc
                             else " → описание из промпта ВЫРЕЗАНО (по умолчанию)"))
                continue

            # ── рендер параллельно (ретраи внутри воркера) ──
            from joblib import Parallel, delayed
            logger.info("Рендер %d картинок в %dpx, workers=%d …", len(tasks), args.target_size, args.workers)
            paths = Parallel(n_jobs=args.workers, backend="loky")(
                delayed(_safe_render)(sg, qid, trel, vis_cfg) for sg, qid, trel in tasks)

            # второй проход по упавшим — последовательно, с большим бэк-оффом
            fail_idx = [i for i, p in enumerate(paths) if p is None]
            if fail_idx:
                logger.warning("Первый проход: %d рендеров упали (блокировки файлов?). "
                               "Повтор последовательно…", len(fail_idx))
                for i in fail_idx:
                    sg, qid, trel = tasks[i]
                    paths[i] = _safe_render(sg, qid, trel, vis_cfg, retries=6, sleep_s=1.0)

            # ── пересборка примеров: меняем image_path + neighbors_text + чистим утечку ──
            new_rows, dropped, n_leak, n_stripped, dropped_leak = [], [], 0, 0, 0
            for row, p, nt, nf, leak in zip(kept_rows, paths, neigh_texts, neigh_feats, leak_flags):
                if p is None:
                    dropped.append(row["query_id"])
                    continue
                r = dict(row)
                # ОТНОСИТЕЛЬНЫЙ путь от корня репо + прямые слэши → датасет переносимый
                # между Windows (myenv) и Linux (A100/песочница); резолвится через
                # cfg.image_root (все три загрузчика картинок его уважают).
                r["image_path"] = os.path.relpath(Path(p).resolve(), _ROOT).replace("\\", "/")
                r["neighbors_text"] = nt          # matched текстовый контекст (те же рёбра)
                r["neighbors_feat"] = nf          # тот же текст + признаки движка (text_feat)
                r["desc_leak"] = bool(leak)        # маркер: имя хвоста было в описании головы
                if leak:
                    n_leak += 1
                if args.drop_leak and leak:
                    dropped_leak += 1
                    continue
                if not args.keep_desc:
                    new_p = strip_description(r["prompt"])
                    if new_p != r["prompt"]:
                        n_stripped += 1
                    r["prompt"] = new_p
                new_rows.append(r)
            if dropped:
                logger.warning("%s: ПРОПУЩЕНО %d примеров (рендер не удался): %s%s",
                               split, len(dropped), dropped[:10],
                               " …" if len(dropped) > 10 else "")
            logger.info("%s: утечка описания=%d (вырезано описаний из промпта=%d, "
                        "выкинуто по --drop_leak=%d)", split, n_leak, n_stripped, dropped_leak)
            out = Dataset.from_list(new_rows)
            out.save_to_disk(str(out_datasets / split))
            summary[split] = {"n": len(new_rows), "no_edges": no_edges, "dropped": len(dropped),
                              "desc_leak": n_leak, "desc_stripped": n_stripped,
                              "dropped_leak": dropped_leak}
            logger.info("Сохранено: %s (%d примеров, без рёбер: %d, пропущено: %d)",
                        out_datasets / split, len(new_rows), no_edges, len(dropped))

    if not args.dry_run:
        import json
        meta_path = out_datasets / "metadata.json"
        prev = {}
        if meta_path.exists():
            try:
                prev = json.loads(meta_path.read_text(encoding="utf-8")).get("splits", {})
            except Exception:
                prev = {}
        prev.update(summary)   # частичный прогон (напр. только test) не стирает train/valid
        meta = {"method": args.method, "k": args.k, "target_size": args.target_size,
                "src_root": args.src_root,
                "strip_description": not args.keep_desc, "drop_leak": args.drop_leak,
                "splits": prev}
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n" + "=" * 60)
        print(f"Перерендер готов: {args.method}@{args.k}, {args.target_size}px")
        for s, v in summary.items():
            print(f"  {s:<14} {v['n']:>7} примеров  (без рёбер: {v['no_edges']})")
        print(f"Новый датасет: {out_datasets}")
        print("Для обучения: cfg.dataset_root =", out_datasets)
        print("=" * 60)


if __name__ == "__main__":
    main()
