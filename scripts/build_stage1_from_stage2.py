"""Собрать stage1_graph_comprehension для hybrid8 БЕЗ Neo4j и без нового рендера.

Переиспользует уже отрендеренные картинки и колонку neighbors_text из stage2_train
(она = ровно та окрестность, что нарисована), воспроизводя формат и задачи
оригинального Stage1DataGenerator (generators.py):

    node_count, edge_count, node_degree (×3), highest_degree, node_listing, triple_listing

Как на изображении, учитываются узел-заглушка [?] и пунктирное ребро head—[rel]→[?]:
    visible_nodes = real_nodes + [?]
    visible_edges = real_edges + (head→[?])
    degree(head) += 1 ;  degree([?]) = 1

Схема строки: {image_path, prompt, answer, task_type, query_id} — идентична старому
stage1. image_path берётся из stage2 как есть (относительный → портируемо).

    # полный (все графы, все 6 задач):
    python scripts/build_stage1_from_stage2.py

    # ЛЁГКИЙ (рекомендуется): ~2500 случайных графов × 3 задачи чтения ≈ 7.5k примеров.
    # Чтение — навык формата и обобщается, поэтому полные 6000×8 избыточны.
    python scripts/build_stage1_from_stage2.py \
        --graphs 2500 --tasks node_count,node_listing,triple_listing --degree_samples 0
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from collections import Counter

from datasets import Dataset, load_from_disk

_QUESTION_NODE = "[?]"

# Формулировки — 1:1 из generators.Stage1DataGenerator._PROMPTS.
_PROMPTS = {
    "node_count": [
        "How many nodes are there in the graph?",
        "Count the total number of nodes shown.",
        "What is the total number of vertices in this graph?",
        "How many entities are depicted in the image?",
        "Tell me the node count of this knowledge graph.",
    ],
    "edge_count": [
        "How many edges are there in the graph?",
        "Count the total number of connections.",
        "What is the total number of edges in this graph?",
        "How many relationships are shown in the image?",
        "Tell me the edge count of this knowledge graph.",
    ],
    "node_degree": [
        "What is the degree of the node '{node}'?",
        "How many connections does '{node}' have?",
        "What is the number of edges incident to '{node}'?",
        "Tell me the degree of vertex '{node}'.",
        "How many neighbors does the node '{node}' have?",
    ],
    "highest_degree": [
        "Which node has the highest degree in the graph?",
        "Identify the node with the most connections.",
        "What is the name of the node with maximum degree?",
        "Which entity has the most relationships in this graph?",
        "Find the node with the highest number of edges.",
    ],
    "node_listing": [
        "List all nodes shown in the graph.",
        "Name all entities depicted in the image.",
        "What are all the nodes in this knowledge graph?",
        "Enumerate all vertices shown.",
        "Provide a list of all node names in the graph.",
    ],
    "triple_listing": [
        "List all triples in the graph.",
        "Enumerate all (head, relation, tail) triples shown.",
        "What are all the relationships in this graph?",
        "List all edges as (source, label, target) triples.",
        "Describe the graph by listing all its triples.",
    ],
}

# "- SOURCE --RELATION--> TARGET"
_LINE = re.compile(r"^-\s*(.+?)\s+--(.+?)-->\s+(.+?)\s*$")


def parse_edges(neighbors_text: str):
    """Разбирает neighbors_text в список рёбер {source, relation, target}."""
    edges = []
    for ln in (neighbors_text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = _LINE.match(ln)
        if m:
            s, rel, t = (x.strip() for x in m.groups())
            edges.append({"source": s, "relation": rel, "target": t})
    return edges


def _drawn(head, qrel, real_edges):
    """Эталон = ровно нарисованное: тот же отбор узлов, что в рендере
    (compute_visible_subgraph: первые max_nodes_display), и подписи через
    format_label (обрезка, '_' → ' '). Без этого при K=16 stage1 учил бы
    «читать» соседей, которых на картинке нет, и сырые имена с '_'.
    Структура (счёт узлов, степени) — по СЫРЫМ именам: два узла с одинаковой
    обрезанной подписью на картинке — два разных бокса. Возвращает видимые
    рёбра (сырые имена) и функцию подписи."""
    import sys as _sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from config import VisualizationConfig
    from visualizer import compute_visible_subgraph, format_label

    order = []
    for n in [head] + [x for e in real_edges for x in (e["source"], e["target"])]:
        if n not in order:
            order.append(n)
    vcfg = VisualizationConfig()
    sg = {"head": head, "rel": qrel, "nodes": [{"name": n} for n in order],
          "edges": real_edges}
    _, vis = compute_visible_subgraph(sg, vcfg)
    L = vcfg.max_label_len
    return [dict(e) for e in vis], (lambda x: x if x == _QUESTION_NODE else format_label(x, L))


def gen_examples(row, rng: random.Random, tasks: set, degree_k: int = 3,
                 drawn: bool = False):
    head = row["head"]
    qrel = row["relation"]                       # отношение запроса → подпись на head→[?]
    real_edges = parse_edges(row["neighbors_text"])
    if not real_edges:
        return []
    lab = lambda x: x                            # noqa: E731
    if drawn:
        real_edges, lab = _drawn(head, qrel, real_edges)
        if not real_edges:
            return []

    # Множество нарисованных узлов: head первым, затем по первому появлению.
    seen, real_nodes = set(), []

    def _push(n):
        if n not in seen:
            seen.add(n)
            real_nodes.append(n)

    _push(head)
    for e in real_edges:
        _push(e["source"])
        _push(e["target"])

    visual_nodes = real_nodes + [_QUESTION_NODE]
    visual_edges = real_edges + [{"source": head, "target": _QUESTION_NODE, "relation": qrel}]

    deg = Counter()
    for e in real_edges:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    deg[head] = deg.get(head, 0) + 1             # ребро head→[?]
    deg[_QUESTION_NODE] = 1

    img, qid = row["image_path"], row["query_id"]
    out = []

    def _q(task):
        return "<image>\n" + rng.choice(_PROMPTS[task])

    def _add(task, answer, prompt=None):
        if task not in tasks:                    # тип задачи отключён через --tasks
            return
        out.append({"image_path": img, "prompt": prompt or _q(task),
                    "answer": answer, "task_type": task, "query_id": qid})

    _add("node_count", str(len(visual_nodes)))
    _add("edge_count", str(len(visual_edges)))

    if "node_degree" in tasks and degree_k > 0:
        pool = [n for n in visual_nodes if n in deg]
        for node in rng.sample(pool, min(degree_k, len(pool))):
            _add("node_degree", str(deg[node]), prompt=_q("node_degree").format(node=lab(node)))

    if deg:
        _add("highest_degree", lab(max(deg, key=deg.get)))

    _add("node_listing", ", ".join(lab(n) for n in visual_nodes))
    _add("triple_listing",
         "; ".join(f"({lab(e['source'])}, {lab(e['relation'])}, {lab(e['target'])})"
                   for e in visual_edges))
    return out


def main():
    ap = argparse.ArgumentParser()
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--stage2",
                    default=os.path.join(_root, "graphvis_fb15k_data_hybrid8_672",
                                         "datasets", "stage2_train"))
    ap.add_argument("--out",
                    default=os.path.join(_root, "graphvis_fb15k_data_hybrid8_672",
                                         "datasets", "stage1_graph_comprehension"))
    ap.add_argument("--limit", type=int, default=0,
                    help="взять ПЕРВЫЕ N графов (0 = все); для случайной выборки см. --graphs")
    ap.add_argument("--graphs", type=int, default=0,
                    help="СЛУЧАЙНАЯ выборка N графов (seeded); 0 = все/--limit. Чтение графа — "
                         "навык ФОРМАТА и обобщается, поэтому полные 6000 избыточны: "
                         "~2000-2500 разнообразных графов насыщают навык.")
    ap.add_argument("--tasks",
                    default="node_count,edge_count,node_degree,highest_degree,node_listing,triple_listing",
                    help="типы задач через запятую. Для лёгкого stage1 достаточно "
                         "node_count,node_listing,triple_listing — именно listings заставляют "
                         "прочитать все имена и отношения (ближе всего к link prediction).")
    ap.add_argument("--degree_samples", type=int, default=3,
                    help="сколько вопросов node_degree на граф (0 = выкл; было 3 — избыточно)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drawn", action="store_true",
                    help="эталон = НАРИСОВАННОЕ (лимит узлов рендера + подписи format_label). "
                         "Обязательно при K > 13: иначе в ответах соседи, которых нет на картинке")
    args = ap.parse_args()

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    _known = set(_PROMPTS)
    if not tasks or (tasks - _known):
        raise SystemExit(f"--tasks: неизвестные типы {sorted(tasks - _known)}; "
                         f"допустимо: {sorted(_known)}")

    ds = load_from_disk(args.stage2)
    rng = random.Random(args.seed)

    idx = list(range(len(ds)))
    if args.graphs and args.graphs < len(ds):
        idx = sorted(rng.sample(idx, args.graphs))          # случайная выборка графов
    elif args.limit and args.limit < len(ds):
        idx = idx[:args.limit]                               # первые N

    rows, empty = [], 0
    for i in idx:
        ex = gen_examples(ds[i], rng, tasks, args.degree_samples, drawn=args.drawn)
        if not ex:
            empty += 1
        rows.extend(ex)

    # Пересборка деривативного датасета: если по пути уже лежит HF-датасет — заменяем
    # (это артефакт билдера, регенерируется); иначе не трогаем чужую папку.
    if os.path.isdir(args.out):
        if any(os.path.exists(os.path.join(args.out, f))
               for f in ("dataset_info.json", "state.json")):
            shutil.rmtree(args.out)
            print(f"(перезаписываю существующий датасет: {args.out})")
        else:
            raise SystemExit(f"--out существует и не похож на HF-датасет: {args.out}")

    out_ds = Dataset.from_list(rows)
    out_ds.save_to_disk(args.out)

    dist = Counter(r["task_type"] for r in rows)
    print(f"Графов обработано: {len(idx)}  (пустых neighbors_text: {empty})")
    print(f"Примеров stage1:   {len(rows)}")
    print(f"По задачам:        {dict(dist)}")
    print(f"Задачи:            {sorted(tasks)}  degree_samples={args.degree_samples}")
    print(f"Сохранено:         {args.out}")


if __name__ == "__main__":
    main()
