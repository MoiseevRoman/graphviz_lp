"""Зонд читаемости: что энкодер реально «видит» на графе — по сетке (разрешение × k).

Идея: НЕ обучать VLM, а zero-shot спросить модель про то, что она видит на картинке
(counting / чтение имён и подписей рёбер). Точность = потолок читаемости энкодера.
Прогоняется для каждой пары (target_size R, k_neighbors k), чтобы решить, куда идти:
поднимать разрешение или уменьшать число соседей.

Задачи-зонды (ground-truth берётся из того, что РЕАЛЬНО нарисовано —
`compute_visible_subgraph`):
    node_count      — сколько сущностей-боксов (без [?]);
    edge_count      — сколько подписанных рёбер (без пунктирного к [?]);
    read_name       — имя соседа, на которого указывает ребро (head --rel--> ?);
    read_relation   — подпись отношения на конкретном ребре.

read_relation / read_name — самые важные: именно они нужны для link prediction.

Требуется запущенный Neo4j (для подграфов) и GPU (для zero-shot Qwen).

Usage:
    python -m graphvis_lp.scripts.legibility_probe \
        --n 300 --resolutions 336,672,896 --neighbors 4,8,12 \
        --model Qwen/Qwen2.5-VL-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile

# ── пути: репозиторий использует ДВА стиля импортов ──────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # /workspace
_PARENT = os.path.dirname(_ROOT)
for _p in (_ROOT, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Пакет импортируется как `graphvis_lp`, но папка может называться иначе
# (напр. graphviz_lp). Регистрируем алиас на корень репозитория — тогда
# graphvis_lp.* резолвится независимо от имени папки.
import types as _types
if "graphvis_lp" not in sys.modules:
    _pkg = _types.ModuleType("graphvis_lp"); _pkg.__path__ = [_ROOT]
    sys.modules["graphvis_lp"] = _pkg

import numpy as np
from PIL import Image

# top-level модули (retriever/visualizer внутри делают `from config import ...`)
import config as _cfg
from retriever import FB15k237GraphRetriever
from visualizer import FB15kGraphVisualizer, compute_visible_subgraph
from utils import normalize_answer

# пакетные модули обучения
from graphvis_lp.config import TrainConfig
from graphvis_lp.training.families import detect_family, format_eval_text
from graphvis_lp.training.model import build_model_and_processor

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
def _int_in(text: str):
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group()) if m else None


def ask(model, processor, tokenizer, family, image, question, max_new_tokens=24):
    import torch
    with torch.no_grad():
        text = format_eval_text(processor, family, question)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        plen = inputs["input_ids"].shape[1]
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             num_beams=1, do_sample=False)
    return tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()


def build_probes(subgraph, viscfg, rng):
    """GT для 4 зондов из ТОГО, что реально нарисовано."""
    nodes, edges = compute_visible_subgraph(subgraph, viscfg)
    head = subgraph["head"]
    probes = []

    probes.append(("node_count",
                   "How many entity boxes are shown in the image, "
                   "NOT counting the dashed '[?]' box? Answer with a number only.",
                   len(nodes)))
    probes.append(("edge_count",
                   "How many labeled edges (arrows with a text label) are shown, "
                   "NOT counting the dashed edge to '[?]'? Answer with a number only.",
                   len(edges)))

    head_edges = [e for e in edges if e["source"] == head]
    if head_edges:
        e = head_edges[rng.randrange(len(head_edges))]
        rel = e["relation"].replace("_", " ")
        gt_names = {normalize_answer(x["target"])
                    for x in head_edges if x["relation"] == e["relation"]}
        probes.append(("read_name",
                       f"In the image, which entity does '{head}' point to "
                       f"via the relation '{rel}'? Answer with the entity name only.",
                       gt_names))
    if edges:
        e = edges[rng.randrange(len(edges))]
        a, b = e["source"], e["target"]
        gt_rel = {normalize_answer(x["relation"])
                  for x in edges if x["source"] == a and x["target"] == b}
        probes.append(("read_relation",
                       f"What is the relation label written on the edge from "
                       f"'{a}' to '{b}'? Answer with the label only.",
                       gt_rel))
    return probes


def evaluate_config(model, processor, tokenizer, family, retr, samples, R, k, tmproot):
    viscfg = _cfg.VisualizationConfig(
        output_dir=os.path.join(tmproot, f"R{R}_k{k}"),
        target_size=R, cache_enabled=False)
    vis = FB15kGraphVisualizer(viscfg)
    import random as _r
    rng = _r.Random(1234)

    hit = {t: [0, 0] for t in ("node_count", "edge_count", "read_name", "read_relation")}
    for s in samples:
        sg = retr.get_head_subgraph(s["head"], s["relation"], s["tail"], split="test")
        if "error" in sg or not sg.get("edges"):
            continue
        path = vis.render(sg, s["query_id"] + f"_R{R}k{k}", s["relation"])
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue
        for name, q, gt in build_probes(sg, viscfg, rng):
            ans = ask(model, processor, tokenizer, family, image, q)
            if name in ("node_count", "edge_count"):
                ok = (_int_in(ans) == gt)
            else:
                ok = (normalize_answer(ans) in gt) or any(
                    g and g in normalize_answer(ans) for g in gt)
            hit[name][0] += int(ok)
            hit[name][1] += 1
    return {t: (c[0] / c[1] if c[1] else float("nan")) for t, c in hit.items()}, \
           {t: c[1] for t, c in hit.items()}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--resolutions", default="336,672,896")
    ap.add_argument("--neighbors", default="4,8,12")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    # Потолок визуальных токенов Qwen. Дефолт TrainConfig ≈720 токенов (564k px)
    # режет R≥896 даунсэмплом. Подняв это, проверяем ЧЕСТНОЕ высокое разрешение.
    ap.add_argument("--max_tokens", type=int, default=0,
                    help="верхний предел визуальных токенов (0 = дефолт из TrainConfig)")
    ap.add_argument("--split", default="stage2_test")
    ap.add_argument("--dataset_root", default=os.path.join(_ROOT, "graphvis_fb15k_data", "datasets"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "graphvis_ckpts", "legibility_probe.json"))
    args = ap.parse_args()

    resolutions = [int(x) for x in args.resolutions.split(",")]
    ks = [int(x) for x in args.neighbors.split(",")]

    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(args.dataset_root, args.split))
    idx = sorted(np.random.RandomState(42).choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist())
    samples = [ds[i] for i in idx]
    logger.info("Выборка: %d запросов", len(samples))

    cfg = TrainConfig()
    cfg.model_name = args.model
    cfg.use_lora = False
    if args.max_tokens > 0:
        cfg.qwen_max_pixels = args.max_tokens * 28 * 28
        logger.info("qwen_max_pixels поднят до %d px (~%d токенов)",
                    cfg.qwen_max_pixels, args.max_tokens)
    family = detect_family(cfg.model_name, cfg.model_family)
    model, processor, tokenizer = build_model_and_processor(cfg)
    model.eval()

    neo = _cfg.Neo4jConfig.from_env()
    results = {}   # (R,k) -> {task: acc}
    counts = {}
    tmproot = tempfile.mkdtemp(prefix="legib_")
    for k in ks:
        rcfg = _cfg.RetrieverConfig(neo4j=neo, k_neighbors=k)
        with FB15k237GraphRetriever(rcfg) as retr:
            for R in resolutions:
                logger.info("=== R=%d, k=%d ===", R, k)
                acc, cnt = evaluate_config(model, processor, tokenizer, family,
                                           retr, samples, R, k, tmproot)
                results[f"R{R}_k{k}"] = acc
                counts[f"R{R}_k{k}"] = cnt
                logger.info("acc: %s", {t: round(v, 3) for t, v in acc.items()})

    # ── печать тепловых карт ──
    def table(task):
        print(f"\n{task}  (строки=k, столбцы=R)")
        print("k\\R   " + "  ".join(f"{R:>6}" for R in resolutions))
        for k in ks:
            row = "  ".join(f"{results[f'R{R}_k{k}'].get(task, float('nan')):>6.3f}" for R in resolutions)
            print(f"{k:>3}   {row}")

    for t in ("node_count", "edge_count", "read_name", "read_relation"):
        table(t)

    # ── рекомендация по read_relation (ключевая метрика) ──
    key = "read_relation"
    kmax = max(ks)
    accs_R = [results[f"R{R}_k{kmax}"].get(key, 0.0) for R in resolutions]
    best = max(accs_R)
    # минимальное R в пределах 2% от лучшего при k=kmax
    R_star = next((R for R, a in zip(resolutions, accs_R) if a >= best - 0.02), resolutions[-1])
    gain_R = best - accs_R[0]
    Rmax = max(resolutions)
    accs_k = [results[f"R{Rmax}_k{k}"].get(key, 0.0) for k in ks]
    gain_k = max(accs_k) - results[f"R{Rmax}_k{kmax}"].get(key, 0.0)  # выигрыш от УМЕНЬШЕНИЯ k при max R

    print("\n" + "=" * 64)
    print(f"read_relation при k={kmax}: {dict(zip(resolutions, [round(a,3) for a in accs_R]))}")
    print(f"  насыщение примерно при R*={R_star} (прирост от роста R: {gain_R:+.3f})")
    print(f"read_relation при R={Rmax}: {dict(zip(ks, [round(a,3) for a in accs_k]))}")
    print(f"  прирост от уменьшения k при макс. R: {gain_k:+.3f}")
    if gain_R >= max(gain_k, 0.03):
        print("РЕКОМЕНДАЦИЯ: главный рычаг — РАЗРЕШЕНИЕ. Рендерить в R*≈%d, k держать." % R_star)
    elif gain_k > gain_R:
        print("РЕКОМЕНДАЦИЯ: вы clutter-bound — УМЕНЬШАЙТЕ k (разрешение уже насыщено).")
    else:
        print("РЕКОМЕНДАЦИЯ: оба рычага дают мало — узкое место не в картинке (см. neighbor_ranking_eval).")
    print("=" * 64)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"results": results, "counts": counts,
               "resolutions": resolutions, "neighbors": ks, "model": args.model},
              open(args.out, "w"), indent=2, ensure_ascii=False)
    print("Сохранено:", args.out)


if __name__ == "__main__":
    main()
