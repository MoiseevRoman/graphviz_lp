"""Выбор метода ранжирования соседей — интринсик-оценка (без обучения VLM).

При k=12 из сотен соседей ВЫБОР этих 12 решает, выводим ли ответ. Сравниваем методы
ранжирования по метрикам, не требующим рендера и (для ядра) вообще никакой модели:

Методы:
    baseline_semantic — текущий: порядок ретривера (cosine(head, tail), query-agnostic);
    relation_sim      — query-aware: cos(embed(rel_i), embed(r)) — близость отношения к запросу;
    pmi               — query-aware статистика: PMI(rel_i, r) по обучающему KG;
    mmr               — relation_sim как релевантность + разнообразие по типам (MMR);
    hybrid            — z(relation_sim)+z(pmi)+z(hub_idf), затем MMR-диверсификация.

Метрики на (метод, k):
    has_query_rel@k — доля запросов, где среди выбранных есть ребро с отношением r;
    pmi_support@k   — доля, где выбран хоть один rel из топ-PMI отношения r;
    rel_entropy     — энтропия типов отношений среди выбранных (разнообразие);
    redundancy      — 1 − уникальные_отношения/k (избыточность);
    hub_frac        — доля выбранных соседей-хабов (высокая степень);
    textlp_acc      — (опц.) exact-match предсказание хвоста текстовой моделью по
                      сериализованным выбранным соседям. Это ВЕРХНЯЯ ГРАНИЦА визуального
                      метода: картинка не даст больше, чем отобранные соседи.

Порядок использования: сначала этим скриптом выбрать метод, потом legibility_probe (R×k),
потом подтвердить полным QLoRA. Требуется Neo4j; для --textlp hf — GPU.

Usage:
    python -m graphvis_lp.scripts.neighbor_ranking_eval \
        --n 500 --ks 4,8,12 --textlp none          # быстро, только интринсик
    python -m graphvis_lp.scripts.neighbor_ranking_eval \
        --n 300 --ks 12 --textlp hf --model Qwen/Qwen2.5-VL-3B-Instruct
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_ROOT)
for _p in (_ROOT, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Пакет импортируется как `graphvis_lp`, но папка может называться иначе
# (напр. graphviz_lp). Алиас на корень репо → graphvis_lp.* резолвится
# независимо от имени папки (нужно для --textlp hf).
import types as _types
if "graphvis_lp" not in sys.modules:
    _pkg = _types.ModuleType("graphvis_lp"); _pkg.__path__ = [_ROOT]
    sys.modules["graphvis_lp"] = _pkg

import numpy as np

import config as _cfg
from retriever import FB15k237GraphRetriever
from utils import normalize_answer

logger = logging.getLogger(__name__)

BIG_K = 1000          # «весь» пул соседей головы
PMI_TOPM = 10         # сколько топ-PMI отношений считать «поддержкой»


def _lastseg(r: str) -> str:
    return r.split("/")[-1] or r


# ──────────────────────────────────────────────────────────────────
# Статистики из обучающего KG (name-space, last-seg отношения)
# ──────────────────────────────────────────────────────────────────
def build_stats(data_dir):
    mid2name = {}
    for line in open(os.path.join(data_dir, "FB15k_mid2name.txt"), encoding="utf-8"):
        p = line.rstrip("\n").split("\t", 1)
        if len(p) == 2:
            mid2name[p[0]] = p[1].strip()
    def name(m): return mid2name.get(m, m.split("/")[-1].replace("_", " "))

    head_rels = collections.defaultdict(set)   # head_name -> {rel}
    rel_freq = collections.Counter()
    deg = collections.Counter()
    n_heads = 0
    for line in open(os.path.join(data_dir, "train.txt"), encoding="utf-8"):
        p = line.strip().split("\t")
        if len(p) != 3:
            continue
        h, r, t = name(p[0]), _lastseg(p[1]), name(p[2])
        head_rels[h].add(r)
        deg[h] += 1; deg[t] += 1

    # частоты и совместная встречаемость отношений на одной голове
    co = collections.Counter()
    for h, rels in head_rels.items():
        n_heads += 1
        rl = list(rels)
        for r in rl:
            rel_freq[r] += 1
        for i in range(len(rl)):
            for j in range(len(rl)):
                if i != j:
                    co[(rl[i], rl[j])] += 1

    def pmi(ri, rq):
        # Сосед с ТЕМ ЖЕ отношением r — сильный контекст (у головы уже есть r-рёбра):
        # даём высокий self-скор (само-информация rq), тем больше, чем реже rq.
        if ri == rq:
            p_q = rel_freq.get(rq, 1) / max(n_heads, 1)
            return -math.log(p_q + 1e-12)
        c = co.get((ri, rq), 0)
        if c == 0:
            return -9.0
        p_i = rel_freq[ri] / n_heads
        p_q = rel_freq[rq] / n_heads
        p_ij = c / n_heads
        return math.log(p_ij / (p_i * p_q) + 1e-12)

    # топ-PMI соседи для каждого отношения
    all_rels = list(rel_freq)
    pmi_top = {}
    for rq in all_rels:
        scored = sorted(((pmi(ri, rq), ri) for ri in all_rels), reverse=True)
        pmi_top[rq] = {ri for _, ri in scored[:PMI_TOPM]}

    hub_thresh = np.percentile(list(deg.values()), 90) if deg else 0
    return {"pmi": pmi, "pmi_top": pmi_top, "deg": deg,
            "hub_thresh": hub_thresh, "rels": all_rels}


def build_rel_embeddings(rels, device="cpu"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    texts = [r.replace("_", " ") for r in rels]
    emb = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return {r: e for r, e in zip(rels, emb)}


# ──────────────────────────────────────────────────────────────────
# Кандидаты (рёбра, инцидентные голове; tail уже исключён ретривером)
# ──────────────────────────────────────────────────────────────────
def candidate_edges(subgraph, head):
    out = []
    for e in subgraph.get("edges", []):
        if e["source"] == head:
            nb, rel = e["target"], _lastseg(e["relation"])
        elif e["target"] == head:
            nb, rel = e["source"], _lastseg(e["relation"])
        else:
            continue
        out.append({"nb": nb, "rel": rel,
                    "dir": "out" if e["source"] == head else "in"})
    return out                     # в порядке ретривера (= semantic_weight desc)


# ── методы ранжирования: возвращают список индексов cand в порядке предпочтения ──
def rank_indices(method, cand, rq, stats, relemb):
    n = len(cand)
    if method == "baseline_semantic":
        return list(range(n))
    if method == "relation_sim":
        rqv = relemb.get(rq)
        def sim(c):
            v = relemb.get(c["rel"])
            return float(np.dot(rqv, v)) if (rqv is not None and v is not None) else -1.0
        return sorted(range(n), key=lambda i: sim(cand[i]), reverse=True)
    if method == "pmi":
        return sorted(range(n), key=lambda i: stats["pmi"](cand[i]["rel"], rq), reverse=True)
    if method in ("mmr", "hybrid"):
        rqv = relemb.get(rq)
        def rel_sim(c):
            v = relemb.get(c["rel"])
            return float(np.dot(rqv, v)) if (rqv is not None and v is not None) else 0.0
        if method == "hybrid":
            rs = np.array([rel_sim(c) for c in cand])
            pm = np.array([stats["pmi"](c["rel"], rq) for c in cand])
            hb = np.array([-math.log(1 + stats["deg"].get(c["nb"], 1)) for c in cand])
            z = lambda a: (a - a.mean()) / (a.std() + 1e-9) if len(a) > 1 else a * 0
            base = z(rs) + z(pm) + z(hb)
        else:
            base = np.array([rel_sim(c) for c in cand])
        # MMR-диверсификация по эмбеддингам отношений.
        # Прунинг: MMR только среди топ-POOL по base, остальное — хвостом по base
        # (нам нужны лишь топ-k, поэтому отбираем CAP кандидатов).
        lam = 0.7
        POOL = min(n, 64)
        CAP = min(n, 24)
        pool = sorted(range(n), key=lambda i: base[i], reverse=True)[:POOL]
        def csim(i, j):
            a, b = relemb.get(cand[i]["rel"]), relemb.get(cand[j]["rel"])
            return float(np.dot(a, b)) if (a is not None and b is not None) else float(cand[i]["rel"] == cand[j]["rel"])
        chosen, remaining = [], list(pool)
        while remaining and len(chosen) < CAP:
            best_i, best_v = None, -1e18
            for i in remaining:
                div = max((csim(i, j) for j in chosen), default=0.0)
                v = lam * base[i] - (1 - lam) * div
                if v > best_v:
                    best_v, best_i = v, i
            chosen.append(best_i); remaining.remove(best_i)
        tail = [i for i in sorted(range(n), key=lambda i: base[i], reverse=True) if i not in set(chosen)]
        return chosen + tail
    raise ValueError(method)


# ── интринсик-метрики набора выбранных рёбер ──
def set_metrics(selected, rq, stats):
    rels = [c["rel"] for c in selected]
    has_q = any(r == rq for r in rels)
    pmi_sup = any(r in stats["pmi_top"].get(rq, set()) for r in rels)
    cnt = collections.Counter(rels)
    probs = np.array(list(cnt.values()), dtype=float); probs /= probs.sum()
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    redund = 1.0 - len(cnt) / max(len(rels), 1)
    hub = np.mean([stats["deg"].get(c["nb"], 0) >= stats["hub_thresh"] for c in selected]) if selected else 0.0
    return has_q, pmi_sup, ent, redund, float(hub)


# ── text-LP (опц.): верхняя граница по сериализованным соседям ──
def serialize(head, rq, selected):
    lines = []
    for c in selected:
        if c["dir"] == "out":
            lines.append(f"- {head} --{c['rel'].replace('_',' ')}--> {c['nb']}")
        else:
            lines.append(f"- {c['nb']} --{c['rel'].replace('_',' ')}--> {head}")
    facts = "\n".join(lines)
    return (f"Known facts about '{head}':\n{facts}\n\n"
            f"Question: {head} --{rq.replace('_',' ')}--> ?\n"
            f"Answer with the target entity name only.")


def make_textlp(backend, model_name):
    if backend == "none":
        return None
    if backend == "hf":
        from graphvis_lp.config import TrainConfig
        from graphvis_lp.training.families import detect_family
        from graphvis_lp.training.model import build_model_and_processor
        import torch
        cfg = TrainConfig(); cfg.model_name = model_name; cfg.use_lora = False
        family = detect_family(cfg.model_name, cfg.model_family)
        model, processor, tokenizer = build_model_and_processor(cfg); model.eval()

        def gen(prompt):
            with torch.no_grad():
                if family == "qwen":
                    text = processor.apply_chat_template(
                        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                        tokenize=False, add_generation_prompt=True)
                else:
                    text = prompt
                inputs = processor(text=[text], return_tensors="pt", padding=True)
                dev = next(model.parameters()).device
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                plen = inputs["input_ids"].shape[1]
                out = model.generate(**inputs, max_new_tokens=24, num_beams=1, do_sample=False)
            return tokenizer.decode(out[0][plen:], skip_special_tokens=True)
        return gen
    if backend == "litellm":
        from openai import OpenAI
        client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"),
                        api_key=os.environ.get("OPENAI_API_KEY", "x"))

        def gen(prompt):
            r = client.chat.completions.create(
                model=model_name, temperature=0,
                messages=[{"role": "user", "content": prompt}], max_tokens=24)
            return r.choices[0].message.content
        return gen
    raise ValueError(backend)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--ks", default="4,8,12")
    ap.add_argument("--methods", default="baseline_semantic,relation_sim,pmi,mmr,hybrid")
    ap.add_argument("--textlp", choices=["none", "hf", "litellm"], default="none")
    ap.add_argument("--textlp_n", type=int, default=200, help="подвыборка для text-LP")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "FB15k-237"))
    ap.add_argument("--dataset_root", default=os.path.join(_ROOT, "graphvis_fb15k_data", "datasets"))
    ap.add_argument("--split", default="stage2_test")
    ap.add_argument("--out", default=os.path.join(_ROOT, "graphvis_ckpts", "neighbor_ranking.json"))
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    methods = args.methods.split(",")

    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(args.dataset_root, args.split))
    idx = sorted(np.random.RandomState(42).choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist())
    queries = [{"head": ds[i]["head"], "relation": _lastseg(ds[i]["relation"]),
                "tail": ds[i]["tail"], "query_id": ds[i]["query_id"]} for i in idx]
    logger.info("Выборка: %d запросов", len(queries))

    logger.info("Считаю статистики KG …")
    stats = build_stats(args.data_dir)
    # Эмбеддинги отношений нужны только query-aware эмбеддинг-методам.
    needs_emb = any(m in methods for m in ("relation_sim", "mmr", "hybrid"))
    relemb = build_rel_embeddings(stats["rels"]) if needs_emb else {}

    textlp = make_textlp(args.textlp, args.model)

    # Собираем пул соседей один раз на голову (большой k), tail исключён.
    neo = _cfg.Neo4jConfig.from_env()
    rcfg = _cfg.RetrieverConfig(neo4j=neo, k_neighbors=BIG_K, include_neighbor_edges=False)
    pools = {}
    with FB15k237GraphRetriever(rcfg) as retr:
        for q in queries:
            sg = retr.get_head_subgraph(q["head"], q["relation"], q["tail"], split="test")
            pools[q["query_id"]] = candidate_edges(sg, q["head"]) if "error" not in sg else []

    # ── оценка ──
    report = {}
    for method in methods:
        for k in ks:
            agg = collections.defaultdict(list)
            tl_hit, tl_tot = 0, 0
            for qi, q in enumerate(queries):
                cand = pools[q["query_id"]]
                if not cand:
                    continue
                order = rank_indices(method, cand, q["relation"], stats, relemb)
                sel = [cand[i] for i in order[:k]]
                has_q, pmi_sup, ent, red, hub = set_metrics(sel, q["relation"], stats)
                agg["has_query_rel"].append(has_q)
                agg["pmi_support"].append(pmi_sup)
                agg["rel_entropy"].append(ent)
                agg["redundancy"].append(red)
                agg["hub_frac"].append(hub)
                if textlp is not None and qi < args.textlp_n:
                    pred = textlp(serialize(q["head"], q["relation"], sel))
                    tl_hit += int(normalize_answer(q["tail"]) in normalize_answer(pred))
                    tl_tot += 1
            row = {m: float(np.mean(v)) for m, v in agg.items()}
            if tl_tot:
                row["textlp_acc"] = tl_hit / tl_tot
                row["textlp_n"] = tl_tot
            report[f"{method}@{k}"] = row
            logger.info("%s@%d: %s", method, k, {m: round(v, 3) for m, v in row.items()})

    # ── печать ──
    cols = ["has_query_rel", "pmi_support", "rel_entropy", "redundancy", "hub_frac"]
    if textlp is not None:
        cols.append("textlp_acc")
    print("\n" + "=" * 88)
    print(f"{'method@k':<26}" + "".join(f"{c:>13}" for c in cols))
    print("-" * 88)
    for key in report:
        r = report[key]
        print(f"{key:<26}" + "".join(f"{r.get(c, float('nan')):>13.3f}" for c in cols))
    print("=" * 88)
    sel_metric = "textlp_acc" if textlp is not None else "pmi_support"
    best = max(report, key=lambda k: report[k].get(sel_metric, -1))
    print(f"Лучший по «{sel_metric}»: {best} = {report[best].get(sel_metric):.3f}")
    if textlp is not None:
        print("textlp_acc — это ВЕРХНЯЯ ГРАНИЦА визуального метода при данном ранжировании.")
    else:
        print("Для окончательного выбора запустите с --textlp hf (верхняя граница).")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("Сохранено:", args.out)


if __name__ == "__main__":
    main()
