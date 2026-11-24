"""Model-free проверка candidate-aware подграфа с путями (path_retriever).

Ничего не обучает и не требует GPU/Neo4j. На ТЕХ ЖЕ запросах, что и общий тест
статьи (индексы берутся из test_details_rank_*.json чекпойнтов), считает:

  1. покрытие: как часто золотой хвост попадает в N показанных кандидатов
     (предотбор freq — без структуры; cn_freq — со структурой);
  2. размер подграфа: число посредников, узлов, рёбер (медиана / p90);
  3. стоимость: токены текстовых кодировок (токенизатор Qwen2.5-VL) против
     edge-list основной статьи и 400 визуальных токенов картинки 560×560;
  4. потолок сигнала: filtered MRR эвристик по тем же 50 кандидатам, включая
     «то, что реально есть на картинке» (показанные пути, ≤ M на кандидата),
     рядом с MRR обученных text/visual из статьи;
  5. комплементарность: ловит ли структура запросы, на которых text-модель
     промахивается.

    python scripts/path_subgraph_check.py
    python scripts/path_subgraph_check.py --render 6 --out tmp/path_subgraph
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pyarrow as pa

from path_retriever import (PathGraph, PathSubgraphConfig, encode_cn,
                            encode_paths, render)


def load_test(path: str):
    t = pa.ipc.open_stream(path).read_all()
    return t.select(["head", "relation", "tail", "neighbors_text"]).to_pylist()


def load_details(path: str) -> dict:
    return {d["idx"]: d for d in json.load(open(path, encoding="utf-8"))}


def rank_of(scores: dict, gold: str, others: set, realistic: bool) -> float:
    """Правило training/rank_eval: строгое «>», отфильтрованные не считаются."""
    g = scores[gold]
    higher = sum(1 for c, s in scores.items() if c != gold and c not in others and s > g)
    if not realistic:
        return 1 + higher
    ties = sum(1 for c, s in scores.items() if c != gold and c not in others and s == g)
    return 1 + higher + ties / 2


def summarize(ranks) -> dict:
    r = np.asarray(ranks, dtype=float)
    return {"MRR": round(float(np.mean(1 / r)), 4),
            "H@1": round(float(np.mean(r <= 1) * 100), 2),
            "H@10": round(float(np.mean(r <= 10) * 100), 2),
            "n": int(len(r))}


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "FB15k-237"))
    ap.add_argument("--test", default=os.path.join(
        _ROOT, "graphvis_scaling/k4/datasets/stage2_test/data-00000-of-00001.arrow"))
    ap.add_argument("--ckpt_root", default=os.path.join(_ROOT, "graphvis_ckpts_scaling_common"))
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/qwen_tok/tokenizer.json"))
    ap.add_argument("--configs", default="5x3,10x3,20x3,10x1",
                    help="N×M: N показанных кандидатов × M путей на кандидата")
    ap.add_argument("--render", type=int, default=4, help="сколько примеров отрисовать")
    ap.add_argument("--out", default=os.path.join(_ROOT, "tmp/path_subgraph"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load_test(args.test)
    det = {}
    for k in (4, 16):
        for cond in ("text", "visual"):
            p = os.path.join(args.ckpt_root, f"k{k}", f"ctx_{cond}")
            f = [x for x in os.listdir(p) if x.startswith("test_details_rank")] if os.path.isdir(p) else []
            if f:
                det[f"{cond}@K{k}"] = load_details(os.path.join(p, sorted(f)[0]))
    idx = sorted(det["text@K4"].keys())
    print(f"запросов общего теста: {len(idx)}; модели: {sorted(det)}")

    g = PathGraph(args.data_dir, split="test")
    tok = None
    if os.path.exists(args.tokenizer):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
    ntok = (lambda s: len(tok.encode(s).ids)) if tok else (lambda s: math.ceil(len(s) / 4))

    report = {"n_queries": len(idx), "tokenizer": "qwen2.5-vl" if tok else "chars/4"}

    # ── имена: все ли головы/хвосты есть в графе ──
    miss_h = sum(1 for i in idx if rows[i]["head"] not in g.adj)
    miss_t = sum(1 for i in idx if rows[i]["tail"] not in g.adj)
    report["names_missing"] = {"head": miss_h, "tail": miss_t}

    # ── 4. эвристики по полному набору кандидатов ──
    heur = collections.defaultdict(lambda: {"opt": [], "real": []})
    per_query_cnfreq_rank = {}
    gold_direct, nongold_direct = [], []
    for i in idx:
        h, r, gold = rows[i]["head"], rows[i]["relation"], rows[i]["tail"]
        cands = g.candidates(r, gold, 50)
        others = g.filtered_out(h, r, gold)
        freq = g.rel2tails.get(r, collections.Counter())
        cn, aa = {}, {}
        for c in cands:
            xs = g.common(h, c, r, gold)
            cn[c] = len(xs)
            aa[c] = sum(1 / math.log(max(g.degree(x), 2)) for x in xs)
            if c not in others:
                d = len(g._pair_edges(h, c, h, r, gold))
                (gold_direct if c == gold else nongold_direct).append(d > 0)
        scores = {
            "freq (без контекста)": {c: freq[c] for c in cands},
            "common neighbors": cn,
            "CN + freq": {c: cn[c] * 1e6 + freq[c] for c in cands},
            "Adamic-Adar + freq": {c: aa[c] * 1e6 + freq[c] for c in cands},
        }
        for name, sc in scores.items():
            heur[name]["opt"].append(rank_of(sc, gold, others, False))
            heur[name]["real"].append(rank_of(sc, gold, others, True))
        per_query_cnfreq_rank[i] = heur["CN + freq"]["opt"][-1]

        # «то, что есть на картинке»: показанные пути ≤ M; у непоказанных 0 путей,
        # но частотный приор сохраняется (модель знает его и без картинки)
        for conf in ("10x3", "20x3", "20x10"):
            n, m = map(int, conf.split("x"))
            for pre in ("freq", "cn_freq"):
                cfg = PathSubgraphConfig(n_show=n, max_paths_per_cand=m, preselect=pre)
                sg = g.build(h, r, gold, cfg)   # пул без золотого (top_candidates)
                sc = {c: (sg.cn_shown.get(c, 0) + (1 if sg.direct.get(c) else 0)) * 1e6 + freq[c]
                      for c in cands}
                key = f"видимое на картинке {conf}, предотбор {pre}"
                heur[key]["opt"].append(rank_of(sc, gold, others, False))
                heur[key]["real"].append(rank_of(sc, gold, others, True))

    report["heuristics"] = {k: {"optimistic": summarize(v["opt"]), "realistic": summarize(v["real"])}
                            for k, v in heur.items()}
    report["models_from_paper(optimistic)"] = {
        k: summarize([d[i]["rank"] for i in idx if i in d]) for k, d in det.items()}
    report["direct_edge_rate"] = {"gold": round(float(np.mean(gold_direct)) * 100, 2),
                                  "non_gold": round(float(np.mean(nongold_direct)) * 100, 2)}

    # ── 5. комплементарность ──
    t4 = det["text@K4"]
    miss = [i for i in idx if t4[i]["rank"] > 10]
    hit = [i for i in miss if per_query_cnfreq_rank[i] <= 10]
    both = [i for i in idx if t4[i]["rank"] <= 10 or per_query_cnfreq_rank[i] <= 10]
    report["complementarity"] = {
        "text@K4 miss@10": len(miss),
        "of them CN+freq hit@10": len(hit),
        "oracle(text OR CN+freq) H@10": round(len(both) / len(idx) * 100, 2),
        "text@K4 H@10": round(sum(t4[i]["rank"] <= 10 for i in idx) / len(idx) * 100, 2),
    }

    # ── 1–3. покрытие, размер, токены по конфигурациям ──
    base_tok = [ntok(rows[i]["neighbors_text"]) for i in idx]
    report["tokens_edge_list_K4_context_block"] = round(float(np.mean(base_tok)), 1)
    k16 = args.test.replace("/k4/", "/k16/")
    if os.path.exists(k16):
        rows16 = load_test(k16)
        report["tokens_edge_list_K16_context_block"] = round(
            float(np.mean([ntok(rows16[i]["neighbors_text"]) for i in idx])), 1)

    cfg_rep = {}
    for conf in args.configs.split(","):
        n, m = map(int, conf.split("x"))
        for pre in ("freq", "cn_freq"):
            cfg = PathSubgraphConfig(n_show=n, max_paths_per_cand=m, preselect=pre)
            cfg_lab = PathSubgraphConfig(n_show=n, max_paths_per_cand=m, preselect=pre,
                                         label_intermediates=True)
            cov, gold_path, ng_path, n_inter, n_nodes, n_edges = [], [], [], [], [], []
            t_anon, t_lab, t_cn = [], [], []
            for i in idx:
                h, r, gold = rows[i]["head"], rows[i]["relation"], rows[i]["tail"]
                sg = g.build(h, r, gold, cfg)
                cov.append(gold in sg.shown)
                if gold in sg.shown:
                    gold_path.append(sg.cn_shown[gold] > 0)
                ng_path += [sg.cn_shown[c] > 0 for c in sg.shown if c != gold]
                n_inter.append(len(sg.inter))
                n_nodes.append(1 + len(sg.shown) + len(sg.inter))
                n_edges.append(len(sg.edges))
                t_anon.append(ntok(encode_paths(sg, cfg)))
                t_lab.append(ntok(encode_paths(sg, cfg_lab)))
                t_cn.append(ntok(encode_cn(sg)))
            cfg_rep[f"{conf} {pre}"] = {
                "gold_in_shown_%": round(float(np.mean(cov)) * 100, 2),
                "gold_has_path_if_shown_%": round(float(np.mean(gold_path)) * 100, 2) if gold_path else None,
                "nongold_has_path_%": round(float(np.mean(ng_path)) * 100, 2) if ng_path else None,
                "intermediates_median/p90": [pct(n_inter, 50), pct(n_inter, 90)],
                "nodes_median/p90": [pct(n_nodes, 50), pct(n_nodes, 90)],
                "edges_median/p90": [pct(n_edges, 50), pct(n_edges, 90)],
                "tokens_text_paths_anon": round(float(np.mean(t_anon)), 1),
                "tokens_text_paths_labeled": round(float(np.mean(t_lab)), 1),
                "tokens_text_cn": round(float(np.mean(t_cn)), 1),
                "tokens_visual_560": 400,
            }
    report["configs"] = cfg_rep

    # ── примеры рендера ──
    rng = np.random.RandomState(0)
    samples = []
    for i in rng.choice(idx, size=min(args.render, len(idx)), replace=False):
        i = int(i)
        h, r, gold = rows[i]["head"], rows[i]["relation"], rows[i]["tail"]
        for lab in (False, True):
            cfg = PathSubgraphConfig(n_show=10, max_paths_per_cand=3, preselect="cn_freq",
                                     label_intermediates=lab)
            sg = g.build(h, r, gold, cfg)
            tag = "lab" if lab else "anon"
            p = render(sg, cfg, os.path.join(args.out, f"q{i}_{tag}.png"))
            with open(os.path.join(args.out, f"q{i}_{tag}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Query: ({h}, {r}, ?)  gold={gold}  gold_shown={gold in sg.shown}\n\n")
                f.write(encode_paths(sg, cfg) + "\n\n" + encode_cn(sg) + "\n")
            samples.append(p)
    report["rendered"] = samples

    out_json = os.path.join(args.out, "path_subgraph_check.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("→", out_json)


if __name__ == "__main__":
    main()
