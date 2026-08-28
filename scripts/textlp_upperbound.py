"""Валидная текстовая ВЕРХНЯЯ ГРАНИЦА метода — сколько в отобранных соседях сигнала.

Чем отличается от `--textlp hf` в neighbor_ranking_eval (который дал ~5%):
  • тот зонд = zero-shot open-generation + substring-match → мерил в основном НЕ-
    следование формату необученной модели, а не информативность соседей;
  • здесь — (1) FEW-SHOT промпт (снимает проблему формата) и (2) предсказание
    РАНЖИРОВАНИЕМ по правдоподобию среди type-constrained кандидатов-хвостов
    (как в filtered-KGC), с метриками MRR/Hits@1/3/10.

Это честный потолок: «если сильная ТЕКСТОВАЯ модель по сериализованным соседям не
может отранжировать истинный хвост — картинка (ограниченная читаемостью) не сможет».
Сравнивая методы ранжирования соседей (baseline/pmi/hybrid) по этому потолку,
выбираем тот, что оставляет модели больше всего сигнала.

Кандидаты: хвосты, встречавшиеся у отношения r в train (type-constrained), топ по
частоте (--cand_cap) + золотой. Filtered: прочие истинные хвосты (h,r,·) из
train+valid+test убираются из ранжирования.

Требуется Neo4j; сама оценка — GPU (кроме --dry_run, который проверяет сборку
кандидатов/ранжирования без модели).

Usage:
    python scripts/textlp_upperbound.py --methods baseline_semantic,pmi,hybrid \
        --n 300 --ks 4 --cand_cap 50 --fewshot 3
    python scripts/textlp_upperbound.py --dry_run --n 20 --ks 4   # без GPU
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import sys
import types
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if "graphvis_lp" not in sys.modules:
    _pkg = types.ModuleType("graphvis_lp"); _pkg.__path__ = [_ROOT]
    sys.modules["graphvis_lp"] = _pkg

import numpy as np

import config as _cfg
from retriever import FB15k237GraphRetriever
from utils import normalize_answer
# переиспользуем ранжирование/сериализацию из соседнего скрипта
import neighbor_ranking_eval as nre

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("textlp_ub")


# ── few-shot преамбула: фиксирует формат «Known facts → Question → Answer» ──
_FEWSHOT = [
    ("Known facts about 'Christopher Nolan':\n"
     "- Christopher Nolan --film director of--> Inception\n"
     "- Christopher Nolan --nationality--> United Kingdom\n"
     "Question: Christopher Nolan --profession--> ?", "Film director"),
    ("Known facts about 'Serena Williams':\n"
     "- Serena Williams --sport--> Tennis\n"
     "- Serena Williams --country of nationality--> United States\n"
     "Question: Serena Williams --gender--> ?", "Female"),
    ("Known facts about 'The Beatles':\n"
     "- The Beatles --genre--> Rock music\n"
     "- The Beatles --origin--> Liverpool\n"
     "Question: The Beatles --country--> ?", "United Kingdom"),
]


def build_name_maps(data_dir):
    """mid→name и (по именам): rel→Counter(tail), (head,rel)→{tails} для filtered."""
    mid2name = {}
    p = os.path.join(data_dir, "FB15k_mid2name.txt")
    for line in open(p, encoding="utf-8"):
        a = line.rstrip("\n").split("\t", 1)
        if len(a) == 2:
            mid2name[a[0]] = a[1].strip()

    def name(m):
        return mid2name.get(m, m.split("/")[-1].replace("_", " "))

    rel2tails = collections.defaultdict(collections.Counter)   # rel -> Counter(tail_name)
    hr2tails = collections.defaultdict(set)                    # (head,rel) -> {tail_name}
    for fn in ("train.txt", "valid.txt", "test.txt"):
        fp = os.path.join(data_dir, fn)
        if not os.path.exists(fp):
            continue
        for line in open(fp, encoding="utf-8"):
            a = line.strip().split("\t")
            if len(a) != 3:
                continue
            h, r, t = name(a[0]), nre._lastseg(a[1]), name(a[2])
            hr2tails[(h, r)].add(t)
            if fn == "train.txt":
                rel2tails[r][t] += 1
    return rel2tails, hr2tails


def build_entity_universe(data_dir):
    """(имена всех сущностей, |E|) — для KGT5-протокола: валидация сэмплов и
    ранг несэмплированных = |E|. Сущность = любой h или t из train/valid/test."""
    mid2name = {}
    p = os.path.join(data_dir, "FB15k_mid2name.txt")
    for line in open(p, encoding="utf-8"):
        a = line.rstrip("\n").split("\t", 1)
        if len(a) == 2:
            mid2name[a[0]] = a[1].strip()

    def name(m):
        return mid2name.get(m, m.split("/")[-1].replace("_", " "))

    ents = set()
    for fn in ("train.txt", "valid.txt", "test.txt"):
        fp = os.path.join(data_dir, fn)
        if not os.path.exists(fp):
            continue
        for line in open(fp, encoding="utf-8"):
            a = line.strip().split("\t")
            if len(a) == 3:
                ents.add(a[0]); ents.add(a[2])
    names = {name(m) for m in ents}
    return names, len(ents)


def make_candidates(rq, gold, rel2tails, cap):
    """Type-constrained кандидаты: топ-cap хвостов отношения r по частоте + золотой."""
    cnt = rel2tails.get(rq)
    cands = [t for t, _ in cnt.most_common(cap)] if cnt else []
    if gold not in cands:
        cands.append(gold)
    return cands


# ── скоринг кандидатов текстовой моделью (teacher-forcing, длинно-нормировка) ──
def make_scorer(model_name):
    import torch
    from graphvis_lp.config import TrainConfig
    from graphvis_lp.training.families import detect_family
    from graphvis_lp.training.model import build_model_and_processor

    cfg = TrainConfig(); cfg.model_name = model_name; cfg.use_lora = False
    family = detect_family(cfg.model_name, cfg.model_family)
    model, processor, tokenizer = build_model_and_processor(cfg); model.eval()
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def prefix_text(user_text):
        if family == "qwen":
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False, add_generation_prompt=True)
        return user_text + "\nAnswer:"

    # Скоринг требует ПРАВОГО паддинга: префикс выровнен слева, значит позиции ≥ plen
    # — токены кандидата. Qwen по умолчанию паддит слева → форсируем right.
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"

    # Оконные логиты: считаем lm_head только по последним `keep` позициям (окно
    # ответа), а не по всей длине. Полные логиты [B, ~200, 152k] fp32 = единицы ГБ
    # и OOM на 12 ГБ, особенно для длинных промптов с соседями (hybrid).
    _keep_kw = {"name": None}

    def _forward_keep(ids, attn, keep):
        if _keep_kw["name"] == "__none__":
            return model(input_ids=ids, attention_mask=attn).logits[:, -keep:, :]
        if _keep_kw["name"] is not None:
            return model(input_ids=ids, attention_mask=attn, **{_keep_kw["name"]: keep}).logits
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            try:
                out = model(input_ids=ids, attention_mask=attn, **{kw: keep}).logits
                _keep_kw["name"] = kw
                return out
            except TypeError:
                continue
        _keep_kw["name"] = "__none__"
        return model(input_ids=ids, attention_mask=attn).logits[:, -keep:, :]

    @torch.no_grad()
    def score(user_text, candidates, cand_batch=32):
        pref = prefix_text(user_text)
        plen = int(tokenizer(pref, return_tensors="pt")["input_ids"].shape[1])
        scores = np.full(len(candidates), -1e30, dtype=np.float64)
        for s in range(0, len(candidates), cand_batch):
            chunk = candidates[s:s + cand_batch]
            enc = tokenizer([pref + " " + c for c in chunk],
                            return_tensors="pt", padding=True, truncation=True, max_length=1024)
            ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            L = int(ids.shape[1])
            keep = max(2, L - plen + 1)                        # окно [plen-1 .. L-1]
            logits = _forward_keep(ids, attn, keep).float()    # [B, keep, V] — маленький
            lp = torch.log_softmax(logits[:, :-1], dim=-1)     # предсказывают ids[:, plen:]
            tgt = ids[:, plen:]
            m = attn[:, plen:] > 0
            Lt = min(lp.shape[1], tgt.shape[1])
            lp, tgt, m = lp[:, :Lt], tgt[:, :Lt], m[:, :Lt]
            tok_lp = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            norm = ((tok_lp * m).sum(-1) / m.sum(-1).clamp_min(1)).cpu().numpy()
            for j, v in enumerate(norm):
                scores[s + j] = float(v)
            del logits, lp
        return scores

    return score


def build_user_text(head, rq, selected, fewshot_k):
    body = nre.serialize(head, rq, selected)   # Known facts … Question … Answer with …
    if fewshot_k <= 0:
        return body
    shots = "\n\n".join(f"{q}\nAnswer with the target entity name only.\nAnswer: {a}"
                        for q, a in _FEWSHOT[:fewshot_k])
    return ("Predict the target entity from the known facts. Examples:\n\n"
            f"{shots}\n\nNow answer this one.\n\n{body}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="baseline_semantic,pmi,hybrid")
    ap.add_argument("--ks", default="4")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--cand_cap", type=int, default=50)
    ap.add_argument("--fewshot", type=int, default=3)
    ap.add_argument("--cand_batch", type=int, default=32)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "FB15k-237"))
    ap.add_argument("--dataset_root", default=os.path.join(_ROOT, "graphvis_fb15k_data", "datasets"))
    ap.add_argument("--split", default="stage2_test")
    ap.add_argument("--dry_run", action="store_true", help="без модели: проверить кандидатов/ранжирование")
    ap.add_argument("--out", default=os.path.join(_ROOT, "graphvis_ckpts", "textlp_upperbound.json"))
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    methods = args.methods.split(",")

    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(args.dataset_root, args.split))
    idx = sorted(np.random.RandomState(42).choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist())
    queries = [{"head": ds[i]["head"], "relation": nre._lastseg(ds[i]["relation"]),
                "tail": ds[i]["tail"], "query_id": ds[i]["query_id"]} for i in idx]
    logger.info("Выборка: %d запросов", len(queries))

    logger.info("Статистики KG + карты имён …")
    stats = nre.build_stats(args.data_dir)
    rel2tails, hr2tails = build_name_maps(args.data_dir)
    needs_emb = any(m in methods for m in ("relation_sim", "mmr", "hybrid"))
    relemb = nre.build_rel_embeddings(stats["rels"]) if needs_emb else {}

    # пул соседей на голову (большой k), tail исключён
    neo = _cfg.Neo4jConfig.from_env()
    rcfg = _cfg.RetrieverConfig(neo4j=neo, k_neighbors=nre.BIG_K, include_neighbor_edges=False)
    pools = {}
    with FB15k237GraphRetriever(rcfg) as retr:
        for q in queries:
            sg = retr.get_head_subgraph(q["head"], q["relation"], q["tail"], split="test")
            pools[q["query_id"]] = nre.candidate_edges(sg, q["head"]) if "error" not in sg else []

    # средний размер кандидатного набора (диагностика)
    cand_sizes = [len(make_candidates(q["relation"], q["tail"], rel2tails, args.cand_cap)) for q in queries]
    logger.info("Кандидатов на запрос: медиана %.0f, макс %d",
                float(np.median(cand_sizes)), int(np.max(cand_sizes)))

    if args.dry_run:
        # Проверяем сборку без модели: доля запросов, где золотой попал в кандидаты,
        # и что ранжирование соседей отдаёт непустой топ-k.
        for method in methods:
            for k in ks:
                gold_in, nonempty = 0, 0
                for q in queries:
                    cand = pools[q["query_id"]]
                    if method in ("none", "leak"):
                        sel = [] if method == "none" else [{"nb": q["tail"], "rel": q["relation"], "dir": "out"}]
                    else:
                        if not cand:
                            continue
                        order = nre.rank_indices(method, cand, q["relation"], stats, relemb)
                        sel = [cand[i] for i in order[:k]]
                    cands = make_candidates(q["relation"], q["tail"], rel2tails, args.cand_cap)
                    gold_in += int(q["tail"] in cands)
                    nonempty += int(len(sel) > 0)
                n = len(queries)
                logger.info("[dry] %s@%d: gold∈cands=%.2f, есть top-k=%.2f (n=%d)",
                            method, k, gold_in / n, nonempty / n, n)
        # пример промпта
        ex = queries[0]
        cand = pools[ex["query_id"]]
        if cand:
            order = nre.rank_indices(methods[0], cand, ex["relation"], stats, relemb)
            sel = [cand[i] for i in order[:ks[0]]]
            print("\n--- пример few-shot промпта ---\n")
            print(build_user_text(ex["head"], ex["relation"], sel, args.fewshot)[:1200])
        print("\n[dry_run] сборка кандидатов/ранжирования проверена, модель не грузилась.")
        return

    score = make_scorer(args.model)

    report = {}
    per_q = {}   # key -> {query_id: rank}  (для ПАРНЫХ тестов по общим запросам)
    for method in methods:
        for k in ks:
            ranks = []
            rank_by_q = {}
            for qi, q in enumerate(queries):
                if (qi + 1) % 50 == 0:
                    logger.info("  %s@%d: %d/%d запросов…", method, k, qi + 1, len(queries))
                cand = pools[q["query_id"]]
                if method == "none":
                    sel = []                       # baseline: только head+relation, без соседей
                elif method == "leak":
                    # SANITY: k-1 реальных соседей + факт с ОТВЕТОМ (head --rq--> gold).
                    # Читает ли ридер факты вообще? Работает → MRR≈1.0; нет → ≈none.
                    if cand:
                        order = nre.rank_indices("hybrid", cand, q["relation"], stats, relemb)
                        sel = [cand[i] for i in order[:max(0, k - 1)]]
                    else:
                        sel = []
                    sel = sel + [{"nb": q["tail"], "rel": q["relation"], "dir": "out"}]
                else:
                    if not cand:
                        continue
                    order = nre.rank_indices(method, cand, q["relation"], stats, relemb)
                    sel = [cand[i] for i in order[:k]]
                cands = make_candidates(q["relation"], q["tail"], rel2tails, args.cand_cap)
                if q["tail"] not in cands or len(cands) < 2:
                    continue
                user_text = build_user_text(q["head"], q["relation"], sel, args.fewshot)
                sc = score(user_text, cands, args.cand_batch)
                gpos = cands.index(q["tail"])
                gscore = sc[gpos]
                others = hr2tails.get((q["head"], q["relation"]), set()) - {q["tail"]}
                higher = sum(1 for i, c in enumerate(cands)
                             if i != gpos and c not in others and sc[i] > gscore)
                ranks.append(1 + higher)
                rank_by_q[q["query_id"]] = 1 + higher
            per_q[f"{method}@{k}"] = rank_by_q
            r = np.array(ranks, dtype=float)
            rr = 1.0 / r if len(r) else np.array([0.0])
            row = {
                "MRR": float(rr.mean()),
                "Hits@1": float((r <= 1).mean()) * 100 if len(r) else 0.0,
                "Hits@3": float((r <= 3).mean()) * 100 if len(r) else 0.0,
                "Hits@10": float((r <= 10).mean()) * 100 if len(r) else 0.0,
                "n": int(len(r)),
                "mean_candidates": float(np.mean(cand_sizes)),
            }
            report[f"{method}@{k}"] = row
            logger.info("%s@%d: %s", method, k, {m: round(v, 3) for m, v in row.items()})

    print("\n" + "=" * 78)
    print(f"{'method@k':<24}{'MRR':>8}{'Hits@1':>9}{'Hits@3':>9}{'Hits@10':>9}{'n':>6}")
    print("-" * 78)
    for key, r in report.items():
        print(f"{key:<24}{r['MRR']:>8.3f}{r['Hits@1']:>9.1f}{r['Hits@3']:>9.1f}"
              f"{r['Hits@10']:>9.1f}{r['n']:>6}")
    print("=" * 78)
    if report:
        best = max(report, key=lambda k: report[k]["MRR"])
        print(f"Лучший по MRR: {best} = {report[best]['MRR']:.3f}")

    # ── статзначимость контекста: ПАРНЫЕ тесты (method@k) против baseline none@k ──
    # ΔMRR с bootstrap-95%CI (эффект + неопределённость, а не только точка) и
    # McNemar по Hits@10/@1. CI, включающий 0 → различие НЕ значимо (но это НЕ
    # доказывает эквивалентность — при малом n мощности не хватает даже на эффект
    # размера KGT5-context ~0.05).
    sig = {}
    base_keys = [k for k in per_q if k.startswith("none@")]
    if base_keys:
        rng = np.random.RandomState(42)
        B = 10000
        for base in base_keys:
            kk = base.split("@")[1]
            for key, rbq in per_q.items():
                if key == base or not key.endswith("@" + kk):
                    continue
                common = sorted(set(per_q[base]) & set(rbq))
                if len(common) < 10:
                    continue
                rb = np.array([per_q[base][q] for q in common], dtype=float)
                rm = np.array([rbq[q] for q in common], dtype=float)
                d_mrr = float((1.0 / rm).mean() - (1.0 / rb).mean())
                # парный bootstrap по общим запросам
                n = len(common)
                idx = rng.randint(0, n, size=(B, n))
                boot = (1.0 / rm[idx]).mean(1) - (1.0 / rb[idx]).mean(1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                # McNemar по Hits@10 / Hits@1 (a = контекст попал & base мимо, b = наоборот)
                def _mcnemar(thr):
                    hb, hm = rb <= thr, rm <= thr
                    a = int((hm & ~hb).sum()); b = int((hb & ~hm).sum())
                    if a + b == 0:
                        return a, b, 1.0
                    z = (abs(a - b) - 1) / math.sqrt(a + b)   # continuity-corrected
                    p = math.erfc(abs(z) / math.sqrt(2))
                    return a, b, p
                a10, b10, p10 = _mcnemar(10)
                a1, b1, p1 = _mcnemar(1)
                sig[f"{key} vs {base}"] = {
                    "n_paired": n, "dMRR": d_mrr, "dMRR_CI95": [float(lo), float(hi)],
                    "signif_MRR": bool(lo > 0 or hi < 0),
                    "H@10": {"ctx_win": a10, "base_win": b10, "p": p10},
                    "H@1": {"ctx_win": a1, "base_win": b1, "p": p1},
                }
        if sig:
            print("\n" + "=" * 78)
            print("СТАТЗНАЧИМОСТЬ контекста (парно, vs none; bootstrap B=10000)")
            print("-" * 78)
            for name, s in sig.items():
                verdict = "ЗНАЧИМО" if s["signif_MRR"] else "не значимо (CI включает 0)"
                print(f"{name}  (n={s['n_paired']})")
                print(f"   ΔMRR = {s['dMRR']:+.4f}  95%CI [{s['dMRR_CI95'][0]:+.4f}, "
                      f"{s['dMRR_CI95'][1]:+.4f}]  → {verdict}")
                print(f"   McNemar H@10: контекст+={s['H@10']['ctx_win']} / base+={s['H@10']['base_win']} "
                      f"p={s['H@10']['p']:.3f}   H@1: +{s['H@1']['ctx_win']}/{s['H@1']['base_win']} "
                      f"p={s['H@1']['p']:.3f}")
            print("=" * 78)

    print("Это ВЕРХНЯЯ ГРАНИЦА визуального метода: картинка (ограниченная читаемостью)")
    print("не даст больше, чем текстовая модель извлекает из этих же соседей.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"metrics": report, "significance": sig}, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("Сохранено:", args.out)


if __name__ == "__main__":
    main()
